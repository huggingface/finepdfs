import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import ContentLayer, DocItem, DoclingDocument
from docling_core.types.io import DocumentStream
from pydantic import ValidationError

from docling_eval.datamodels.cvat_types import (
    AnnotatedDoc,
    AnnotatedImage,
    AnnotationBBox,
    AnnotationOverview,
    BenchMarkDirs,
    DocLinkLabel,
    TableComponentLabel,
)
from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.utils.utils import get_binhash, insert_images_from_pil

# Configure logging
_log = logging.getLogger(__name__)


class CvatPreannotationBuilder:
    """
    Builder class for creating CVAT preannotations from a dataset.

    This class takes an existing dataset (ground truth or predictions) and prepares
    files and preannotations for CVAT annotation.
    """

    def __init__(
        self,
        dataset_source: Path,
        target: Path,
        bucket_size: int = 200,
        use_predictions: bool = False,
    ):
        """
        Initialize the CvatPreannotationBuilder.

        Args:
            dataset_source: Directory containing the source dataset
            target: Directory where CVAT preannotations will be saved
            bucket_size: Number of documents per bucket for CVAT tasks
        """
        self.source_dir = dataset_source
        self.target_dir = target
        self.bucket_size = bucket_size
        self.benchmark_dirs = BenchMarkDirs()
        self.benchmark_dirs.set_up_directory_structure(
            source=dataset_source, target=target
        )
        self.overview = AnnotationOverview()
        self.use_predictions = use_predictions

    def _export_from_dataset(self) -> AnnotationOverview:
        """
        Export supplementary files from the dataset.

        This method extracts document data from the dataset and creates the necessary
        files needed for CVAT annotation, ensuring proper file paths for ground truth.

        Returns:
            AnnotationOverview object with dataset information
        """
        # Load dataset from parquet files
        test_files = sorted(
            glob.glob(str(self.benchmark_dirs.source_dir / "*.parquet"))
        )
        ds = load_dataset("parquet", data_files={"test": test_files})

        if ds is None:
            raise ValueError(
                f"Failed to load dataset from {self.benchmark_dirs.source_dir}"
            )

        ds_selection = ds["test"]
        overview = AnnotationOverview()

        for data in ds_selection:
            try:
                record: DatasetRecord
                if self.use_predictions:
                    try:
                        record = DatasetRecordWithPrediction.model_validate(data)
                        document = record.predicted_doc
                        pictures = record.predicted_pictures
                        page_images = record.predicted_page_images
                    except ValidationError:
                        _log.error(
                            "The provided input dataset does not have predictions. Set use_predictions = False."
                        )
                        raise
                else:
                    try:
                        # Load as a regular ground truth record
                        record = DatasetRecord.model_validate(data)
                        document = record.ground_truth_doc
                        pictures = record.ground_truth_pictures
                        page_images = record.ground_truth_page_images
                    except ValidationError:
                        _log.error(
                            "The provided input dataset does not contain valid records."
                        )
                        raise

                assert record is not None
                assert document is not None

                # Use document ID as name - ensure it's a consistent identifier
                doc_name = f"{record.doc_id}"
                doc_hash = record.doc_hash or ""

                if doc_hash == "" and record.original is not None:
                    bin_data = record.original
                    if isinstance(bin_data, Path):
                        with open(bin_data, "rb") as f:
                            doc_hash = get_binhash(f.read())
                    elif isinstance(bin_data, DocumentStream):
                        bin_data.stream.seek(0)
                        doc_hash = get_binhash(bin_data.stream.read())
                        bin_data.stream.seek(0)

                # Insert images into document
                document = insert_images_from_pil(document, pictures, page_images)

                # Write ground truth document to JSON - this is the ONLY JSON file we'll save
                # The name must be consistent for later retrieval
                json_file = self.benchmark_dirs.json_true_dir / f"{doc_name}.json"
                document.save_as_json(filename=json_file)
                _log.info(f"Saved ground truth document to {json_file}")

                # Get MIME type and determine file extension
                mime_type = record.mime_type
                bin_ext = ".bin"  # Default extension

                if mime_type == "application/pdf":
                    bin_ext = ".pdf"
                elif mime_type == "image/png":
                    bin_ext = ".png"
                elif mime_type in ["image/jpg", "image/jpeg"]:
                    bin_ext = ".jpg"
                else:
                    _log.warning(
                        f"Unsupported mime-type {mime_type}, using .bin extension"
                    )

                # Write binary document
                bin_name = f"{doc_hash}{bin_ext}"
                bin_file = self.benchmark_dirs.bins_dir / bin_name

                if record.original is not None:
                    bin_data = record.original
                    with open(bin_file, "wb") as fw:
                        if isinstance(bin_data, Path):
                            with open(bin_data, "rb") as fr:
                                fw.write(fr.read())
                        elif isinstance(bin_data, DocumentStream):
                            bin_data.stream.seek(0)
                            fw.write(bin_data.stream.read())
                            bin_data.stream.seek(0)

                # Add to overview - using consistent file paths
                overview.doc_annotations.append(
                    AnnotatedDoc(
                        mime_type=mime_type,
                        document_file=json_file,  # The one and only JSON file
                        bin_file=bin_file,
                        doc_hash=doc_hash,
                        doc_name=doc_name,
                    )
                )

            except Exception as e:
                _log.error(f"Error processing record: {str(e)}")
                raise
                # continue

        return overview

    def _create_project_properties(self) -> None:
        """
        Create CVAT project properties file.

        This file defines the label categories and their attributes
        for the CVAT annotation project.
        """
        results = []

        # Add DocItemLabel properties
        for item in DocItemLabel:
            r, g, b = DocItemLabel.get_color(item)

            results.append(
                {
                    "name": item.value,
                    "color": rgb_to_hex(r, g, b),
                    "type": "rectangle",
                    "attributes": [],
                }
            )

            # Add specific attributes for certain labels
            if item in [DocItemLabel.LIST_ITEM, DocItemLabel.SECTION_HEADER]:
                results[-1]["attributes"].append(
                    {
                        "name": "level",
                        "input_type": "number",
                        "mutable": True,
                        "values": ["1", "10", "1"],
                        "default_value": "1",
                    }
                )

            if item == DocItemLabel.FORMULA:
                results[-1]["attributes"].append(
                    {
                        "name": "latex",
                        "mutable": True,
                        "input_type": "text",
                        "values": [""],
                        "default_value": "",
                    }
                )

            if item == DocItemLabel.CODE:
                results[-1]["attributes"].append(
                    {
                        "name": "code",
                        "mutable": True,
                        "input_type": "text",
                        "values": [""],
                        "default_value": "",
                    }
                )

            if item == DocItemLabel.PICTURE:
                results[-1]["attributes"].append(
                    {
                        "name": "json",
                        "mutable": True,
                        "input_type": "text",
                        "values": [""],
                        "default_value": "",
                    }
                )

        # Add TableComponentLabel properties
        for table_item in TableComponentLabel:
            r, g, b = TableComponentLabel.get_color(table_item)

            results.append(
                {
                    "name": table_item.value,
                    "color": rgb_to_hex(r, g, b),
                    "type": "rectangle",
                    "attributes": [],
                }
            )

        # Add DocLinkLabel properties
        for link_item in DocLinkLabel:
            r, g, b = DocLinkLabel.get_color(link_item)

            results.append(
                {
                    "name": link_item.value,
                    "color": rgb_to_hex(r, g, b),
                    "type": "polyline",
                    "attributes": [],
                }
            )

        _log.info(
            f"Writing project description: {str(self.benchmark_dirs.project_desc_file)}"
        )
        with open(str(self.benchmark_dirs.project_desc_file), "w") as fw:
            json.dump(results, fw, indent=2)

    def _create_preannotation_files(self) -> None:
        """
        Create CVAT preannotation files.

        This method processes each document in the overview and generates
        CVAT annotation XML files and image files for each page, with the
        correct file paths for later processing.
        """
        # Dictionary to store annotations by bucket ID
        bucket_annotations: Dict[int, List[str]] = {}

        img_id = 0
        for doc_overview in self.overview.doc_annotations:
            try:
                # Load document from the saved JSON file
                doc = DoclingDocument.load_from_json(doc_overview.document_file)

                # Process each page in the document
                for page_no, page in doc.pages.items():
                    img_id += 1

                    # Calculate bucket ID consistently for both folder and XML naming
                    bucket_id = (img_id - 1) // self.bucket_size
                    bucket_dir = self.benchmark_dirs.tasks_dir / f"task_{bucket_id:02}"
                    os.makedirs(bucket_dir, exist_ok=True)

                    # Initialize bucket annotation list if needed
                    if bucket_id not in bucket_annotations:
                        bucket_annotations[bucket_id] = []

                    # Use document name and hash for consistent naming
                    doc_name = doc_overview.doc_name
                    doc_hash = doc_overview.doc_hash

                    # Create unique filename for the page image
                    filename = f"doc_{doc_hash}_page_{page_no:06}.png"

                    # Create annotated image record - using the SAME document file path
                    annotated_image = AnnotatedImage(
                        img_id=img_id,
                        mime_type=doc_overview.mime_type,
                        document_file=doc_overview.document_file,  # Use the consistent file path
                        bin_file=doc_overview.bin_file,
                        doc_name=doc_name,
                        doc_hash=doc_hash,
                        bucket_dir=bucket_dir,
                        img_file=bucket_dir / filename,
                    )

                    # Save page image to both task directory and page images directory
                    page_img_file = self.benchmark_dirs.page_imgs_dir / filename
                    annotated_image.page_img_files = [page_img_file]

                    # Extract and save page image
                    page_image_ref = page.image
                    if page_image_ref is not None:
                        page_image = page_image_ref.pil_image

                        if page_image is not None:
                            page_image.save(str(annotated_image.img_file))
                            page_image.save(str(annotated_image.page_img_files[0]))

                            annotated_image.img_w = page_image.width
                            annotated_image.img_h = page_image.height
                            annotated_image.page_nos = [page_no]

                            # Add to overview using filename as key
                            self.overview.img_annotations[filename] = annotated_image
                        else:
                            _log.warning(
                                f"Missing pillow image for page {page_no}, skipping..."
                            )
                            continue
                    else:
                        _log.warning(
                            f"Missing image reference for page {page_no}, skipping..."
                        )
                        continue

                    # Extract bounding boxes for annotation
                    page_bboxes = self._extract_page_bounding_boxes(
                        doc, page_no, annotated_image.img_w, annotated_image.img_h
                    )

                    annotated_image.bbox_annotations = page_bboxes
                    bucket_annotations[bucket_id].append(annotated_image.to_cvat())

            except Exception as e:
                _log.error(
                    f"Error processing document {doc_overview.doc_name}: {str(e)}"
                )
                continue

        # Write preannotation XML files for each bucket
        for bucket_id, annotations in bucket_annotations.items():
            self._write_preannotation_file(bucket_id, annotations)

        # Save overview with all the properly set file paths
        self.overview.save_as_json(self.benchmark_dirs.overview_file)
        _log.info(
            f"Saved annotation overview to {self.benchmark_dirs.overview_file} with {len(self.overview.img_annotations)} images"
        )

    def _extract_page_bounding_boxes(
        self, doc: DoclingDocument, page_no: int, img_w: int, img_h: int
    ) -> List[AnnotationBBox]:
        """
        Extract bounding boxes for all items on a page.

        Args:
            doc: The document to extract from
            page_no: Page number to extract boxes from
            img_w: Width of the image
            img_h: Height of the image

        Returns:
            List of AnnotationBBox objects
        """
        page_bboxes: List[AnnotationBBox] = []
        for item, _ in doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        ):
            assert isinstance(item, DocItem)

            for prov in item.prov:
                if page_no == prov.page_no:
                    page_w = doc.pages[prov.page_no].size.width
                    page_h = doc.pages[prov.page_no].size.height

                    # Convert document coordinates to image coordinates
                    page_bbox = prov.bbox.to_top_left_origin(page_height=page_h)

                    page_bboxes.append(
                        AnnotationBBox(
                            bbox_id=len(page_bboxes),
                            label=item.label,
                            bbox=BoundingBox(
                                l=page_bbox.l / page_w * img_w,
                                r=page_bbox.r / page_w * img_w,
                                t=page_bbox.t / page_h * img_h,
                                b=page_bbox.b / page_h * img_h,
                                coord_origin=page_bbox.coord_origin,
                            ),
                        )
                    )
        return page_bboxes

    def _write_preannotation_file(self, bucket_id: int, annotations: List[str]) -> None:
        """
        Write CVAT preannotation XML file for a bucket.

        Args:
            bucket_id: ID of the bucket
            annotations: List of annotation strings to include
        """
        preannot_file = (
            self.benchmark_dirs.tasks_dir / f"task_{bucket_id:02}_preannotate.xml"
        )

        with open(preannot_file, "w") as fw:
            fw.write('<?xml version="1.0" encoding="utf-8"?>\n')
            fw.write("<annotations>\n")
            for annotation in annotations:
                fw.write(f"{annotation}\n")
            fw.write("</annotations>\n")

        _log.info(
            f"Created preannotation file {preannot_file} with {len(annotations)} annotations"
        )

    def prepare_for_annotation(self) -> None:
        """
        Prepare all necessary files for CVAT annotation.

        This is the main method to call to prepare a dataset for CVAT annotation.
        """
        _log.info(f"Preparing dataset from {self.source_dir} for CVAT annotation")
        self._create_project_properties()
        self.overview = self._export_from_dataset()
        self._create_preannotation_files()
        _log.info(f"CVAT annotation preparation complete in {self.target_dir}")


def rgb_to_hex(r, g, b):
    """
    Converts RGB values to a HEX color code.

    Args:
        r (int): Red value (0-255)
        g (int): Green value (0-255)
        b (int): Blue value (0-255)

    Returns:
        str: HEX color code (e.g., "#RRGGBB")
    """
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB values must be in the range 0-255")

    return f"#{r:02X}{g:02X}{b:02X}"
