import glob
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    ImageRef,
    PageItem,
    ProvenanceItem,
    Size,
)
from docling_core.types.io import DocumentStream
from PIL.Image import Image
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns
from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval.utils.utils import (
    add_pages_to_true_doc,
    convert_html_table_into_docling_tabledata,
    crop_bounding_box,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
)

# Get logger
_log = logging.getLogger(__name__)

# Labels to export in HTML visualization
TRUE_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    # Additional
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}

PRED_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    # Additional
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class OmniDocBenchDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    OmniDocBench dataset builder implementing the base dataset builder interface.

    This builder processes the OmniDocBench dataset, which contains document
    layout annotations for a variety of document types.
    """

    def __init__(
        self,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the OmniDocBench dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="OmniDocBench: end-to-end",
            dataset_source=HFSource(repo_id="opendatalab/OmniDocBench"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        self.must_retrieve = True

    def update_gt_into_map(self, gt: List[Dict]) -> Dict[str, Dict]:
        """
        Convert list of annotation items to a map keyed by image path.

        Args:
            gt: List of ground truth annotations

        Returns:
            Dictionary mapping image paths to their annotations
        """
        result = {}
        for item in gt:
            path = item["page_info"]["image_path"]
            result[path] = item
        return result

    def get_filenames(self, omnidocbench_dir: Path) -> List[Tuple[str, str]]:
        """
        Get pairs of image and PDF paths from the dataset directory.

        Args:
            omnidocbench_dir: Path to the OmniDocBench directory

        Returns:
            List of (image_path, pdf_path) tuples
        """
        page_images = sorted(glob.glob(str(omnidocbench_dir / "images/*.jpg")))
        page_pdfs = sorted(glob.glob(str(omnidocbench_dir / "ori_pdfs/*.pdf")))

        assert len(page_images) == len(
            page_pdfs
        ), f"len(page_images)!=len(page_pdfs) => {len(page_images)}!={len(page_pdfs)}"

        return list(zip(page_images, page_pdfs))

    def update_doc_with_gt(
        self,
        gt: Dict,
        true_doc: DoclingDocument,
        page: PageItem,
        page_image: Image,
        page_width: float,
        page_height: float,
    ) -> DoclingDocument:
        """
        Update document with ground truth annotations.

        Args:
            gt: Ground truth annotations
            true_doc: Document to update
            page: Page object
            page_image: Page image
            page_width: Page width
            page_height: Page height

        Returns:
            Updated document
        """
        gt_width = float(gt["page_info"]["width"])
        gt_height = float(gt["page_info"]["height"])

        for item in gt["layout_dets"]:
            label = item["category_type"]
            text = f"&lt;omitted text for {label}&gt;"
            if "text" in item:
                text = item["text"]

            # Find bounding box coordinates
            min_x = item["poly"][0]
            max_x = item["poly"][0]
            min_y = item["poly"][1]
            max_y = item["poly"][1]

            for i in range(0, 4):
                min_x = min(min_x, item["poly"][2 * i])
                max_x = max(max_x, item["poly"][2 * i])
                min_y = min(min_y, item["poly"][2 * i + 1])
                max_y = max(max_y, item["poly"][2 * i + 1])

            # Create bounding box
            bbox = BoundingBox(
                l=min_x * page_width / gt_width,
                r=max_x * page_width / gt_width,
                t=min_y * page_height / gt_height,
                b=max_y * page_height / gt_height,
                coord_origin=CoordOrigin.TOPLEFT,
            )

            # Create provenance
            prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))

            # Crop the image element - use page directly since we've updated the signature
            img = crop_bounding_box(page_image=page_image, page=page, bbox=bbox)

            # Add element to document based on label
            if label == "title":
                true_doc.add_heading(text=text, orig=text, level=1, prov=prov)

            elif label == "text_block":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "text_mask":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "table":
                table_data = convert_html_table_into_docling_tabledata(
                    table_html=item["html"]
                )
                true_doc.add_table(data=table_data, caption=None, prov=prov)

            elif label == "table_caption":
                true_doc.add_text(
                    label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov
                )

            elif label == "table_footnote":
                true_doc.add_text(
                    label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov
                )

            elif label == "table_mask":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "figure":
                uri = from_pil_to_base64uri(img)
                imgref = ImageRef(
                    mimetype="image/png",
                    dpi=72,
                    size=Size(width=img.width, height=img.height),
                    uri=uri,
                )
                true_doc.add_picture(prov=prov, image=imgref)

            elif label == "figure_caption":
                true_doc.add_text(
                    label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov
                )

            elif label == "figure_footnote":
                true_doc.add_text(
                    label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov
                )

            elif label == "equation_isolated":
                true_doc.add_text(
                    label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov
                )

            elif label == "equation_caption":
                true_doc.add_text(
                    label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov
                )

            elif label == "code_txt":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "abandon":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "need_mask":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "header":
                true_doc.add_text(
                    label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov
                )

            elif label == "footer":
                true_doc.add_text(
                    label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov
                )

            elif label == "reference":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "page_footnote":
                true_doc.add_text(
                    label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov
                )

            elif label == "page_number":
                true_doc.add_text(
                    label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov
                )

            else:
                _log.error(f"label {label} is not assigned!")

        return true_doc

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Iterate through the dataset and yield DatasetRecord objects.

        Yields:
            DatasetRecord objects
        """
        if not self.retrieved and self.must_retrieve:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None

        # Load the ground truth
        with open(self.dataset_local_path / "OmniDocBench.json", "r") as fr:
            gt = json.load(fr)

        gt = self.update_gt_into_map(gt)

        # Create visualization directory if needed
        viz_dir = self.target / "vizualisations"
        viz_dir.mkdir(exist_ok=True)

        # Get all file paths
        page_tuples = self.get_filenames(self.dataset_local_path)
        total_items = len(page_tuples)

        # Apply index range
        begin, end = self.get_effective_indices(total_items)
        page_tuples = page_tuples[begin:end]
        selected_items = len(page_tuples)

        # Log stats
        self.log_dataset_stats(total_items, selected_items)

        for page_tuple in tqdm(
            page_tuples,
            total=selected_items,
            ncols=128,
            desc="Processing files for OmniDocBench",
        ):
            jpg_path = page_tuple[0]
            pdf_path = Path(page_tuple[1])

            # Check if ground truth exists for this image
            jpg_basename = os.path.basename(jpg_path)
            if jpg_basename not in gt:
                _log.error(f"Did not find ground-truth for {jpg_basename}")
                continue

            gt_doc = gt[jpg_basename]

            # Create the ground truth Document
            true_doc = DoclingDocument(name=f"ground-truth {jpg_basename}")
            true_doc, true_page_images = add_pages_to_true_doc(
                pdf_path=pdf_path, true_doc=true_doc, image_scale=2.0
            )

            assert len(true_page_images) == 1, "len(true_page_images)==1"

            # The true_doc.pages is a dict with the page numbers as indices starting at 1
            page_width = true_doc.pages[1].size.width
            page_height = true_doc.pages[1].size.height

            # Update document with ground truth
            true_doc = self.update_doc_with_gt(
                gt=gt_doc,
                true_doc=true_doc,
                page=true_doc.pages[1],
                page_image=true_page_images[0],
                page_width=page_width,
                page_height=page_height,
            )

            # Extract images from the ground truth document
            true_doc, true_pictures, true_page_images = extract_images(
                document=true_doc,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
            )

            # Get PDF as binary data
            pdf_bytes = get_binary(pdf_path)
            pdf_stream = DocumentStream(name=pdf_path.name, stream=BytesIO(pdf_bytes))

            # Create dataset record
            record = DatasetRecord(
                doc_id=str(os.path.basename(jpg_path)),
                doc_hash=get_binhash(pdf_bytes),
                ground_truth_doc=true_doc,
                ground_truth_pictures=true_pictures,
                ground_truth_page_images=true_page_images,
                original=pdf_stream,
                mime_type="application/pdf",
            )

            yield record
