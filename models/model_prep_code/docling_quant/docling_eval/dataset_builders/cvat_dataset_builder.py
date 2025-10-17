import glob
import json
import logging
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

import xmltodict  # type: ignore[import]
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    FloatingItem,
    GraphData,
    ImageRef,
    PageItem,
    ProvenanceItem,
    TableData,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.page import SegmentedPage, SegmentedPdfPage, TextCellUnit
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm

from docling_eval.datamodels.cvat_types import (
    AnnotatedImage,
    AnnotationOverview,
    BenchMarkDirs,
)
from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns, EvaluationModality
from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
from docling_eval.utils.utils import (
    crop_bounding_box,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
)

# Configure logging
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
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class CvatDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    Dataset builder that creates a dataset from CVAT annotations.

    This class processes CVAT annotations and creates a new ground truth dataset.
    """

    def __init__(
        self,
        name: str,
        dataset_source: Path,
        target: Path,
        split: str = "test",
    ):
        """
        Initialize the CvatDatasetBuilder.

        Args:
            name: Name of the dataset
            cvat_source_dir: Directory containing CVAT annotations
            target: Path where the new dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing
            end_index: End index for processing
            do_visualization: Whether to generate HTML visualizations
        """
        super().__init__(
            name=name,
            dataset_source=dataset_source,
            target=target,
            dataset_local_path=None,
            split=split,
        )
        self.must_retrieve = False
        self.benchmark_dirs = BenchMarkDirs()
        self.benchmark_dirs.set_up_directory_structure(
            source=dataset_source, target=dataset_source
        )

    def unzip_annotation_files(self, output_dir: Path) -> List[Path]:
        """
        Unzip annotation files to the specified directory.

        Args:
            output_dir: Directory to unzip files to

        Returns:
            List of paths to unzipped files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        unzipped_files = []
        zip_files = sorted(
            glob.glob(str(self.benchmark_dirs.annotations_zip_dir / "*.zip"))
        )

        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zf:
                for file_name in zf.namelist():
                    # Resolve filename collisions
                    original_file_name = file_name
                    file_path = os.path.join(output_dir, file_name)
                    base, ext = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(file_path):
                        # Append a numeric suffix to resolve collisions
                        file_name = f"{base}_{counter}{ext}"
                        file_path = os.path.join(output_dir, file_name)
                        counter += 1

                    # Extract file and add to the list
                    with open(file_path, "wb") as f:
                        f.write(zf.read(original_file_name))
                    unzipped_files.append(Path(file_path))

        return unzipped_files

    def get_annotation_files(self) -> List[Path]:
        """
        Get annotation files from zip files or directly from directory.

        Returns:
            List of paths to annotation files
        """
        zip_files = sorted(
            glob.glob(str(self.benchmark_dirs.annotations_zip_dir / "*.zip"))
        )

        if len(zip_files) > 0:
            _log.info(f"Found {len(zip_files)} zip files")

            existing_xml_files = sorted(
                glob.glob(str(self.benchmark_dirs.annotations_xml_dir / "*.xml"))
            )
            _log.info(
                f"Found {len(existing_xml_files)} existing XML files, clearing..."
            )

            for xml_file in existing_xml_files:
                os.remove(xml_file)

            xml_files = self.unzip_annotation_files(
                self.benchmark_dirs.annotations_xml_dir
            )
        else:
            xml_files = sorted(self.benchmark_dirs.annotations_xml_dir.glob("*.xml"))

        _log.info(f"Processing {len(xml_files)} XML annotation files")
        return xml_files

    def save_to_disk(
        self,
        chunk_size: int = 80,
        max_num_chunks: int = sys.maxsize,
        do_visualization: bool = False,
    ) -> None:

        if do_visualization:
            html_output_dir = self.target / "visualizations"
            os.makedirs(html_output_dir, exist_ok=True)

        super().save_to_disk(
            chunk_size, max_num_chunks, do_visualization=do_visualization
        )

    def find_box(
        self, boxes: List[Dict], point: Tuple[float, float]
    ) -> Tuple[int, Dict]:
        """
        Find the box containing a point.

        Args:
            boxes: List of boxes
            point: Point coordinates (x, y)

        Returns:
            Tuple of (box_index, box)
        """
        index = -1
        area = 1e9
        box_result = {}

        for i, box in enumerate(boxes):
            # Ensure box coordinates are valid
            if not (box["l"] < box["r"] and box["t"] < box["b"]):
                continue

            if box["l"] <= point[0] <= box["r"] and box["t"] <= point[1] <= box["b"]:
                current_area = (box["r"] - box["l"]) * (box["b"] - box["t"])
                if index == -1 or current_area < area:
                    area = current_area
                    index = i
                    box_result = box

        if index == -1:
            _log.error(f"Point {point} is not in any bounding-box!")
            return -1, {}

        return index, box_result

    def parse_annotation(self, image_annot: Dict) -> Tuple:
        """
        Parse a CVAT annotation for an image.

        Args:
            image_annot: Annotation data for an image

        Returns:
            Tuple of parsed annotation data
        """
        basename: str = image_annot["@name"]
        keep: bool = False

        boxes: List[Dict] = []
        lines: List[Dict] = []

        reading_order: Dict = {}

        to_captions: List[Dict] = []
        to_footnotes: List[Dict] = []
        to_values: List[Dict] = []

        merges: List[Dict] = []
        group: List[Dict] = []

        if "box" not in image_annot or "polyline" not in image_annot:
            _log.warning(
                f"Skipping {basename} because no `box` nor `polyline` is found"
            )
            return (
                basename,
                keep,
                boxes,
                lines,
                reading_order,
                to_captions,
                to_footnotes,
                to_values,
                merges,
                group,
            )

        # Process boxes
        if isinstance(image_annot["box"], dict):
            boxes = [image_annot["box"]]
        elif isinstance(image_annot["box"], list):
            boxes = image_annot["box"]
        else:
            _log.error(f"Could not get boxes for {basename}")
            return (
                basename,
                keep,
                boxes,
                lines,
                reading_order,
                to_captions,
                to_footnotes,
                to_values,
                merges,
                group,
            )

        # Process polylines
        if isinstance(image_annot["polyline"], dict):
            lines = [image_annot["polyline"]]
        elif isinstance(image_annot["polyline"], list):
            lines = image_annot["polyline"]
        else:
            _log.error(f"Could not get lines for {basename}")
            return (
                basename,
                keep,
                boxes,
                lines,
                reading_order,
                to_captions,
                to_footnotes,
                to_values,
                merges,
                group,
            )

        # Convert box coordinates to floats
        for i, box in enumerate(boxes):
            boxes[i]["b"] = float(box["@ybr"])
            boxes[i]["t"] = float(box["@ytl"])
            boxes[i]["l"] = float(box["@xtl"])
            boxes[i]["r"] = float(box["@xbr"])

        # Process polyline points and find corresponding boxes
        for i, line in enumerate(lines):
            points = []
            for point_str in line["@points"].split(";"):
                coords = point_str.split(",")
                if len(coords) == 2:
                    points.append((float(coords[0]), float(coords[1])))

            boxids = []
            for point in points:
                bind, _ = self.find_box(boxes=boxes, point=point)
                if 0 <= bind < len(boxes):
                    boxids.append(bind)
                else:
                    boxids.append(-1)

            lines[i]["points"] = points
            lines[i]["boxids"] = boxids

        # Categorize lines by label
        for line in lines:
            label = line["@label"]
            if label == "reading_order":
                # There should be only one reading order
                if reading_order:
                    _log.warning(
                        f"Multiple reading orders found in {basename}, using the last one"
                    )
                keep = True
                reading_order = line
            elif label == "to_caption":
                to_captions.append(line)
            elif label == "to_footnote":
                to_footnotes.append(line)
            elif label == "to_value":
                to_values.append(line)
            elif label in ["next_text", "merge"]:
                merges.append(line)
            elif label in ["next_figure", "group"]:
                group.append(line)

        return (
            basename,
            keep,
            boxes,
            lines,
            reading_order,
            to_captions,
            to_footnotes,
            to_values,
            merges,
            group,
        )

    def create_prov(
        self,
        box: Dict,
        page_no: int,
        img_width: float,
        img_height: float,
        pdf_width: float,
        pdf_height: float,
        origin: CoordOrigin = CoordOrigin.TOPLEFT,
    ) -> Tuple[ProvenanceItem, BoundingBox]:
        """
        Create provenance item and bounding box from a box.

        Args:
            box: Box data
            page_no: Page number
            img_width: Image width
            img_height: Image height
            pdf_width: PDF width
            pdf_height: PDF height
            origin: Coordinate origin

        Returns:
            Tuple of (provenance_item, bounding_box)
        """
        bbox = BoundingBox(
            l=pdf_width * box["l"] / float(img_width),
            r=pdf_width * box["r"] / float(img_width),
            b=pdf_height * box["b"] / float(img_height),
            t=pdf_height * box["t"] / float(img_height),
            coord_origin=origin,
        )
        prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, 0))
        return prov, bbox

    def get_label_prov_and_text(
        self,
        box: Dict,
        page_no: int,
        img_width: float,
        img_height: float,
        pdf_width: float,
        pdf_height: float,
        parsed_page: SegmentedPdfPage,
    ) -> Tuple[DocItemLabel, ProvenanceItem, str]:
        """
        Get label, provenance, and text from a box.

        Args:
            box: Box data
            page_no: Page number
            img_width: Image width
            img_height: Image height
            pdf_width: PDF width
            pdf_height: PDF height
            parser: PDF parser
            parsed_page: Parsed page data

        Returns:
            Tuple of (label, provenance, text)
        """
        assert page_no > 0

        prov, bbox = self.create_prov(
            box=box,
            page_no=page_no,
            img_width=img_width,
            img_height=img_height,
            pdf_width=pdf_width,
            pdf_height=pdf_height,
        )

        label = DocItemLabel(box["@label"])

        # PDF coordinates have origin at bottom-left, convert accordingly
        text_cells = parsed_page.get_cells_in_bbox(
            cell_unit=TextCellUnit.LINE, bbox=prov.bbox
        )
        text = " ".join([t.text for t in text_cells])
        text = text.replace("  ", " ")
        return label, prov, text

    def get_page_imageref(self, page_no: int, doc: DoclingDocument) -> ImageRef:
        """
        Get the image reference for a page.

        Args:
            page_no: Page number
            doc: Document

        Returns:
            ImageRef for the page
        """
        if page_no not in doc.pages:
            raise ValueError(f"Page {page_no} not found in document")

        if doc.pages[page_no].image is None:
            raise ValueError(f"Page {page_no} has no image reference")

        return cast(ImageRef, doc.pages[page_no].image)

    def get_next_provs(
        self,
        page_no: int,
        boxid: int,
        text: str,
        boxes: List[Dict],
        merges: List[Dict],
        already_added: List[int],
        true_doc: DoclingDocument,
        parsed_page: SegmentedPdfPage,
    ) -> Tuple[List[ProvenanceItem], str, List[int]]:
        """
        Get next provenance items for merged text.

        Args:
            page_no: Page number
            boxid: Box ID
            text: Current text
            boxes: List of boxes
            merges: List of merge lines
            already_added: List of already added box IDs
            true_doc: Document
            parser: PDF parser
            parsed_page: Parsed page data

        Returns:
            Tuple of (next_provenance_items, updated_text, updated_already_added)
        """
        true_page_imageref = self.get_page_imageref(page_no=page_no, doc=true_doc)

        next_provs = []
        for merge in merges:
            if len(merge["boxids"]) > 1 and merge["boxids"][0] == boxid:
                for l in range(1, len(merge["boxids"])):
                    boxid_ = merge["boxids"][l]
                    already_added.append(boxid_)

                    _, prov_, text_ = self.get_label_prov_and_text(
                        box=boxes[boxid_],
                        page_no=page_no,
                        img_width=true_page_imageref.size.width,
                        img_height=true_page_imageref.size.height,
                        pdf_width=true_doc.pages[page_no].size.width,
                        pdf_height=true_doc.pages[page_no].size.height,
                        parsed_page=parsed_page,
                    )

                    prov_.charspan = (len(text) + 1, len(text) + 1 + len(text_))
                    text = text + " " + text_
                    next_provs.append(prov_)

        return next_provs, text, already_added

    def add_captions_to_item(
        self,
        basename: str,
        to_captions: List[Dict],
        item: FloatingItem,
        page_no: int,
        boxid: int,
        boxes: List[Dict],
        already_added: List[int],
        true_doc: DoclingDocument,
        parsed_page: SegmentedPdfPage,
    ) -> Tuple[DoclingDocument, List[int]]:
        """
        Add captions to a floating item.

        Args:
            basename: Base filename
            to_captions: List of caption lines
            item: Floating item
            page_no: Page number
            boxid: Box ID
            boxes: List of boxes
            already_added: List of already added box IDs
            true_doc: Document
            parser: PDF parser
            parsed_page: Parsed page data

        Returns:
            Tuple of (updated_document, updated_already_added)
        """
        true_page_imageref = self.get_page_imageref(page_no=page_no, doc=true_doc)

        for to_caption in to_captions:
            if to_caption["boxids"][0] == boxid:
                for l in range(1, len(to_caption["boxids"])):
                    boxid_ = to_caption["boxids"][l]
                    already_added.append(boxid_)

                    caption_box = boxes[boxid_]

                    label, prov, text = self.get_label_prov_and_text(
                        box=caption_box,
                        page_no=page_no,
                        img_width=true_page_imageref.size.width,
                        img_height=true_page_imageref.size.height,
                        pdf_width=true_doc.pages[page_no].size.width,
                        pdf_height=true_doc.pages[page_no].size.height,
                        parsed_page=parsed_page,
                    )

                    caption_ref = true_doc.add_text(
                        label=DocItemLabel.CAPTION, prov=prov, text=text
                    )
                    item.captions.append(caption_ref.get_ref())

                    if label != DocItemLabel.CAPTION:
                        _log.warning(f"{label} != DocItemLabel.CAPTION for {basename}")

        return true_doc, already_added

    def add_footnotes_to_item(
        self,
        basename: str,
        to_footnotes: List[Dict],
        item: FloatingItem,
        page_no: int,
        boxid: int,
        boxes: List[Dict],
        already_added: List[int],
        true_doc: DoclingDocument,
        parsed_page: SegmentedPdfPage,
    ) -> Tuple[DoclingDocument, List[int]]:
        """
        Add footnotes to a floating item.

        Args:
            basename: Base filename
            to_footnotes: List of footnote lines
            item: Floating item
            page_no: Page number
            boxid: Box ID
            boxes: List of boxes
            already_added: List of already added box IDs
            true_doc: Document
            parser: PDF parser
            parsed_page: Parsed page data

        Returns:
            Tuple of (updated_document, updated_already_added)
        """
        true_page_imageref = self.get_page_imageref(page_no=page_no, doc=true_doc)

        for to_footnote in to_footnotes:
            if to_footnote["boxids"][0] == boxid:
                for l in range(1, len(to_footnote["boxids"])):
                    boxid_ = to_footnote["boxids"][l]
                    already_added.append(boxid_)

                    footnote_box = boxes[boxid_]

                    label, prov, text = self.get_label_prov_and_text(
                        box=footnote_box,
                        page_no=page_no,
                        img_width=true_page_imageref.size.width,
                        img_height=true_page_imageref.size.height,
                        pdf_width=true_doc.pages[page_no].size.width,
                        pdf_height=true_doc.pages[page_no].size.height,
                        parsed_page=parsed_page,
                    )

                    footnote_ref = true_doc.add_text(
                        label=DocItemLabel.FOOTNOTE, prov=prov, text=text
                    )
                    item.footnotes.append(footnote_ref.get_ref())

                    if label != DocItemLabel.FOOTNOTE:
                        _log.warning(f"{label} != DocItemLabel.FOOTNOTE for {basename}")

        return true_doc, already_added

    def create_true_document(
        self, basename: str, annot: Dict, desc: AnnotatedImage
    ) -> Optional[DoclingDocument]:
        """
        Create a ground truth document from CVAT annotations.

        Args:
            basename: Base filename
            annot: Annotation data
            desc: Annotated image description

        Returns:
            DoclingDocument or None if creation fails
        """
        # Parse the annotation
        (
            given_basename,
            keep,
            boxes,
            lines,
            reading_order,
            to_captions,
            to_footnotes,
            to_values,
            merges,
            group,
        ) = self.parse_annotation(annot)

        assert given_basename == basename

        if not keep:
            _log.error(f"Incorrect annotation for {basename}: no reading order found")
            return None

        # Original Groundtruth and Prediction files
        orig_file = desc.document_file

        if not (os.path.exists(orig_file)):
            _log.error(f"Missing original files for {basename}")
            return None

        orig_doc = DoclingDocument.load_from_json(filename=orig_file)

        # Original PDF file
        pdf_file = desc.bin_file
        if not os.path.exists(pdf_file):
            _log.error(f"Missing PDF file for {basename}")
            return None

        in_doc = InputDocument(
            path_or_stream=pdf_file,
            format=InputFormat.PDF,
            backend=DoclingParseV4DocumentBackend,
        )

        doc_backend: DoclingParseV4DocumentBackend = in_doc._backend  # type: ignore

        # Parse each page
        parsed_pages: Dict[int, SegmentedPdfPage] = {}
        for i, page_no in enumerate(desc.page_nos):
            seg_page = doc_backend.load_page(page_no - 1).get_segmented_page()
            assert seg_page is not None
            parsed_pages[page_no] = seg_page

        doc_backend.unload()

        # Create Ground Truth document
        new_doc = DoclingDocument(name=f"{basename}")

        # Copy page images from predicted document
        for i, page_no in enumerate(desc.page_nos):
            # PDF dimensions
            pdf_width = parsed_pages[page_no].dimension.width
            pdf_height = parsed_pages[page_no].dimension.height

            # Image file
            img_file = desc.page_img_files[i]
            page_image = Image.open(str(img_file))
            img_width = page_image.width
            img_height = page_image.height

            # Check if predicted document has the page
            if page_no not in orig_doc.pages or orig_doc.pages[page_no] is None:
                _log.error(f"Missing page {page_no} in predicted document, skipping...")
                continue

            orig_page_item = orig_doc.pages[page_no]
            orig_page_image_ref = orig_page_item.image

            if orig_page_image_ref is None:
                _log.error(f"Missing image reference for page {page_no}, skipping...")
                continue

            # Create image reference and page item
            image_ref = ImageRef(
                mimetype="image/png",
                dpi=orig_page_image_ref.dpi,
                size=Size(width=float(img_width), height=float(img_height)),
                uri=from_pil_to_base64uri(page_image),
            )

            page_item = PageItem(
                page_no=page_no,
                size=Size(width=float(pdf_width), height=float(pdf_height)),
                image=image_ref,
            )

            new_doc.pages[page_no] = page_item

        # Process items based on reading order
        already_added: List[int] = []

        # Sanity check for reading order
        if not reading_order or "boxids" not in reading_order:
            _log.error(f"Invalid reading order for {basename}")
            return None

        for boxid in reading_order["boxids"]:
            if boxid in already_added:
                _log.warning(f"Box {boxid} is already added, skipping...")
                continue

            # FIXME: For simplicity, assume all boxes are on page 1
            page_no = 1

            if page_no not in new_doc.pages or new_doc.pages[page_no] is None:
                _log.error(f"Page {page_no} not found in document, skipping...")
                continue

            # Get image reference for the page
            true_page_imageref = self.get_page_imageref(page_no=page_no, doc=new_doc)

            assert true_page_imageref.pil_image is not None
            true_page_pilimage: Image.Image = true_page_imageref.pil_image

            # Get label, provenance, and text for the box
            label, prov, text = self.get_label_prov_and_text(
                box=boxes[boxid],
                page_no=page_no,
                img_width=true_page_imageref.size.width,
                img_height=true_page_imageref.size.height,
                pdf_width=new_doc.pages[page_no].size.width,
                pdf_height=new_doc.pages[page_no].size.height,
                parsed_page=parsed_pages[page_no],
            )

            # Process merged text
            next_provs, text, already_added = self.get_next_provs(
                page_no=page_no,
                boxid=boxid,
                text=text,
                boxes=boxes,
                merges=merges,
                already_added=already_added,
                true_doc=new_doc,
                parsed_page=parsed_pages[page_no],
            )

            # Add item to document based on label
            if label in [
                DocItemLabel.TEXT,
                DocItemLabel.PARAGRAPH,
                DocItemLabel.REFERENCE,
                DocItemLabel.PAGE_HEADER,
                DocItemLabel.PAGE_FOOTER,
                DocItemLabel.TITLE,
                DocItemLabel.FOOTNOTE,
            ]:
                current_item = new_doc.add_text(label=label, prov=prov, text=text)
                for next_prov in next_provs:
                    current_item.prov.append(next_prov)

            elif label == DocItemLabel.SECTION_HEADER:
                new_doc.add_text(label=label, prov=prov, text=text)

            elif label == DocItemLabel.CAPTION:
                new_doc.add_text(label=label, prov=prov, text=text)

            elif label in [
                DocItemLabel.CHECKBOX_SELECTED,
                DocItemLabel.CHECKBOX_UNSELECTED,
            ]:
                new_doc.add_text(label=label, prov=prov, text=text)

            elif label == DocItemLabel.LIST_ITEM:
                new_doc.add_list_item(prov=prov, text=text)

            elif label == DocItemLabel.FORMULA:
                new_doc.add_text(label=label, prov=prov, text=text)

            elif label == DocItemLabel.CODE:
                code_item = new_doc.add_code(text=text, prov=prov)

                new_doc, already_added = self.add_captions_to_item(
                    basename=basename,
                    to_captions=to_captions,
                    item=code_item,
                    page_no=page_no,
                    boxid=boxid,
                    boxes=boxes,
                    already_added=already_added,
                    true_doc=new_doc,
                    parsed_page=parsed_pages[page_no],
                )

            elif label == DocItemLabel.FORM:
                graph = GraphData(cells=[], links=[])
                new_doc.add_form(graph=graph, prov=prov)

            elif label == DocItemLabel.KEY_VALUE_REGION:
                graph = GraphData(cells=[], links=[])
                new_doc.add_key_values(graph=graph, prov=prov)

            elif label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]:
                # Try to find table data in original document
                table_data = find_table_data(doc=orig_doc, prov=prov)

                if table_data.num_rows <= 0 or table_data.num_cols <= 0:
                    # Create an empty table structure if no data found
                    table_data = TableData(num_rows=0, num_cols=0, table_cells=[])

                table_item = new_doc.add_table(label=label, data=table_data, prov=prov)

                # Add captions and footnotes to table
                new_doc, already_added = self.add_captions_to_item(
                    basename=basename,
                    to_captions=to_captions,
                    item=table_item,
                    page_no=page_no,
                    boxid=boxid,
                    boxes=boxes,
                    already_added=already_added,
                    true_doc=new_doc,
                    parsed_page=parsed_pages[page_no],
                )

                new_doc, already_added = self.add_footnotes_to_item(
                    basename=basename,
                    to_footnotes=to_footnotes,
                    item=table_item,
                    page_no=page_no,
                    boxid=boxid,
                    boxes=boxes,
                    already_added=already_added,
                    true_doc=new_doc,
                    parsed_page=parsed_pages[page_no],
                )

            elif label == DocItemLabel.PICTURE:
                # Crop image from page based on bounding box
                crop_image = crop_bounding_box(
                    page_image=true_page_pilimage,
                    page=new_doc.pages[page_no],
                    bbox=prov.bbox,
                )

                # Create image reference
                imgref = ImageRef(
                    mimetype="image/png",
                    dpi=72,
                    size=Size(width=crop_image.width, height=crop_image.height),
                    uri=from_pil_to_base64uri(crop_image),
                )

                if new_doc.pages[page_no].image is not None:
                    imgref.dpi = new_doc.pages[page_no].image.dpi  # type: ignore

                # Add picture to document
                picture_item = new_doc.add_picture(prov=prov, image=imgref)

                # Add captions and footnotes to picture
                new_doc, already_added = self.add_captions_to_item(
                    basename=basename,
                    to_captions=to_captions,
                    item=picture_item,
                    page_no=page_no,
                    boxid=boxid,
                    boxes=boxes,
                    already_added=already_added,
                    true_doc=new_doc,
                    parsed_page=parsed_pages[page_no],
                )

                new_doc, already_added = self.add_footnotes_to_item(
                    basename=basename,
                    to_footnotes=to_footnotes,
                    item=picture_item,
                    page_no=page_no,
                    boxid=boxid,
                    boxes=boxes,
                    already_added=already_added,
                    true_doc=new_doc,
                    parsed_page=parsed_pages[page_no],
                )

        return new_doc

    def contains_reading_order(self, image_annot: Dict) -> bool:
        """
        Check if an image annotation contains reading order.

        Args:
            image_annot: Image annotation data

        Returns:
            True if the annotation contains reading order, False otherwise
        """
        if "box" not in image_annot:
            return False

        if "polyline" not in image_annot:
            return False

        if isinstance(image_annot["polyline"], dict):
            lines = [image_annot["polyline"]]
        elif isinstance(image_annot["polyline"], list):
            lines = image_annot["polyline"]
        else:
            return False

        cnt = 0
        for line in lines:
            if line["@label"] == "reading_order":
                cnt += 1

        return cnt == 1

    def from_cvat_to_docling_document(
        self, annotation_filenames: List[Path], overview: AnnotationOverview
    ) -> Iterable[Tuple[str, AnnotatedImage, Optional[DoclingDocument]]]:
        """
        Convert CVAT annotations to DoclingDocument objects.

        Args:
            annotation_filenames: List of annotation filenames
            overview: Annotation overview

        Yields:
            Tuple of (basename, annotated_image, docling_document)
        """
        for annot_file in annotation_filenames:
            with open(str(annot_file), "r") as fr:
                xml_data = fr.read()

            # Convert XML to a Python dictionary
            annot_data = xmltodict.parse(xml_data)

            for image_annot in annot_data["annotations"]["image"]:
                basename = image_annot["@name"]

                if basename not in overview.img_annotations:
                    _log.warning(f"Skipping {basename}: not in overview file")
                    continue

                if not self.contains_reading_order(image_annot):
                    _log.warning(f"Skipping {basename}: no reading-order detected")
                    yield basename, overview.img_annotations[basename], None
                else:
                    try:
                        true_doc = self.create_true_document(
                            basename=basename,
                            annot=image_annot,
                            desc=overview.img_annotations[basename],
                        )
                    except Exception as e:
                        _log.fatal("Exception occured", e)
                    yield basename, overview.img_annotations[basename], true_doc

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Create dataset records from CVAT annotations.

        This method processes all annotation files and creates dataset records
        from valid annotations with reading order information.

        Returns:
            List of DatasetRecord objects
        """

        # Load the overview file
        try:
            overview = AnnotationOverview.load_from_json(
                self.benchmark_dirs.overview_file
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            _log.error(f"Failed to load annotation overview file: {e}")
            raise

        # Get annotation files
        annot_files = self.get_annotation_files()
        if not annot_files:
            _log.error("No annotation files found in CVAT dir")
            raise

        # Calculate total items for effective indices
        total_items = len(overview.img_annotations)
        begin, end = self.get_effective_indices(total_items)

        # Log statistics
        self.log_dataset_stats(total_items, end - begin)

        item_count = 0

        _log.info(f"Processing annotations from index {begin} to {end}")

        try:
            for basename, desc, true_doc in tqdm(
                self.from_cvat_to_docling_document(annot_files, overview),
                total=len(overview.img_annotations),
                ncols=128,
                desc="Creating documents from annotations",
            ):
                # Skip if no document was created
                if true_doc is None:
                    _log.warning(f"No document created for {basename}, skipping")
                    continue

                # Apply index filtering
                item_count += 1
                if item_count < begin:
                    continue
                if item_count >= end:
                    return

                # Save the document as JSON
                json_path = self.benchmark_dirs.json_anno_dir / f"{basename}.json"
                true_doc.save_as_json(json_path)

                # Extract images
                true_doc, true_pictures, true_page_images = extract_images(
                    document=true_doc,
                    pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                    page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                )

                # Get PDF as binary data
                pdf_file = Path(desc.bin_file)
                if not pdf_file.exists():
                    _log.warning(f"PDF file {pdf_file} not found, skipping")
                    continue

                pdf_bytes = get_binary(pdf_file)
                pdf_stream = DocumentStream(
                    name=pdf_file.name, stream=BytesIO(pdf_bytes)
                )

                # Create dataset record
                record = DatasetRecord(
                    doc_id=str(basename),
                    doc_path=str(basename),
                    doc_hash=get_binhash(pdf_bytes),
                    ground_truth_doc=true_doc,
                    ground_truth_pictures=true_pictures,
                    ground_truth_page_images=true_page_images,
                    original=pdf_stream,
                    mime_type="application/pdf",
                    modalities=[
                        EvaluationModality.LAYOUT,
                        EvaluationModality.READING_ORDER,
                        EvaluationModality.CAPTIONING,
                    ],
                )

                yield record
        except Exception as e:
            _log.error(f"Error processing annotations: {str(e)}")


def find_table_data(doc: DoclingDocument, prov, iou_cutoff: float = 0.90):
    """
    Find table data in a document based on provenance.

    Args:
        doc: Document to search in
        prov: Provenance to match
        iou_cutoff: IoU threshold for matching

    Returns:
        TableData structure from the matching table or an empty structure
    """
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            for item_prov in item.prov:
                if item_prov.page_no != prov.page_no:
                    continue

                # page_height = doc.pages[item_prov.page_no].size.height
                iou = item_prov.bbox.intersection_over_union(prov.bbox)

                if iou > iou_cutoff:
                    _log.info(f"Found matching table data with IoU: {iou:.2f}")
                    return item.data

    _log.warning("No matching table data found")

    # Return empty table data
    return TableData(num_rows=-1, num_cols=-1, table_cells=[])
