import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Set

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
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
TRUE_HTML_EXPORT_LABELS: Set[DocItemLabel] = {
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

PRED_HTML_EXPORT_LABELS: Set[DocItemLabel] = {
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


class DPBenchDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    DPBench dataset builder implementing the base dataset builder interface.

    This builder processes the DPBench dataset, which contains document
    understanding benchmarks for various document types.
    """

    def __init__(
        self,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the DPBench dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="DPBench",
            dataset_source=HFSource(repo_id="upstage/dp-bench"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        self.must_retrieve = True

    def _update_gt_doc(
        self,
        doc: DoclingDocument,
        annots: Dict,
        page,
        page_image: Image,
        page_width: float,
        page_height: float,
    ) -> None:
        """
        Update ground truth document with annotations.

        Args:
            doc: DoclingDocument to update
            annots: Annotation data
            page: Page object
            page_image: Page image
            page_width: Page width
            page_height: Page height
        """
        label = annots["category"]

        # Extract coordinates
        min_x = annots["coordinates"][0]["x"]
        max_x = annots["coordinates"][0]["x"]
        min_y = annots["coordinates"][0]["y"]
        max_y = annots["coordinates"][0]["y"]

        for coor in annots["coordinates"]:
            min_x = min(min_x, coor["x"])
            max_x = max(max_x, coor["x"])
            min_y = min(min_y, coor["y"])
            max_y = max(max_y, coor["y"])

        text = annots["content"]["text"].replace("\n", " ")
        html = annots["content"]["html"]

        # Create bounding box
        bbox = BoundingBox(
            l=min_x * page_width,
            r=max_x * page_width,
            t=min_y * page_height,
            b=max_y * page_height,
            coord_origin=CoordOrigin.TOPLEFT,
        )

        # Create provenance
        prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))

        # Crop image element
        img = crop_bounding_box(page_image=page_image, page=page, bbox=bbox)

        # Add element to document based on label
        if label == "Header":
            doc.add_text(
                label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov
            )

        elif label == "Footer":
            doc.add_text(
                label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov
            )

        elif label == "Paragraph":
            doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label == "Index":
            # FIXME: ultra approximate solution
            text = annots["content"]["text"]
            rows = text.split("\n")

            num_rows = len(rows)
            num_cols = 2

            row_span = 1
            col_span = 1

            cells = []
            for row_idx, row in enumerate(rows):
                parts = row.split(" ")

                col_idx = 0
                cell = TableCell(
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    text=" ".join(parts[:-1]),
                )
                cells.append(cell)

                col_idx = 1
                cell = TableCell(
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    text=parts[-1],
                )
                cells.append(cell)

            table_data = TableData(
                num_rows=num_rows, num_cols=num_cols, table_cells=cells
            )
            doc.add_table(
                data=table_data,
                caption=None,
                prov=prov,
                label=DocItemLabel.DOCUMENT_INDEX,
            )

        elif label == "List":
            doc.add_list_item(text=text, orig=text, prov=prov)

        elif label == "Caption":
            doc.add_text(label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov)

        elif label == "Equation":
            doc.add_text(label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov)

        elif label == "Figure":
            uri = from_pil_to_base64uri(img)
            imgref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=img.width, height=img.height),
                uri=uri,
            )
            doc.add_picture(prov=prov, image=imgref)

        elif label == "Table":
            table_data = convert_html_table_into_docling_tabledata(table_html=html)
            doc.add_table(data=table_data, caption=None, prov=prov)

        elif label == "Chart":
            uri = from_pil_to_base64uri(img)
            imgref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=img.width, height=img.height),
                uri=uri,
            )
            doc.add_picture(prov=prov, image=imgref)

        elif label == "Footnote":
            doc.add_text(label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov)

        elif label == "Heading1":
            doc.add_heading(text=text, orig=text, level=1, prov=prov)

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
        reference_path = self.dataset_local_path / "dataset/reference.json"
        with open(reference_path, "r") as fr:
            gt = json.load(fr)

        # Sort the filenames for deterministic ordering
        sorted_filenames = sorted(gt.keys())
        total_files = len(sorted_filenames)

        # Apply index range
        begin, end = self.get_effective_indices(total_files)
        selected_filenames = sorted_filenames[begin:end]

        # Log stats
        self.log_dataset_stats(total_files, len(selected_filenames))
        _log.info(f"Processing DP-Bench dataset with {len(selected_filenames)} files")

        for filename in tqdm(
            selected_filenames,
            desc="Processing files for DP-Bench",
            ncols=128,
        ):
            # Get annotations for this file
            annots = gt[filename]
            pdf_path = self.dataset_local_path / f"dataset/pdfs/{filename}"

            # Create the ground truth Document
            true_doc = DoclingDocument(
                name=f"ground-truth {os.path.basename(pdf_path)}"
            )
            true_doc, true_page_images = add_pages_to_true_doc(
                pdf_path=pdf_path, true_doc=true_doc, image_scale=2.0
            )

            assert len(true_page_images) == 1, "len(true_page_images)==1"

            # Get page dimensions
            page_width = true_doc.pages[1].size.width
            page_height = true_doc.pages[1].size.height

            # Process each element in the annotation
            for elem in annots["elements"]:
                self._update_gt_doc(
                    true_doc,
                    elem,
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
                doc_id=str(filename),
                doc_hash=get_binhash(pdf_bytes),
                ground_truth_doc=true_doc,
                ground_truth_pictures=true_pictures,
                ground_truth_page_images=true_page_images,
                original=pdf_stream,
                mime_type="application/pdf",
            )

            yield record
