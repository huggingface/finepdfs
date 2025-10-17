import io
import itertools
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_from_disk
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    GroupLabel,
    ImageRef,
    PageItem,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import GraphCell, GraphData, GraphLink
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel
from docling_core.types.doc.tokens import TableToken
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns, EvaluationModality
from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
from docling_eval.utils.utils import (
    classify_cells,
    crop_bounding_box,
    extract_images,
    from_pil_to_base64uri,
    get_binhash,
    sort_cell_ids,
)

# Get logger
_log = logging.getLogger(__name__)


class DocLayNetV2DatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    DocLayNet V2 dataset builder implementing the base dataset builder interface.

    This builder processes the DocLayNet V2 dataset, which contains document
    layout annotations and key-value data for a variety of document types.
    """

    def __init__(
        self,
        dataset_source: Path,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the DocLayNet V2 dataset builder.

        Args:
            dataset_source: Path to the pre-downloaded dataset
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="DocLayNetV2: end-to-end",
            dataset_source=dataset_source,  # Local Path to dataset
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        self.must_retrieve = True

    def extract_tokens_and_text(self, s: str) -> Tuple[List[str], List[str]]:
        """
        Extract tokens and text from a string.

        Args:
            s: Input string

        Returns:
            Tuple of (tokens, text_parts)
        """
        # Pattern to match anything enclosed by < > (including the angle brackets themselves)
        pattern = r"(<[^>]+>)"
        # Find all tokens (e.g. "<otsl>", "<loc_140>", etc.)
        tokens = re.findall(pattern, s)
        # Remove any tokens that start with "<loc_"
        tokens = [
            token
            for token in tokens
            if not (token.startswith("<loc_") or token in ["<otsl>", "</otsl>"])
        ]
        # Split the string by those tokens to get the in-between text
        text_parts = re.split(pattern, s)
        text_parts = [
            token
            for token in text_parts
            if not (token.startswith("<loc_") or token in ["<otsl>", "</otsl>"])
        ]
        # Remove any empty or purely whitespace strings from text_parts
        text_parts = [part for part in text_parts if part.strip()]

        return tokens, text_parts

    def parse_texts(
        self, texts: List[str], tokens: List[str]
    ) -> Tuple[List[TableCell], List[List[str]]]:
        """
        Parse tokens and texts into table cells.

        Args:
            texts: List of text parts
            tokens: List of tokens

        Returns:
            Tuple of (table_cells, split_row_tokens)
        """
        split_word = TableToken.OTSL_NL.value
        split_row_tokens = [
            list(y)
            for x, y in itertools.groupby(tokens, lambda z: z == split_word)
            if not x
        ]
        table_cells = []
        r_idx = 0
        c_idx = 0

        def count_right(
            tokens: List[List[str]], c_idx: int, r_idx: int, which_tokens: List[str]
        ) -> int:
            span = 0
            c_idx_iter = c_idx
            while tokens[r_idx][c_idx_iter] in which_tokens:
                c_idx_iter += 1
                span += 1
                if c_idx_iter >= len(tokens[r_idx]):
                    return span
            return span

        def count_down(
            tokens: List[List[str]], c_idx: int, r_idx: int, which_tokens: List[str]
        ) -> int:
            span = 0
            r_idx_iter = r_idx
            while tokens[r_idx_iter][c_idx] in which_tokens:
                r_idx_iter += 1
                span += 1
                if r_idx_iter >= len(tokens):
                    return span
            return span

        for i, text in enumerate(texts):
            cell_text = ""
            if text in [
                TableToken.OTSL_FCEL.value,
                TableToken.OTSL_ECEL.value,
                TableToken.OTSL_CHED.value,
                TableToken.OTSL_RHED.value,
                TableToken.OTSL_SROW.value,
            ]:
                row_span = 1
                col_span = 1
                right_offset = 1
                if text != TableToken.OTSL_ECEL.value:
                    cell_text = texts[i + 1]
                    right_offset = 2

                # Check next element(s) for lcel / ucel / xcel, set properly row_span, col_span
                if i + right_offset < len(texts):
                    next_right_cell = texts[i + right_offset]

                    next_bottom_cell = ""
                    if r_idx + 1 < len(split_row_tokens):
                        next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

                    if next_right_cell in [
                        TableToken.OTSL_LCEL.value,
                        TableToken.OTSL_XCEL.value,
                    ]:
                        # we have horisontal spanning cell or 2d spanning cell
                        col_span += count_right(
                            split_row_tokens,
                            c_idx + 1,
                            r_idx,
                            [TableToken.OTSL_LCEL.value, TableToken.OTSL_XCEL.value],
                        )
                    if next_bottom_cell in [
                        TableToken.OTSL_UCEL.value,
                        TableToken.OTSL_XCEL.value,
                    ]:
                        # we have a vertical spanning cell or 2d spanning cell
                        row_span += count_down(
                            split_row_tokens,
                            c_idx,
                            r_idx + 1,
                            [TableToken.OTSL_UCEL.value, TableToken.OTSL_XCEL.value],
                        )

                table_cells.append(
                    TableCell(
                        text=cell_text.strip(),
                        row_span=row_span,
                        col_span=col_span,
                        start_row_offset_idx=r_idx,
                        end_row_offset_idx=r_idx + row_span,
                        start_col_offset_idx=c_idx,
                        end_col_offset_idx=c_idx + col_span,
                    )
                )
            if text in [
                TableToken.OTSL_FCEL.value,
                TableToken.OTSL_ECEL.value,
                TableToken.OTSL_CHED.value,
                TableToken.OTSL_RHED.value,
                TableToken.OTSL_SROW.value,
                TableToken.OTSL_LCEL.value,
                TableToken.OTSL_UCEL.value,
                TableToken.OTSL_XCEL.value,
            ]:
                c_idx += 1
            if text == TableToken.OTSL_NL.value:
                r_idx += 1
                c_idx = 0
        return table_cells, split_row_tokens

    def parse_table_content(self, otsl_content: str) -> TableData:
        """
        Parse OTSL content into TableData.

        Args:
            otsl_content: OTSL content string

        Returns:
            TableData object
        """
        tokens, mixed_texts = self.extract_tokens_and_text(otsl_content)
        table_cells, split_row_tokens = self.parse_texts(mixed_texts, tokens)

        return TableData(
            num_rows=len(split_row_tokens),
            num_cols=(
                max(len(row) for row in split_row_tokens) if split_row_tokens else 0
            ),
            table_cells=table_cells,
        )

    def convert_bbox(self, bbox_data) -> BoundingBox:
        """
        Convert bbox format to BoundingBox object.

        Args:
            bbox_data: Bounding box data as list or BoundingBox

        Returns:
            BoundingBox object
        """
        if isinstance(bbox_data, list) and len(bbox_data) == 4:
            return BoundingBox(
                l=bbox_data[0], t=bbox_data[1], r=bbox_data[2], b=bbox_data[3]
            )
        elif isinstance(bbox_data, BoundingBox):
            return bbox_data
        else:
            raise ValueError(
                "Invalid bounding box data; expected a list of four numbers or a BoundingBox instance."
            )

    def create_graph_cell(self, cell_data: Dict, label: GraphCellLabel) -> GraphCell:
        """
        Create a graph cell from cell data.

        Args:
            cell_data: Cell data dictionary
            label: GraphCellLabel for the cell

        Returns:
            GraphCell object
        """
        bbox_instance = None
        if "bbox" in cell_data and cell_data["bbox"] is not None:
            bbox_instance = self.convert_bbox(cell_data["bbox"])
            cell_prov = ProvenanceItem(
                page_no=1,
                charspan=(0, 0),
                bbox=bbox_instance,
            )
        else:
            cell_prov = None

        return GraphCell(
            cell_id=cell_data["cell_id"],
            text=cell_data["text"],
            orig=cell_data.get("orig", cell_data["text"]),
            prov=cell_prov,
            label=label,
        )

    def create_graph_link(
        self,
        key_cell: GraphCell,
        value_cell: GraphCell,
        label: GraphLinkLabel = GraphLinkLabel.TO_VALUE,
    ) -> GraphLink:
        """
        Create a graph link between key and value cells.

        Args:
            key_cell: Source GraphCell
            value_cell: Target GraphCell
            label: GraphLinkLabel for the link

        Returns:
            GraphLink object
        """
        return GraphLink(
            source_cell_id=key_cell.cell_id,
            target_cell_id=value_cell.cell_id,
            label=label,
        )

    def get_overall_bbox(
        self, links: List[GraphLink], cell_dict: Dict[int, GraphCell]
    ) -> Optional[BoundingBox]:
        """
        Compute the overall bounding box from all cell ids.

        Args:
            links: List of GraphLink objects
            cell_dict: Dictionary mapping cell_id to GraphCell

        Returns:
            Optional BoundingBox encompassing all cells
        """
        all_bboxes = []
        for link in links:
            src_prov = cell_dict[link.source_cell_id].prov
            tgt_prov = cell_dict[link.target_cell_id].prov
            if src_prov is not None:
                all_bboxes.append(src_prov.bbox)
            if tgt_prov is not None:
                all_bboxes.append(tgt_prov.bbox)

        if len(all_bboxes) == 0:
            return None
        bbox_instance = BoundingBox.enclosing_bbox(all_bboxes)
        return bbox_instance

    def populate_key_value_item(
        self,
        doc: DoclingDocument,
        kv_pairs: List[Dict],
    ) -> None:
        """
        Populate a key-value item in the document.

        Args:
            doc: DoclingDocument to update
            kv_pairs: List of key-value pair dictionaries
        """
        cell_by_id: Dict[int, GraphCell] = {}
        links = []

        for pair in kv_pairs:
            key_data = pair["key"]
            value_data = pair["value"]

            if cell_by_id.get(key_data["cell_id"], None) is None:
                key_cell = self.create_graph_cell(key_data, GraphCellLabel.KEY)
                cell_by_id[key_data["cell_id"]] = key_cell
            else:
                key_cell = cell_by_id[key_data["cell_id"]]

            if cell_by_id.get(value_data["cell_id"], None) is None:
                value_cell = self.create_graph_cell(value_data, GraphCellLabel.VALUE)
                cell_by_id[value_data["cell_id"]] = value_cell
            else:
                value_cell = cell_by_id[value_data["cell_id"]]

            # link between key and value
            kv_link = self.create_graph_link(key_cell, value_cell)
            links.append(kv_link)

        cells = list(cell_by_id.values())

        overall_bbox = self.get_overall_bbox(
            links, cell_dict={cell.cell_id: cell for cell in cells}
        )

        if overall_bbox is not None:
            prov = ProvenanceItem(
                page_no=doc.pages[1].page_no,
                charspan=(0, 0),
                bbox=overall_bbox,
            )
        else:
            prov = None

        graph = GraphData(cells=cells, links=links)

        # update the labels of the cells based on the links with rules
        classify_cells(graph=graph)

        # Add the key_value_item to the document.
        doc.add_key_values(graph=graph, prov=prov)

        # sort the cell ids in the graph
        sort_cell_ids(doc)

    # The minimal fix for DocLayNetV2Builder is to add type annotation to link_pairs:

    def create_kv_pairs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create key-value pairs from document data.

        Args:
            data: Document data dictionary

        Returns:
            List of key-value pair dictionaries
        """
        link_pairs: List[Dict[str, Any]] = []
        seg_with_id = {}
        bbox_with_id = {}

        if (
            "annotation_ids" not in data
            or "boxes" not in data
            or "segments" not in data
            or "links" not in data
        ):
            return link_pairs

        _ids = data["annotation_ids"]
        bboxes = data["boxes"]
        segments = data["segments"]
        links = data["links"]

        # str to integer id mapping
        int_ids = {id: i for i, id in enumerate(_ids)}

        for i, seg in enumerate(segments):
            seg_with_id[_ids[i]] = seg
            bbox_with_id[_ids[i]] = bboxes[i]

        for i, segment in enumerate(segments):
            if links[i] is not None and links[i] in seg_with_id:
                link_pairs.append(
                    {
                        "value": {
                            "cell_id": int_ids[_ids[i]],
                            "bbox": bboxes[i],
                            "text": segment["text"],
                            "label": GraphCellLabel.VALUE,
                        },
                        "key": {
                            "cell_id": int_ids[links[i]],
                            "bbox": bbox_with_id[links[i]],
                            "text": seg_with_id[links[i]]["text"],
                            "label": GraphCellLabel.KEY,
                        },
                    }
                )
        return link_pairs

    def update_doc(
        self,
        true_doc: DoclingDocument,
        current_list,
        img: Image.Image,
        label: str,
        segment: Dict,
        box: List,
    ):
        """
        Update the document with a new element based on its label.

        Args:
            true_doc: DoclingDocument to update
            current_list: Current list context for list items
            img: Page image
            label: Element label
            segment: Segment data dictionary
            box: Bounding box coordinates

        Returns:
            Updated list context
        """
        bbox = BoundingBox.from_tuple(
            tuple(box), CoordOrigin.TOPLEFT
        ).to_bottom_left_origin(page_height=true_doc.pages[1].size.height)
        prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(segment["text"])))
        img_elem = crop_bounding_box(page_image=img, page=true_doc.pages[1], bbox=bbox)

        # Convert label string to DocItemLabel enum (with fallback to TEXT)
        try:
            doc_label = DocItemLabel(label)
        except ValueError:
            _log.warning(f"Unknown label type: {label}, defaulting to TEXT")
            doc_label = DocItemLabel.TEXT

        if doc_label == DocItemLabel.PICTURE:
            current_list = None
            try:
                uri = from_pil_to_base64uri(img_elem)
                imgref = ImageRef(
                    mimetype="image/png",
                    dpi=72,
                    size=Size(width=img_elem.width, height=img_elem.height),
                    uri=uri,
                )
                true_doc.add_picture(prov=prov, image=imgref)
            except Exception as e:
                _log.error(f"Failed to create image reference: {str(e)}")

        elif doc_label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]:
            current_list = None
            if (
                segment.get("data") is not None
                and segment["data"].get("otsl_seq") is not None
            ):
                otsl_str = "".join(segment["data"]["otsl_seq"])
                tbl_data = self.parse_table_content(otsl_str)
                true_doc.add_table(
                    data=tbl_data, caption=None, prov=prov, label=doc_label
                )
            else:
                # Simple fallback for tables without OTSL data
                tbl_cell = TableCell(
                    start_row_offset_idx=0,
                    end_row_offset_idx=0,
                    start_col_offset_idx=0,
                    end_col_offset_idx=0,
                    text=segment["text"],
                )
                tbl_data = TableData(table_cells=[tbl_cell])
                true_doc.add_table(
                    data=tbl_data, caption=None, prov=prov, label=doc_label
                )

        elif doc_label in [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]:
            group_label = GroupLabel.UNSPECIFIED
            if doc_label == DocItemLabel.FORM:
                group_label = GroupLabel.FORM_AREA
            elif doc_label == DocItemLabel.KEY_VALUE_REGION:
                group_label = GroupLabel.KEY_VALUE_AREA
            true_doc.add_group(label=group_label, name=f"{doc_label}_group")

        elif doc_label == DocItemLabel.LIST_ITEM:
            if current_list is None:
                current_list = true_doc.add_group(label=GroupLabel.LIST, name="list")

            true_doc.add_list_item(
                text=segment["text"],
                enumerated=False,
                prov=prov,
                parent=current_list,
            )

        elif doc_label == DocItemLabel.SECTION_HEADER:
            current_list = None
            true_doc.add_heading(
                text=segment["text"], orig=segment["text"], level=2, prov=prov
            )

        elif doc_label == DocItemLabel.TITLE:
            current_list = None
            true_doc.add_heading(
                text=segment["text"], orig=segment["text"], level=1, prov=prov
            )

        else:
            current_list = None
            true_doc.add_text(
                label=doc_label, text=segment["text"], orig=segment["text"], prov=prov
            )

        return current_list

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

        assert isinstance(self.dataset_source, Path)

        try:
            # Load dataset
            ds = load_from_disk(str(self.dataset_source))

            # Get total number of items in the dataset
            total_items = len(ds[self.split])

            # Calculate effective indices
            begin, end = self.get_effective_indices(total_items)

            # Log stats
            self.log_dataset_stats(total_items, end - begin)
            _log.info(f"Processing DocLayNetV2 dataset: {end - begin} documents")

            # Process each document
            for i, doc in enumerate(
                tqdm(
                    ds[self.split],
                    total=end - begin,
                    desc="Processing DocLayNetV2 documents",
                )
            ):
                # Skip documents before begin_index
                if i < begin:
                    continue

                # Stop after end_index
                if i >= end:
                    break

                try:
                    # Extract image
                    img = doc["image"]

                    # Convert image to bytes for storage
                    with io.BytesIO() as img_byte_stream:
                        img.save(img_byte_stream, format=img.format or "PNG")
                        img_byte_stream.seek(0)
                        img_bytes = img_byte_stream.getvalue()

                    # Create ground truth document
                    doc_id = doc["page_hash"]
                    true_doc = DoclingDocument(name=doc_id)

                    # Add page with image
                    image_ref = ImageRef(
                        mimetype=f"image/{img.format.lower() if img.format else 'png'}",
                        dpi=72,
                        size=Size(width=float(img.width), height=float(img.height)),
                        uri=from_pil_to_base64uri(img),
                    )
                    page_item = PageItem(
                        page_no=1,
                        size=Size(width=float(img.width), height=float(img.height)),
                        image=image_ref,
                    )
                    true_doc.pages[1] = page_item

                    # Create key-value pairs if present
                    kv_pairs = self.create_kv_pairs(doc)
                    if kv_pairs:
                        self.populate_key_value_item(true_doc, kv_pairs)

                    # Process layout elements
                    current_list = None
                    boxes = doc["boxes"]
                    labels = list(
                        map(
                            lambda label: label.lower()
                            .replace("-", "_")
                            .replace(" ", "_"),
                            doc["labels"],
                        )
                    )
                    segments = doc["segments"]

                    for label, segment, box in zip(labels, segments, boxes):
                        current_list = self.update_doc(
                            true_doc, current_list, img, label, segment, box
                        )

                    # Extract images from ground truth document
                    true_doc, true_pictures, true_page_images = extract_images(
                        document=true_doc,
                        pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                        page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                    )

                    # Create dataset record
                    record = DatasetRecord(
                        doc_id=doc_id,
                        doc_hash=get_binhash(img_bytes),
                        ground_truth_doc=true_doc,
                        original=DocumentStream(
                            name=doc_id, stream=io.BytesIO(img_bytes)
                        ),
                        mime_type=f"image/{img.format.lower() if img.format else 'png'}",
                        modalities=[
                            EvaluationModality.LAYOUT,
                            EvaluationModality.MARKDOWN_TEXT,
                            EvaluationModality.KEY_VALUE,
                        ],
                        ground_truth_pictures=true_pictures,
                        ground_truth_page_images=true_page_images,
                    )

                    yield record

                except Exception as ex:
                    _log.error(f"Error processing document {i}: {str(ex)}")
                    continue

        except Exception as ex:
            _log.error(f"Error loading dataset: {str(ex)}")
            raise
