import io
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Set

import PIL.Image
from datasets import load_dataset
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    GroupItem,
    GroupLabel,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.io import DocumentStream
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns, EvaluationModality
from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval.utils.utils import (
    add_pages_to_true_doc,
    crop_bounding_box,
    extract_images,
    from_pil_to_base64uri,
    get_binhash,
)

# Get logger
_log = logging.getLogger(__name__)

# Blacklisted document IDs (documents with known issues)
BLACKLISTED_DOC_IDS: Set[str] = {
    "f556167ac3284665652050b1b0bc1e6f5af27f54f17f27566c60c80f6f134a92",
    "dbc51622cbe9b8766f44db3b3fda8d0a745da06b9bfec9935bd003d2bdd494c8",
    "d4c0401fffc04d24e629a9fada23266a3b492ea63e889641b3c33adf815d44e3",
    "cc93b556f49af1f2e366719ec98a131186c16385545d8062d21e4d38b6bf7686",
    "c9755e6972e3150a1c02565ec8070bfc26503d0fe09d056e418d6dcd6ea43cd9",
    "c90d298ac9493e3804baf1b62c9321cdabf388c29eb504c5ad12106b3cdf530b",
    "c2b513a5611d3138726e679c6e2e9e5383e4d3d82a2c588bbe3d5802797e2765",
    "b72bb61059b06ff9859ae023aa66cdb3ff706c354ac72ca5d3c837e107d0a384",
    "b4f5d430d89499474a31f39fe8eb615fdcd7aa682eb0b959a0384206d5c8174c",
    "ab9315a0610ec0e5446a7062cd99a9e137efe3d7da9a7bffa2523894ac68751a",
    "99723d3d3c61db030dbd813faec67579ceb50c6b5dd8c2f500c6e073849e9784",
    "87c7dc9ca13016fafa4a7539efa1bf00401ba27323a473094b4184bc42cb36c0",
    "7c1fa2e7c81a81888c18eb95cfe37edb82a91dd340e75c8123618a6774081f2e",
    "7a231e9b7d841935a142d972ea1c7546d613cba18e301b0e07415f9eb44e3382",
    "5793282eaaa089d0dc71e67c951c68b4157a212cc43edbc3106323e96b385190",
    "55f9167173149b0b4c8d8951baca190ee756450d6565a91655ec04967a08c798",
    "5003688e1ae61558cbeda741d246804b59fe89dac29cf508b4b6ce56d1a4342b",
    "4f6e20223b7bc8436c623b9e6282db6ebd5f221aeb880a8db9b4544326d5a8a6",
    "4232e47097e6ecfdf53d4097cb90bdd56cc63c31508a2f91a6d3908770a4d1ed",
    "3361796dba75fe2c641c43db12ab31a0eb9dbcbbc7c99721288d36c41d759bcd",
    "1fadb433bffa31c43817d1f6bafbb10dff53422ad046d391ed560ebef13d9f83",
    "1a8f46903dbe89dc5b6df43389b4895a376e00ab3b90c7c37f1a1b561d3d51a1",
    "1763e54be635759ccb66ebb462548f8a40d44567f62cecc5ca26f22acd28e823",
    "048a570b2e415b653a62313ef82504adfda480c99f69826fcbeb67758ea3c7a4",
    "0261791e343389682847c913a16789776d0ba41a584901571846c7ddab3cbaa6",
}

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
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class DocLayNetV1DatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    DocLayNet V1 dataset builder implementing the base dataset builder interface.

    This builder processes the DocLayNet V1.2 dataset, which contains document
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
        Initialize the DocLayNet V1 dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="DocLayNetV1",
            dataset_source=HFSource(repo_id="ds4sd/DocLayNet-v1.2"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )
        self.blacklisted_ids = set(BLACKLISTED_DOC_IDS)
        self.category_map = {
            1: "caption",
            2: "footnote",
            3: "formula",
            4: "list_item",
            5: "page_footer",
            6: "page_header",
            7: "picture",
            8: "section_header",
            9: "table",
            10: "text",
            11: "title",
        }

    @staticmethod
    def ltwh_to_ltrb(box: List[float]) -> List[float]:
        """
        Convert left, top, width, height format to left, top, right, bottom.

        Args:
            box: Box in [left, top, width, height] format

        Returns:
            Box in [left, top, right, bottom] format
        """
        l, t, w, h = box
        return [l, t, l + w, t + h]

    def update_doc_with_gt(
        self,
        doc: DoclingDocument,
        current_list: Optional[GroupItem],  # This was incorrectly typed as str | None
        img: PIL.Image.Image,
        old_size: Size,
        label_str: str,
        box: List[float],
        content: str,
    ) -> Optional[GroupItem]:  # Return type should match the parameter type
        """
        Add an element to the document based on its label type.

        Args:
            doc: DoclingDocument to update
            current_list: Current list group for list items
            img: Page image
            old_size: Original page size
            label_str: Element label as string
            box: Bounding box coordinates
            content: Text content

        Returns:
            Updated list group or None
        """
        # Map string label to DocItemLabel
        label_map = {
            "caption": DocItemLabel.CAPTION,
            "footnote": DocItemLabel.FOOTNOTE,
            "formula": DocItemLabel.FORMULA,
            "list_item": DocItemLabel.LIST_ITEM,
            "page_footer": DocItemLabel.PAGE_FOOTER,
            "page_header": DocItemLabel.PAGE_HEADER,
            "picture": DocItemLabel.PICTURE,
            "section_header": DocItemLabel.SECTION_HEADER,
            "table": DocItemLabel.TABLE,
            "text": DocItemLabel.TEXT,
            "title": DocItemLabel.TITLE,
        }

        label = label_map.get(label_str, DocItemLabel.TEXT)

        # Create bounding box
        w, h = img.size
        new_size = Size(width=w, height=h)
        bbox = (
            BoundingBox.from_tuple(tuple(self.ltwh_to_ltrb(box)), CoordOrigin.TOPLEFT)
            .to_bottom_left_origin(page_height=old_size.height)
            .scale_to_size(old_size=old_size, new_size=new_size)
        )

        # Create provenance
        prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(content)))

        # Crop the relevant part of the image
        img_elem = crop_bounding_box(page_image=img, page=doc.pages[1], bbox=bbox)

        # Handle element based on its label
        if label == DocItemLabel.PICTURE:
            current_list = None
            try:
                uri = from_pil_to_base64uri(img_elem)
                imgref = ImageRef(
                    mimetype="image/png",
                    dpi=72,
                    size=Size(width=img_elem.width, height=img_elem.height),
                    uri=uri,
                )
                doc.add_picture(prov=prov, image=imgref)
            except Exception as e:
                _log.error(
                    "Failed to create image reference for %s: %s", doc.name, str(e)
                )

        elif label == DocItemLabel.TABLE:
            current_list = None
            tbl_cell = TableCell(
                start_row_offset_idx=0,
                end_row_offset_idx=0,
                start_col_offset_idx=0,
                end_col_offset_idx=0,
                text=content,
            )
            tbl_data = TableData(table_cells=[tbl_cell])
            doc.add_table(data=tbl_data, caption=None, prov=prov)

        elif label == DocItemLabel.LIST_ITEM:
            if current_list is None:
                current_list = doc.add_group(label=GroupLabel.LIST, name="list")
            doc.add_list_item(
                text=content, enumerated=False, prov=prov, parent=current_list
            )

        elif label == DocItemLabel.SECTION_HEADER:
            current_list = None
            doc.add_heading(text=content, orig=content, level=2, prov=prov)

        elif label == DocItemLabel.TITLE:
            current_list = None
            doc.add_heading(text=content, orig=content, level=1, prov=prov)

        else:
            current_list = None
            doc.add_text(label=label, text=content, orig=content, prov=prov)

        return current_list

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Iterate through the dataset and yield DatasetRecord objects.

        Yields:
            DatasetRecord objects
        """

        path = "ds4sd/DocLayNet-v1.2"
        if self.dataset_local_path is not None:
            path = str(self.dataset_local_path)
        # Load dataset from the retrieved path
        ds = load_dataset(path, split=self.split)

        # Apply HuggingFace's select method for index ranges
        total_ds_len = len(ds)
        begin, end = self.get_effective_indices(total_ds_len)

        # Select the range (HuggingFace datasets have a convenient select method)
        ds = ds.select(range(begin, end))
        selected_ds_len = len(ds)

        # Log stats
        self.log_dataset_stats(total_ds_len, selected_ds_len)

        skipped_rows = 0
        exported_rows = 0

        # Process each document
        for doc in tqdm(
            ds,
            total=selected_ds_len,
            ncols=128,
            desc="Processing DocLayNetV1 documents",
        ):
            try:
                page_hash = doc["metadata"]["page_hash"]

                # Skip blacklisted documents
                if page_hash in self.blacklisted_ids:
                    _log.info("Skip blacklisted doc id: %s", page_hash)
                    continue

                # Get PDF data
                pdf = doc["pdf"]
                pdf_stream = io.BytesIO(pdf)

                # Create ground truth document
                true_doc = DoclingDocument(name=page_hash)
                true_doc, true_page_images = add_pages_to_true_doc(
                    pdf_path=pdf_stream, true_doc=true_doc, image_scale=1.0
                )

                # Set up document dimensions
                img = true_page_images[0]
                old_w, old_h = doc["image"].size
                old_size = Size(width=old_w, height=old_h)

                # Process elements
                current_list = None
                labels = list(
                    map(lambda cid: self.category_map[int(cid)], doc["category_id"])
                )
                bboxes = doc["bboxes"]
                segments = doc["pdf_cells"]
                contents = [
                    " ".join(map(lambda cell: cell["text"], cells))
                    for cells in segments
                ]

                for l, b, c in zip(labels, bboxes, contents):
                    current_list = self.update_doc_with_gt(
                        true_doc, current_list, img, old_size, l, b, c
                    )

                # Extract images from the ground truth document
                true_doc, true_pictures, true_page_images = extract_images(
                    document=true_doc,
                    pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                    page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                )

                pdf_stream.seek(0)
                doc_stream = DocumentStream(name=page_hash, stream=pdf_stream)

                # Create dataset record
                record = DatasetRecord(
                    doc_id=page_hash,
                    doc_hash=get_binhash(pdf),
                    ground_truth_doc=true_doc,
                    original=doc_stream,
                    mime_type="application/pdf",
                    modalities=[
                        EvaluationModality.LAYOUT,
                        EvaluationModality.MARKDOWN_TEXT,
                    ],
                    ground_truth_pictures=true_pictures,
                    ground_truth_page_images=true_page_images,
                )

                exported_rows += 1

                yield record

            except Exception as ex:
                _log.error("Error processing document: %s", str(ex))
                skipped_rows += 1

        _log.info(
            "Exported rows: %s. Skipped rows: %s.",
            exported_rows,
            skipped_rows,
        )
