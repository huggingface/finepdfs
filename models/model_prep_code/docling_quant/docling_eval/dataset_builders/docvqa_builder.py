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
    PageItem,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import (
    GraphCell,
    GraphData,
    GraphLink,
    KeyValueItem,
)
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel
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


class DocVQADatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    DocVQA dataset builder implementing the base dataset builder interface.

    This builder processes the DocVQA dataset, which contains document
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
        Initialize the DocVQA dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="DocVQA",
            dataset_source=HFSource(repo_id="lmms-lab/DocVQA"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

    def _process_document(self, doc_id, qa_items) -> DatasetRecord:
        """Process all QA items for a single document."""
        _log.debug(f"Processing document: {doc_id}")

        doc = DoclingDocument(name=f"{doc_id}")
        image: PIL.Image.Image = qa_items[0]["image"]
        image = image.convert("RGB")
        image_ref = ImageRef(
            mimetype="image/png",
            dpi=72,
            size=Size(width=image.width, height=image.height),
            uri=from_pil_to_base64uri(image),
        )
        page_item = PageItem(
            page_no=1,
            size=Size(width=float(image.width), height=float(image.height)),
            image=image_ref,
        )

        doc.pages[1] = page_item

        cells = []
        links = []
        index = 0

        for qa_item in qa_items:
            _log.debug(f"  Processing QA item data...")
            cells.append(
                GraphCell(
                    label=GraphCellLabel.KEY,
                    cell_id=index,
                    text=qa_item["question"],
                    orig=qa_item["question"],
                )
            )

            answer_index = index + 1
            for answer in qa_item["answers"]:
                cells.append(
                    GraphCell(
                        label=GraphCellLabel.VALUE,
                        cell_id=answer_index,
                        text=answer,
                        orig=answer,
                    )
                )
                links.extend(
                    [
                        GraphLink(
                            label=GraphLinkLabel.TO_VALUE,
                            source_cell_id=index,
                            target_cell_id=answer_index,
                        ),
                        GraphLink(
                            label=GraphLinkLabel.TO_KEY,
                            source_cell_id=answer_index,
                            target_cell_id=index,
                        ),
                    ]
                )
                answer_index += 1

            index = answer_index

        graph = GraphData(cells=cells, links=links)
        doc.add_key_values(graph)

        # Extract images from the ground truth document
        doc, true_pictures, true_page_images = extract_images(
            document=doc,
            pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
            page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
        )

        # Convert image to bytes for storage
        with io.BytesIO() as img_byte_stream:
            image.save(img_byte_stream, format="PNG")
            img_byte_stream.seek(0)
            img_bytes = img_byte_stream.getvalue()

        # Create dataset record
        record = DatasetRecord(
            doc_id=str(doc_id),
            doc_hash=get_binhash(img_bytes),
            ground_truth_doc=doc,
            original=DocumentStream(name=str(doc_id), stream=io.BytesIO(img_bytes)),
            mime_type="image/png",
            modalities=[
                EvaluationModality.LAYOUT,
                EvaluationModality.QUESTION_ANSWERING,
            ],
            ground_truth_pictures=true_pictures,
            ground_truth_page_images=true_page_images,
        )

        return record

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Iterate through the dataset and yield DatasetRecord objects.

        Yields:
            DatasetRecord objects
        """
        assert isinstance(self.dataset_source, HFSource)

        path = self.dataset_source.repo_id
        if self.dataset_local_path is not None:
            path = str(self.dataset_local_path)
        # Load dataset from the retrieved path
        ds = load_dataset(path, split=self.split, name="DocVQA")

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

        sorted_dataset = ds.sort("docId")

        # Initialize variables
        current_doc_id = None
        current_doc_qa_items = []  # type: ignore

        # Iterate through the sorted dataset
        for sample in tqdm(
            sorted_dataset,
            total=selected_ds_len,
            ncols=128,
            desc="Processing DocVQA records...",
        ):
            # Check if we've moved to a new docId
            if sample["docId"] != current_doc_id:
                # Process the previous doc's QA items (skip first iteration)
                if current_doc_qa_items:
                    rec = self._process_document(current_doc_id, current_doc_qa_items)
                    yield rec
                    exported_rows += 1

                # Start a new document group
                current_doc_id = sample["docId"]
                current_doc_qa_items = [sample]
            else:
                current_doc_qa_items.append(sample)

        # Process the final document group
        if current_doc_qa_items:
            rec = self._process_document(current_doc_id, current_doc_qa_items)
            yield rec
            exported_rows += 1

        _log.info(
            "Exported rows: %s. Skipped rows: %s.",
            exported_rows,
            skipped_rows,
        )
