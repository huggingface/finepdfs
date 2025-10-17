import io
import logging
from pathlib import Path
from typing import Any, Iterable, List

from datasets import load_dataset
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    ImageRef,
    PageItem,
    ProvenanceItem,
    Size,
)
from docling_core.types.io import DocumentStream
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import (
    BenchMarkColumns,
    EvaluationModality,
    PageTokens,
)
from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval.utils.utils import (
    convert_html_table_into_docling_tabledata,
    extract_images,
    from_pil_to_base64uri,
    get_binhash,
)

_log = logging.getLogger(__name__)


class TableDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    Base class for table dataset builders.

    This class provides common functionality for building datasets
    focused on table structure recognition tasks.
    """

    def __init__(
        self,
        name: str,
        dataset_source: HFSource,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the table dataset builder.

        Args:
            name: Name of the dataset
            dataset_source: HuggingFace dataset source
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name=name,
            dataset_source=dataset_source,
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

    def retrieve_input_dataset(self) -> Path:
        """
        Download and extract the dataset.

        Returns:
            Path to the retrieved dataset
        """
        assert isinstance(self.dataset_source, HFSource)
        dataset_path = super().retrieve_input_dataset()
        self.retrieved = True
        return dataset_path

    def create_page_tokens(
        self, data: List[Any], height: float, width: float
    ) -> PageTokens:
        """
        Create page tokens from cell data.

        Args:
            data: Table cell data
            height: Page height
            width: Page width

        Returns:
            PageTokens object containing token information
        """
        tokens = []
        cnt = 0
        for i, row in enumerate(data):
            for j, item in enumerate(row):
                text = "".join(item["tokens"])
                tokens.append(
                    {
                        "bbox": {
                            "l": item["bbox"][0],
                            "t": item["bbox"][1],
                            "r": item["bbox"][2],
                            "b": item["bbox"][3],
                            "coord_origin": str(CoordOrigin.TOPLEFT.value),
                        },
                        "text": text,
                        "id": cnt,
                    }
                )
                cnt += 1
        result = {"tokens": tokens, "height": height, "width": width}
        return PageTokens.model_validate(result)

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

        assert isinstance(self.dataset_source, HFSource)
        # Load dataset from the retrieved path

        path = self.dataset_source.repo_id
        if self.dataset_local_path:
            path = str(self.dataset_local_path)

        ds = load_dataset(path, split=self.split)

        # Apply index range
        total_items = len(ds)
        begin, end = self.get_effective_indices(total_items)

        # Use HuggingFace's select method for applying range
        ds = ds.select(range(begin, end))
        selected_items = len(ds)

        # Log stats
        self.log_dataset_stats(total_items, selected_items)
        _log.info(f"Processing {self.name} dataset: {selected_items} items")

        for item in tqdm(ds, desc=f"Processing {self.name} dataset"):
            try:
                filename = item["filename"]
                table_image = item["image"]

                page_tokens = self.create_page_tokens(
                    data=item["cells"],
                    height=table_image.height,
                    width=table_image.width,
                )

                # Create ground truth document
                true_doc = DoclingDocument(name=f"ground-truth {filename}")

                # Add page to document
                page_index = 1
                image_ref = ImageRef(
                    mimetype="image/png",
                    dpi=72,
                    size=Size(
                        width=float(table_image.width), height=float(table_image.height)
                    ),
                    uri=from_pil_to_base64uri(table_image),
                )
                page_item = PageItem(
                    page_no=page_index,
                    size=Size(
                        width=float(table_image.width), height=float(table_image.height)
                    ),
                    image=image_ref,
                )
                true_doc.pages[1] = page_item

                # Create table data
                html = "<table>" + "".join(item["html"]) + "</table>"
                table_data = convert_html_table_into_docling_tabledata(
                    html, text_cells=item["cells"][0]
                )

                for tbl_cell, page_token in zip(
                    table_data.table_cells, page_tokens.tokens, strict=True
                ):
                    tbl_cell.bbox = page_token.bbox

                # Create bounding box for table
                l = 0.0
                b = 0.0
                r = table_image.width
                t = table_image.height
                if "table_bbox" in item:
                    l = item["table_bbox"][0]
                    b = table_image.height - item["table_bbox"][3]
                    r = item["table_bbox"][2]
                    t = table_image.height - item["table_bbox"][1]

                bbox = BoundingBox(
                    l=l,
                    r=r,
                    b=b,
                    t=t,
                    coord_origin=CoordOrigin.BOTTOMLEFT,
                )

                # Create provenance
                prov = ProvenanceItem(page_no=page_index, bbox=bbox, charspan=(0, 0))

                # Add table to document
                true_doc.add_table(data=table_data, caption=None, prov=prov)

                # Extract images
                true_doc, true_pictures, true_page_images = extract_images(
                    document=true_doc,
                    pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                    page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                )

                # Create dataset record
                with io.BytesIO() as img_byte_stream:
                    table_image.save(img_byte_stream, format="PNG")
                    img_byte_stream.seek(0)
                    img_bytes = img_byte_stream.read()

                record = DatasetRecord(
                    doc_id=str(Path(filename).stem),
                    doc_hash=get_binhash(img_bytes),
                    ground_truth_doc=true_doc,
                    original=DocumentStream(
                        name=filename, stream=io.BytesIO(img_bytes)
                    ),
                    mime_type="image/png",
                    modalities=[EvaluationModality.TABLE_STRUCTURE],
                    ground_truth_pictures=true_pictures,
                    ground_truth_page_images=true_page_images,
                )

                yield record
            except Exception as ex:
                _log.error(
                    f"Error processing item {item.get('filename', 'unknown')}: {str(ex)}"
                )


class FintabNetDatasetBuilder(TableDatasetBuilder):
    """Dataset builder for FinTabNet."""

    def __init__(
        self,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the FinTabNet dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="FinTabNet",
            dataset_source=HFSource(repo_id="ds4sd/FinTabNet_OTSL"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )


class PubTabNetDatasetBuilder(TableDatasetBuilder):
    """Dataset builder for PubTabNet."""

    def __init__(
        self,
        target: Path,
        split: str = "val",  # PubTabNet uses "val" instead of "test"
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the PubTabNet dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="PubTabNet",
            dataset_source=HFSource(repo_id="ds4sd/PubTabNet_OTSL"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )


class PubTables1MDatasetBuilder(TableDatasetBuilder):
    """Dataset builder for PubTables-1M."""

    def __init__(
        self,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the PubTables-1M dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="PubTables-1M",
            dataset_source=HFSource(repo_id="ds4sd/PubTables-1M_OTSL-v1.1"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )
