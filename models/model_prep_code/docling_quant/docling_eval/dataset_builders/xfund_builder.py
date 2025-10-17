import io
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import DownloadManager
from docling_core.types import DoclingDocument
from docling_core.types.doc import BoundingBox, ImageRef, PageItem, ProvenanceItem, Size
from docling_core.types.doc.document import GraphCell, GraphData, GraphLink
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel
from PIL import Image
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns, EvaluationModality
from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
from docling_eval.utils.utils import (
    classify_cells,
    extract_images,
    from_pil_to_base64uri,
    get_binhash,
    sort_cell_ids,
)

# Get logger
_log = logging.getLogger(__name__)


class XFUNDDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    XFUND Dataset builder implementing the base dataset builder interface.

    XFUND is a multilingual form understanding dataset that includes forms in
    multiple languages: Chinese, Japanese, Spanish, French, Italian, German, and Portuguese.
    """

    def __init__(
        self,
        dataset_source: Path,
        target: Path,
        split: str = "val",  # XFUND uses "val" instead of "test"
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the XFUND dataset builder.

        Args:
            dataset_source: Path to the dataset source
            target: Path where processed dataset will be saved
            split: Dataset split to use ("val" for XFUND)
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="XFUND",
            dataset_source=dataset_source,
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )
        self._langs = [
            "zh",
            "de",
            "es",
            "fr",
            "it",
            "ja",
            "pt",
        ]  # Fixed supported languages

        self.must_retrieve = True

    def retrieve_input_dataset(self) -> Path:
        """
        Download and extract the XFUND dataset if needed.

        Returns:
            Path to the retrieved dataset
        """
        assert isinstance(self.dataset_source, Path)
        dataset_path = self.dataset_source

        # Check if the dataset already exists
        if not dataset_path.exists() or not (dataset_path / self.split).exists():
            _log.info(f"Downloading XFUND dataset to {dataset_path}")

            # Ensure the dataset path exists
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Create split directory
            split_dir = dataset_path / self.split
            split_dir.mkdir(exist_ok=True)

            # Base URL for XFUND dataset
            _URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0/"

            # Download manager
            dl_manager = DownloadManager()

            # Download and process each language
            for lang in self._langs:
                try:
                    # Download JSON annotations
                    json_url = f"{_URL}{lang}.{self.split}.json"
                    json_path = dl_manager.download(json_url)

                    # Copy JSON to the split directory
                    shutil.copy(json_path, split_dir / f"{lang}.{self.split}.json")

                    # Download and extract ZIP with images
                    zip_url = f"{_URL}{lang}.{self.split}.zip"
                    zip_path = dl_manager.download(zip_url)

                    # Extract ZIP contents to split directory
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(split_dir)

                except Exception as e:
                    _log.error(f"Error downloading {lang} data: {str(e)}")

            _log.info(f"XFUND dataset downloaded to {dataset_path}")

        self.retrieved = True
        return dataset_path

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

    def create_graph_link(
        self,
        key_cell: GraphCell,
        value_cell: GraphCell,
        label: GraphLinkLabel = GraphLinkLabel.TO_VALUE,
    ) -> GraphLink:
        """
        Create a graph link between key and value cells.

        Args:
            key_cell: Source cell (key)
            value_cell: Target cell (value)
            label: Link label

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
            cell_dict: Dictionary mapping cell IDs to GraphCell objects

        Returns:
            BoundingBox encompassing all cells, or None if no bounding boxes
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
        self, doc: DoclingDocument, xfund_data: dict
    ) -> DoclingDocument:
        """
        Populate the key-value item from the XFUND data.

        Args:
            doc: DoclingDocument to update
            xfund_data: XFUND annotation data

        Returns:
            Updated DoclingDocument
        """
        if "document" not in xfund_data:
            raise ValueError("Invalid XFUND data: missing 'document' key.")

        form_items = xfund_data["document"]

        cell_by_id = {}
        for item in form_items:
            # We omit the items that are not relevant for key-value pairs.
            if not item.get("linking", []) and item.get("label", "other") in [
                "header",
                "other",
            ]:
                continue
            cell_id = item["id"]
            cell_text = item.get("text", "")

            bbox_instance = None
            if item.get("box") is not None:
                bbox_instance = self.convert_bbox(item.get("box"))
                cell_prov = ProvenanceItem(
                    page_no=doc.pages[1].page_no,
                    charspan=(0, 0),
                    bbox=bbox_instance,
                )
            else:
                cell_prov = None

            cell = GraphCell(
                cell_id=cell_id,
                text=cell_text,
                orig=cell_text,
                prov=cell_prov,
                label=GraphCellLabel.KEY,  # later to be updated by classify_cells
            )
            cell_by_id[cell_id] = cell

        # Unique linking pairs
        linking_set = set()
        for item in form_items:
            linking = item.get("linking", [])
            for pair in linking:
                if isinstance(pair, list) and len(pair) == 2:
                    linking_set.add(tuple(pair))  # (source_id, target_id)

        # Create links using the cell mapping
        links = []
        for src, tgt in linking_set:
            if src in cell_by_id and tgt in cell_by_id:
                # Mark target as value
                cell_by_id[tgt].label = GraphCellLabel.VALUE
                kv_link = self.create_graph_link(cell_by_id[src], cell_by_id[tgt])
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

        # Update cell labels based on linking structure
        classify_cells(graph=graph)

        doc.add_key_values(graph=graph, prov=prov)

        sort_cell_ids(doc)

        return doc

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Iterate through the dataset and yield DatasetRecord objects.

        Yields:
            DatasetRecord objects
        """
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert isinstance(self.dataset_source, Path)

        # Get split directory
        split_dir = self.dataset_source / self.split

        # Load all JSON files in the split directory
        json_files = list(split_dir.glob("*.json"))
        all_documents = []

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_documents.extend(data.get("documents", []))

        # Sort documents by image filename for deterministic ordering
        all_documents = sorted(all_documents, key=lambda doc: doc["img"]["fname"])
        total_documents = len(all_documents)

        # Apply index range
        begin, end = self.get_effective_indices(total_documents)
        selected_documents = all_documents[begin:end]

        # Log stats
        self.log_dataset_stats(total_documents, len(selected_documents))
        _log.info(
            f"Processing XFUND {self.split} dataset: {len(selected_documents)} documents"
        )

        # Process each document
        for doc_data in tqdm(selected_documents, total=len(selected_documents)):
            try:
                # Get image path
                img_filename = doc_data["img"]["fname"]
                img_path = split_dir / img_filename

                # Load image
                img = Image.open(img_path)

                # Get image bytes
                with io.BytesIO() as img_byte_stream:
                    img.save(img_byte_stream, format=img.format)
                    img_byte_stream.seek(0)
                    img_bytes = img_byte_stream.getvalue()

                # Create ground truth document
                true_doc = DoclingDocument(name=Path(img_filename).stem)

                assert img.format is not None

                # Add page with image
                image_ref = ImageRef(
                    mimetype=f"image/{img.format.lower()}",
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

                # Populate document with key-value data
                true_doc = self.populate_key_value_item(true_doc, doc_data)

                # Extract images
                true_doc, true_pictures, true_page_images = extract_images(
                    document=true_doc,
                    pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                    page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                )

                assert img.format is not None
                # Create dataset record
                record = DatasetRecord(
                    doc_id=Path(img_filename).stem,
                    doc_hash=get_binhash(img_bytes),
                    ground_truth_doc=true_doc,
                    original=None,  # No original PDF/document stream
                    mime_type=f"image/{img.format.lower()}",
                    modalities=[EvaluationModality.KEY_VALUE],
                    ground_truth_pictures=true_pictures,
                    ground_truth_page_images=true_page_images,
                )

                yield record

            except Exception as ex:
                _log.error(f"Error processing document: {str(ex)}")
