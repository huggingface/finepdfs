import io
import json
import logging
import shutil
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


class FUNSDDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    FUNSD Dataset builder implementing the base dataset builder interface.

    This builder handles the Form Understanding in Noisy Scanned Documents (FUNSD) dataset,
    which contains form annotations for form understanding tasks.
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
        Initialize the FUNSD dataset builder.

        Args:
            dataset_source: Path to the dataset source
            target: Path where processed dataset will be saved
            split: Dataset split to use ('train' or 'test')
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="FUNSD",
            dataset_source=dataset_source,
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        self.must_retrieve = True

    def retrieve_input_dataset(self) -> Path:
        """
        Download and extract the FUNSD dataset if needed.

        Returns:
            Path to the retrieved dataset
        """
        assert isinstance(self.dataset_source, Path)
        dataset_path = self.dataset_source

        # Check if the dataset already exists
        if not dataset_path.exists() or not (dataset_path / "training_data").exists():
            _log.info(f"Downloading FUNSD dataset to {dataset_path}")

            # Ensure the dataset path exists
            dataset_path.mkdir(parents=True, exist_ok=True)

            # Download and extract
            dl_manager = DownloadManager()
            extracted_path = dl_manager.download_and_extract(
                "https://drive.google.com/uc?export=download&id=1wdJJQgRIb1c404SJnX1dyBSi7U2mVduI"
            )

            # Move the extracted FUNSD folder to the target location
            extracted_path = Path(extracted_path)
            funsd_dir = extracted_path / "FUNSD"

            # If target exists, remove it first
            if dataset_path.exists():
                shutil.rmtree(dataset_path)

            # Move extracted folder to target location
            shutil.move(str(funsd_dir), str(dataset_path))

            # Fix annotation directories
            train_adj = dataset_path / "training_data" / "adjusted_annotations"
            train_ann = dataset_path / "training_data" / "annotations"
            test_adj = dataset_path / "testing_data" / "adjusted_annotations"
            test_ann = dataset_path / "testing_data" / "annotations"

            if train_adj.exists():
                if train_ann.exists():
                    shutil.rmtree(train_ann)
                shutil.move(str(train_adj), str(train_ann))

            if test_adj.exists():
                if test_ann.exists():
                    shutil.rmtree(test_ann)
                shutil.move(str(test_adj), str(test_ann))

            _log.info(f"FUNSD dataset downloaded to {dataset_path}")

            shutil.rmtree(extracted_path)

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
        self, doc: DoclingDocument, funsd_data: dict
    ) -> DoclingDocument:
        """
        Populate the key-value item from the FUNSD data.

        Args:
            doc: DoclingDocument to update
            funsd_data: FUNSD annotation data

        Returns:
            Updated DoclingDocument
        """
        if "form" not in funsd_data:
            raise ValueError("Invalid FUNSD data: missing 'form' key.")

        form_items = funsd_data["form"]

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

        # unique linking pairs
        linking_set = set()
        for item in form_items:
            linking = item.get("linking", [])
            for pair in linking:
                if isinstance(pair, list) and len(pair) == 2:
                    linking_set.add(tuple(pair))  # (source_id, target_id)

        # creation of links using the cell mapping
        links = []
        for src, tgt in linking_set:
            if src in cell_by_id and tgt in cell_by_id:
                # later to be updated by classify_cells
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

        # Determine image directory based on split
        if self.split == "train":
            image_dir = self.dataset_source / "training_data" / "images"
        elif self.split == "test":
            image_dir = self.dataset_source / "testing_data" / "images"
        else:
            raise ValueError(f"Invalid split: {self.split}. Expected 'train' or 'test'")

        # List all PNG images
        images = sorted(list(image_dir.glob("*.png")))
        total_images = len(images)

        # Apply index range
        begin, end = self.get_effective_indices(total_images)
        images = images[begin:end]

        # Log stats
        self.log_dataset_stats(total_images, len(images))
        _log.info(f"Processing FUNSD {self.split} dataset: {len(images)} images")

        # Process each image
        for img_path in tqdm(images, total=len(images)):
            try:
                # Determine annotation path
                annotation_path = (
                    img_path.parent.parent
                    / "annotations"
                    / img_path.name.replace(".png", ".json")
                )

                # Load image and annotation
                img = Image.open(img_path)
                with open(annotation_path, "r", encoding="utf-8") as f:
                    funsd_data = json.load(f)

                # Get image bytes
                with io.BytesIO() as img_byte_stream:
                    img.save(img_byte_stream, format="PNG")
                    img_byte_stream.seek(0)
                    img_bytes = img_byte_stream.getvalue()

                # Create ground truth document
                true_doc = DoclingDocument(name=img_path.stem)

                # Add page with image
                image_ref = ImageRef(
                    mimetype="image/png",
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
                true_doc = self.populate_key_value_item(true_doc, funsd_data)

                # Extract images
                true_doc, true_pictures, true_page_images = extract_images(
                    document=true_doc,
                    pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                    page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                )

                # Create dataset record
                record = DatasetRecord(
                    doc_id=img_path.stem,
                    doc_hash=get_binhash(img_bytes),
                    ground_truth_doc=true_doc,
                    original=None,
                    mime_type="image/png",
                    modalities=[EvaluationModality.KEY_VALUE],
                    ground_truth_pictures=true_pictures,
                    ground_truth_page_images=true_page_images,
                )

                yield record

            except Exception as ex:
                _log.error(f"Error processing {img_path.name}: {str(ex)}")
                raise ex
