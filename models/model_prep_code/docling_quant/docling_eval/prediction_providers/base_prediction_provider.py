import copy
import logging
import sys
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

from datasets import load_dataset
from docling.datamodel.base_models import ConversionStatus
from docling.utils.utils import chunkify
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.utils.utils import (
    extract_images,
    insert_images_from_pil,
    save_shard_to_disk,
    write_datasets_info,
)
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters

# Get logger
_log = logging.getLogger(__name__)

# Default HTML export labels for visualization
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
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class BasePredictionProvider:
    """
    Base class for all prediction providers.

    Prediction providers are responsible for generating predictions from input data
    in the form of DoclingDocument objects or other formats.
    """

    def __init__(
        self,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
    ):
        """
        Initialize the prediction provider.

        Args:
            do_visualization: Whether to generate visualizations of predictions
            ignore_missing_predictions: Whether to ignore records with missing predictions
            true_labels: Set of DocItemLabel to use for ground truth visualization
            pred_labels: Set of DocItemLabel to use for prediction visualization
        """
        self.do_visualization = do_visualization
        self.ignore_missing_predictions = ignore_missing_predictions

        # Label sets for visualization
        self.true_labels = true_labels or TRUE_HTML_EXPORT_LABELS
        self.pred_labels = pred_labels or PRED_HTML_EXPORT_LABELS

    @abstractmethod
    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Generate a prediction for a dataset record.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction added
        """
        pred_record = self.create_dataset_record_with_prediction(
            record,
            DoclingDocument(name="dummy"),
            None,
        )
        return pred_record

    @abstractmethod
    def info(self) -> Dict[str, str]:
        """
        Get information about the prediction provider.

        Returns:
            Dictionary with provider information
        """
        return {}

    def visualize_results(
        self, prediction_record: DatasetRecordWithPrediction, target_dataset_dir: Path
    ) -> None:
        """
        Create visualizations of prediction results.

        Args:
            prediction_record: Record with prediction to visualize
            target_dataset_dir: Directory to save visualizations
        """
        if (
            prediction_record.predicted_doc is not None
            and prediction_record.ground_truth_page_images
        ):
            gt_doc = insert_images_from_pil(
                prediction_record.ground_truth_doc.model_copy(),
                prediction_record.ground_truth_pictures,
                prediction_record.ground_truth_page_images,
            )
            pred_doc = insert_images_from_pil(
                prediction_record.predicted_doc.model_copy(),
                prediction_record.predicted_pictures,
                prediction_record.predicted_page_images,
            )
            save_comparison_html_with_clusters(
                filename=target_dataset_dir
                / "visualizations"
                / f"{prediction_record.doc_id}.html",
                true_doc=gt_doc,
                pred_doc=pred_doc,
                page_image=prediction_record.ground_truth_page_images[0],
                true_labels=self.true_labels,
                pred_labels=self.pred_labels,
                draw_reading_order=True,
            )

    @property
    @abstractmethod
    def prediction_format(self) -> PredictionFormats:
        """
        Get the format of predictions generated by this provider.

        Returns:
            Prediction format enum value
        """
        pass

    def create_dataset_record_with_prediction(
        self,
        record: DatasetRecord,
        predicted_doc: Optional[DoclingDocument] = None,
        original_prediction: Optional[str] = None,
    ) -> DatasetRecordWithPrediction:
        """
        Create a dataset record with prediction from an input record.

        Args:
            record: Input dataset record
            predicted_doc: Predicted DoclingDocument
            original_prediction: Original prediction text/data

        Returns:
            Dataset record with prediction
        """
        pred_page_images = []
        pred_pictures = []
        if predicted_doc is not None:
            # Extract images from the ground truth document
            predicted_doc, pred_pictures, pred_page_images = extract_images(
                document=predicted_doc,
                pictures_column=BenchMarkColumns.PREDICTION_PICTURES.value,
                page_images_column=BenchMarkColumns.PREDICTION_PAGE_IMAGES.value,
            )

        data = {
            **record.as_record_dict(),
            "predicted_doc": predicted_doc,
            "predicted_page_images": pred_page_images,
            "predicted_pictures": pred_pictures,
            "original_prediction": original_prediction,
            "prediction_format": self.prediction_format,
            "predictor_info": self.info(),
        }
        return DatasetRecordWithPrediction.model_validate(data)

    def add_prediction(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Add a prediction to a dataset record.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction
        """
        # Copy the original input data to avoid modifying it
        input_data = copy.deepcopy(record.original)

        # Convert Path to DocumentStream if needed
        if not isinstance(input_data, DocumentStream):
            if isinstance(input_data, Path):
                input_data = DocumentStream(
                    name=input_data.name, stream=BytesIO(input_data.open("rb").read())
                )

        record.original = input_data
        pred_record = self.predict(record)

        return pred_record

    def get_effective_indices(
        self, total_items: int, begin_index: int, end_index: int
    ) -> Tuple[int, int]:
        """
        Calculate the effective begin and end indices based on dataset size.

        Args:
            total_items: Total number of items available
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all

        Returns:
            Tuple of (effective_begin_index, effective_end_index)
        """
        begin = begin_index if begin_index >= 0 else 0
        end = end_index if end_index > 0 else total_items
        end = min(end, total_items)

        if begin >= total_items:
            _log.warning(
                f"Begin index ({begin}) is greater than or equal to dataset size ({total_items}). "
                f"No items will be processed."
            )
            begin = total_items

        _log.info(
            f"Processing range [{begin}:{end}] out of {total_items} total items "
            f"({end - begin} items)"
        )

        return begin, end

    def create_prediction_dataset(
        self,
        name: str,
        gt_dataset_dir: Path,
        target_dataset_dir: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ) -> None:
        """
        Create a prediction dataset from a ground truth dataset.

        Args:
            name: Name of the dataset
            gt_dataset_dir: Path to ground truth dataset
            target_dataset_dir: Path to save prediction dataset
            split: Dataset split to process
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        # Load the dataset
        parquet_files = str(gt_dataset_dir / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})

        if ds is None:
            _log.error(f"Failed to load dataset from {parquet_files}")
            return

        ds_selection = ds[split]
        total_items = len(ds_selection)

        # Calculate effective indices
        begin, end = self.get_effective_indices(total_items, begin_index, end_index)

        # Apply range
        if begin > 0 or end < total_items:
            ds_selection = ds_selection.select(range(begin, end))

        selected_items = len(ds_selection)
        _log.info(
            f"Dataset '{name}' total items: {total_items}. "
            f"Selected range: [{begin}, {end}] = {selected_items} items"
        )

        def _iterate_predictions() -> Iterable[DatasetRecordWithPrediction]:
            """Generate predictions for each record in the dataset."""
            for i, data in tqdm(
                enumerate(ds_selection),
                desc="Creating predictions",
                ncols=120,
                total=len(ds_selection),
            ):
                try:
                    record = DatasetRecord.model_validate(data)
                    pred_record = self.add_prediction(record)

                    if (
                        self.ignore_missing_predictions
                        and pred_record.status == ConversionStatus.FAILURE
                    ):
                        continue

                    yield pred_record
                except Exception as e:
                    _log.error(f"Error processing record {i}: {str(e)}")
                    if not self.ignore_missing_predictions:
                        raise

        # Create output directories
        test_dir = target_dataset_dir / split
        test_dir.mkdir(parents=True, exist_ok=True)

        if self.do_visualization:
            (target_dataset_dir / "visualizations").mkdir(parents=True, exist_ok=True)

        # Process in chunks
        chunk_size = 80
        max_num_chunks = sys.maxsize

        count = 0
        chunk_count = 0
        for record_chunk in chunkify(_iterate_predictions(), chunk_size):
            if self.do_visualization:
                for r in record_chunk:
                    self.visualize_results(r, target_dataset_dir)

            record_chunk = [r.as_record_dict() for r in record_chunk]

            save_shard_to_disk(
                items=record_chunk, dataset_path=test_dir, shard_id=chunk_count
            )
            count += len(record_chunk)
            chunk_count += 1

            if chunk_count >= max_num_chunks:
                _log.info(
                    f"Reached maximum number of chunks ({max_num_chunks}). Stopping."
                )
                break

            # Write dataset info
            write_datasets_info(
                name=name,
                output_dir=target_dataset_dir,
                num_train_rows=0,
                num_test_rows=count,
            )

        _log.info(f"Saved {count} records in {chunk_count} chunks to {test_dir}")
