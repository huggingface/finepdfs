import logging
from pathlib import Path
from typing import Dict, Optional, Set

from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import (
    DoclingDocument,
    DocTagsDocument,
    DocTagsPage,
)
from PIL import Image

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import PredictionFormats
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)

_log = logging.getLogger(__name__)


class FilePredictionProvider(BasePredictionProvider):
    """
    Prediction provider that reads prediction files from a directory.

    This provider supports various file formats like DOCTAGS, MARKDOWN,
    JSON, and YAML.
    """

    def __init__(
        self,
        prediction_format: PredictionFormats,
        source_path: Path,
        do_visualization: bool = False,
        ignore_missing_files: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
        use_ground_truth_page_images: bool = False,
    ):
        """
        Initialize the file prediction provider.

        Args:
            prediction_format: Format of prediction files
            source_path: Path to directory containing prediction files
            do_visualization: Whether to generate visualizations
            ignore_missing_files: Whether to ignore missing files
            ignore_missing_predictions: Whether to ignore missing predictions
            true_labels: Set of DocItemLabel to use for ground truth visualization
            pred_labels: Set of DocItemLabel to use for prediction visualization
        """
        super().__init__(
            do_visualization=do_visualization,
            ignore_missing_predictions=ignore_missing_predictions,
            true_labels=true_labels,
            pred_labels=pred_labels,
        )
        self._use_ground_truth_page_images = use_ground_truth_page_images

        self._supported_prediction_formats = [
            PredictionFormats.DOCTAGS,
            PredictionFormats.MARKDOWN,
            PredictionFormats.JSON,
            PredictionFormats.YAML,
        ]

        # Validate the prediction format
        if prediction_format not in self._supported_prediction_formats:
            msg = f"Unsupported file prediction format: {prediction_format}."
            msg += f" The prediction format must be one of {self._supported_prediction_formats}"
            raise RuntimeError(msg)

        # Read the input
        self._prediction_format = prediction_format
        self._prediction_source_path = source_path
        self._ignore_missing_files = ignore_missing_files

        # Validate if the source_path exists
        if not self._prediction_source_path.is_dir():
            raise RuntimeError(f"Missing source path: {self._prediction_source_path}")

    def info(self) -> Dict:
        """Get information about the prediction provider."""
        return {
            "supported_prediction_formats": [
                fmt.value for fmt in self._supported_prediction_formats
            ],
            "prediction_format": self._prediction_format.value,
            "source_path": str(self._prediction_source_path),
        }

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Generate a prediction by reading from a file.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction
        """
        raw = None
        pred_doc = None

        # Load document based on prediction format
        if self._prediction_format == PredictionFormats.DOCTAGS:
            pred_doc = self._load_doctags_doc(record)
        elif self._prediction_format == PredictionFormats.MARKDOWN:
            raw = self._load_md_raw(record)
        elif self._prediction_format == PredictionFormats.JSON:
            pred_doc = self._load_json_doc(record)
        elif self._prediction_format == PredictionFormats.YAML:
            pred_doc = self._load_yaml_doc(record)

        # Set status based on whether document was loaded
        status = (
            ConversionStatus.SUCCESS
            if pred_doc is not None
            else ConversionStatus.FAILURE
        )

        # Create prediction record
        pred_record = self.create_dataset_record_with_prediction(
            record,
            pred_doc,
            raw,
        )
        pred_record.status = status
        return pred_record

    @property
    def prediction_format(self) -> PredictionFormats:
        """Get the prediction format."""
        return self._prediction_format

    def _load_doctags_doc(self, record: DatasetRecord) -> Optional[DoclingDocument]:
        """
        Load doctags file into DoclingDocument.

        Args:
            record: Groundtruth dataset record

        Returns:
            DoclingDocument or None if file not found
        """
        # Read the doctags file
        doctags_fn = self._prediction_source_path / f"{record.doc_id}.dt"
        if self._ignore_missing_files and not doctags_fn.is_file():
            return None

        try:
            with open(doctags_fn, "r") as fd:
                doctags = fd.read()

            page_image = None

            if self._use_ground_truth_page_images:
                page_image = record.ground_truth_page_images[0]
            else:
                # Check if an optional page image is present
                page_image_fn = self._prediction_source_path / f"{record.doc_id}.png"
                if page_image_fn.is_file():
                    page_image = Image.open(page_image_fn)

            # Build DoclingDocument
            doctags_page = DocTagsPage(tokens=doctags, image=page_image)
            doctags_doc = DocTagsDocument(pages=[doctags_page])
            doc = DoclingDocument(name=record.doc_id)
            doc.load_from_doctags(doctags_doc)

            return doc
        except Exception as e:
            _log.error(f"Error loading doctags document {record.doc_id}: {str(e)}")
            if not self._ignore_missing_files:
                raise
            return None

    def _load_json_doc(self, record: DatasetRecord) -> Optional[DoclingDocument]:
        """
        Load DoclingDocument from JSON.

        Args:
            record: Groundtruth dataset record

        Returns:
            DoclingDocument or None if file not found
        """
        json_fn = self._prediction_source_path / f"{record.doc_id}.json"
        if self._ignore_missing_files and not json_fn.is_file():
            return None

        try:
            doc: DoclingDocument = DoclingDocument.load_from_json(json_fn)
            return doc
        except Exception as e:
            _log.error(f"Error loading JSON document {record.doc_id}: {str(e)}")
            if not self._ignore_missing_files:
                raise
            return None

    def _load_yaml_doc(self, record: DatasetRecord) -> Optional[DoclingDocument]:
        """
        Load DoclingDocument from YAML.

        Args:
            record: Groundtruth dataset record

        Returns:
            DoclingDocument or None if file not found
        """
        # Try with .yaml extension
        yaml_fn = self._prediction_source_path / f"{record.doc_id}.yaml"

        # If not found, try with .yml extension
        if not yaml_fn.is_file():
            yaml_fn = self._prediction_source_path / f"{record.doc_id}.yml"

        if self._ignore_missing_files and not yaml_fn.is_file():
            return None

        try:
            doc: DoclingDocument = DoclingDocument.load_from_yaml(yaml_fn)
            return doc
        except Exception as e:
            _log.error(f"Error loading YAML document {record.doc_id}: {str(e)}")
            if not self._ignore_missing_files:
                raise
            return None

    def _load_md_raw(self, record: DatasetRecord) -> Optional[str]:
        """
        Load the markdown content.

        Args:
            record: Groundtruth dataset record

        Returns:
            Markdown content or None if file not found
        """
        md_fn = self._prediction_source_path / f"{record.doc_id}.md"
        if self._ignore_missing_files and not md_fn.is_file():
            return None

        try:
            with open(md_fn, "r") as fd:
                md = fd.read()
            return md
        except Exception as e:
            _log.error(f"Error loading markdown document {record.doc_id}: {str(e)}")
            if not self._ignore_missing_files:
                raise
            return None
