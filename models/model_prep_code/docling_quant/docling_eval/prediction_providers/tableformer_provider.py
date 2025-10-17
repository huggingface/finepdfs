import copy
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from docling.datamodel.base_models import (
    Cluster,
    ConversionStatus,
    LayoutPrediction,
    Page,
    Table,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.models.table_structure_model import TableStructureModel
from docling_core.types import DoclingDocument
from docling_core.types.doc import DocItemLabel, TableCell, TableData, TableItem
from docling_core.types.io import DocumentStream
from PIL import Image

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import PageToken, PageTokens, PredictionFormats
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import (
    docling_models_version,
    get_input_document,
    insert_images_from_pil,
)

_log = logging.getLogger(__name__)


class TableFormerPredictionProvider(BasePredictionProvider):
    """
    Prediction provider that uses TableFormer for table structure prediction.

    This provider is specialized for predicting table structures in documents.
    """

    def __init__(
        self,
        mode: TableFormerMode = TableFormerMode.ACCURATE,
        num_threads: int = 16,
        artifacts_path: Optional[Path] = None,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
    ):
        """
        Initialize the TableFormer prediction provider.

        Args:
            mode: TableFormer prediction mode
            num_threads: Number of threads for prediction
            artifacts_path: Path to artifacts
            do_visualization: Whether to generate visualizations
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
        self.tf_updater = TableFormerUpdater(mode, num_threads, artifacts_path)

    @property
    def prediction_format(self) -> PredictionFormats:
        """Get the prediction format."""
        return PredictionFormats.DOCLING_DOCUMENT

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Generate a prediction for table structure.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction

        Raises:
            RuntimeError: If ground truth doc is not available or if mime type is unsupported
        """
        if record.ground_truth_doc is None:
            raise RuntimeError(
                "true_doc must be given for TableFormer prediction provider to work."
            )

        updated = False
        pred_doc = None

        try:
            if record.mime_type == "application/pdf":
                if not isinstance(record.original, DocumentStream):
                    raise RuntimeError(
                        "Original document must be a DocumentStream for PDF files"
                    )

                # Process PDF
                updated, pred_doc = self.tf_updater.replace_tabledata(
                    copy.deepcopy(record.original.stream), record.ground_truth_doc
                )

            elif record.mime_type == "image/png":
                # Process image
                updated, pred_doc = self.tf_updater.replace_tabledata_with_page_tokens(
                    record.ground_truth_doc,
                    record.ground_truth_page_images,
                )
            else:
                raise RuntimeError(
                    f"Unsupported mime type: {record.mime_type}. TableFormerPredictionProvider supports 'application/pdf' and 'image/png'"
                )

            pred_doc = insert_images_from_pil(
                pred_doc,
                record.ground_truth_pictures,
                record.ground_truth_page_images,
            )
            # Set status based on update success
            status = ConversionStatus.SUCCESS if updated else ConversionStatus.FAILURE

        except Exception as e:
            _log.error(f"Error in TableFormer prediction: {str(e)}")
            status = ConversionStatus.FAILURE
            if not self.ignore_missing_predictions:
                raise
            pred_doc = record.ground_truth_doc.model_copy(
                deep=True
            )  # Use copy of ground truth as fallback

        pred_record = self.create_dataset_record_with_prediction(record, pred_doc, None)
        pred_record.status = status
        return pred_record

    def info(self) -> Dict:
        """Get information about the prediction provider."""
        return {"asset": "TableFormer", "version": docling_models_version()}


class TableFormerUpdater:
    """
    Utility class for updating table data using TableFormer.

    This class handles the prediction of table structures using the TableFormer model.
    """

    def __init__(
        self,
        mode: TableFormerMode,
        num_threads: int = 16,
        artifacts_path: Optional[Path] = None,
    ):
        """
        Initialize the TableFormer updater.

        Args:
            mode: TableFormer prediction mode
            num_threads: Number of threads for prediction
            artifacts_path: Path to artifacts
        """
        # Initialize the TableFormer model
        table_structure_options = TableStructureOptions(mode=mode)
        accelerator_options = AcceleratorOptions(
            num_threads=num_threads, device=AcceleratorDevice.AUTO
        )
        self._docling_tf_model = TableStructureModel(
            enabled=True,
            artifacts_path=artifacts_path,
            options=table_structure_options,
            accelerator_options=accelerator_options,
        )
        _log.info(f"Initialized TableFormer in {mode} mode")

    def to_np(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to NumPy array in BGR format.

        Args:
            pil_image: PIL image

        Returns:
            NumPy array in BGR format

        Raises:
            ValueError: If image format is unsupported
        """
        # Convert to NumPy array
        np_image = np.array(pil_image)

        # Handle different formats
        if np_image.ndim == 3:  # RGB or RGBA image
            if np_image.shape[2] == 4:  # RGBA image
                # Discard alpha channel
                np_image = np_image[:, :, :3]

            # Convert RGB to BGR by reversing the last axis
            np_image = np_image[:, :, ::-1]
            return np_image
        else:
            raise ValueError("Unsupported image format")

    def _make_internal_page_with_table(self, input_doc, prov) -> Page:
        """
        Create a page object with a table from input document.

        Args:
            input_doc: Input document
            prov: Provenance item

        Returns:
            Page object with table
        """
        page = Page(page_no=prov.page_no - 1)
        page._backend = input_doc._backend.load_page(page.page_no)

        # Add null checks to avoid mypy errors
        if page._backend is not None and page._backend.is_valid():
            page.cells = list(page._backend.get_text_cells())
            page.size = page._backend.get_size()

            # Create cluster for table
            cluster = Cluster(
                id=0,
                label=DocItemLabel.TABLE,
                bbox=prov.bbox.to_top_left_origin(page.size.height),
            )

            # Add cells that overlap with the cluster
            for cell in page.cells:
                overlap = cell.rect.to_bounding_box().intersection_area_with(
                    cluster.bbox
                )
                if cell.rect.to_bounding_box().area() > 0:
                    overlap_ratio = overlap / cell.rect.to_bounding_box().area()
                    if overlap_ratio > 0.2:
                        cluster.cells.append(cell)

            page.predictions.layout = LayoutPrediction(clusters=[cluster])

        return page

    def replace_tabledata(
        self,
        pdf_path: Union[Path, BytesIO],
        true_doc: DoclingDocument,
    ) -> Tuple[bool, DoclingDocument]:
        """
        Replace table data in document with predictions from TableFormer.

        Args:
            pdf_path: Path to PDF file or PDF data as BytesIO
            true_doc: Document with ground truth tables

        Returns:
            Tuple of (success, updated_document)
        """
        # Make a deep copy of the document
        pred_doc = true_doc.model_copy(deep=True)

        # Parse the PDF
        input_doc = get_input_document(pdf_path)
        if not input_doc.valid:
            _log.error("Could not parse PDF file")
            return False, pred_doc

        conv_res = ConversionResult(input=input_doc)
        updated = False

        # Process each table item in the document
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:
                    try:
                        # Create page with table
                        page = self._make_internal_page_with_table(input_doc, prov)

                        # Fix mypy error with next() by converting iterator to list
                        model_results = list(self._docling_tf_model(conv_res, [page]))

                        if model_results and hasattr(
                            model_results[0].predictions, "tablestructure"
                        ):
                            page = model_results[0]
                            if (
                                page.predictions.tablestructure is not None
                                and hasattr(
                                    page.predictions.tablestructure, "table_map"
                                )
                                and page.predictions.tablestructure.table_map
                            ):
                                tbl: Table = page.predictions.tablestructure.table_map[
                                    0
                                ]
                                table_data: TableData = TableData(
                                    num_rows=tbl.num_rows,
                                    num_cols=tbl.num_cols,
                                    table_cells=tbl.table_cells,
                                )

                                # Update item data
                                item.data = table_data
                                updated = True
                    except Exception as e:
                        raise
                    finally:
                        # Ensure page backend is unloaded to free resources
                        if hasattr(page, "_backend") and page._backend is not None:
                            page._backend.unload()

        return updated, pred_doc

    def _tf_predict_with_page_tokens(
        self,
        page_image: Image.Image,
        page_tokens: PageTokens,
        table_bbox: Tuple[float, float, float, float],
        image_scale: float = 1.0,
    ) -> TableData:
        """
        Predict table structure from image using page tokens.

        Args:
            page_image: Page image
            page_tokens: Page tokens
            table_bbox: Table bounding box coordinates (l, t, r, b)
            image_scale: Image scale factor

        Returns:
            Predicted table data
        """
        # Prepare input for TableFormer
        table_bboxes = [[table_bbox[0], table_bbox[1], table_bbox[2], table_bbox[3]]]
        ocr_page = page_tokens.dict()
        ocr_page["image"] = self.to_np(page_image)
        ocr_page["table_bboxes"] = table_bboxes

        # Get predictor from model
        predictor = self._docling_tf_model.tf_predictor

        # Run prediction
        tf_output = predictor.multi_table_predict(
            ocr_page,
            table_bboxes=table_bboxes,
            do_matching=True,
            correct_overlapping_cells=False,
            sort_row_col_indexes=True,
        )

        # Extract table data
        table_out = tf_output[0]
        table_cells = []

        # Process each cell
        for element in table_out["tf_responses"]:
            tc = TableCell.model_validate(element)
            if tc.bbox is not None:
                tc.bbox = tc.bbox.scaled(1 / image_scale)
            table_cells.append(tc)

        # Get table dimensions
        num_rows = table_out["predict_details"]["num_rows"]
        num_cols = table_out["predict_details"]["num_cols"]

        # Create table data
        table_data = TableData(
            num_rows=num_rows, num_cols=num_cols, table_cells=table_cells
        )

        return table_data

    def replace_tabledata_with_page_tokens(
        self,
        true_doc: DoclingDocument,
        true_page_images: List[Image.Image],
        page_tokens: Optional[PageTokens] = None,
    ) -> Tuple[bool, DoclingDocument]:
        """
        Replace table data in document using page tokens and images.

        Args:
            true_doc: Document with ground truth tables
            true_page_images: Page images
            page_tokens: Optional page tokens

        Returns:
            Tuple of (success, updated_document)
        """
        # Make a deep copy of the document
        pred_doc = copy.deepcopy(true_doc)
        updated = False

        # Ensure document has exactly one page
        if len(pred_doc.pages) != 1:
            _log.error("Document must have exactly one page")
            return False, pred_doc

        page_size = pred_doc.pages[1].size

        # Process each table item
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:
                    try:
                        # Get page image
                        page_image = true_page_images[prov.page_no - 1]

                        # Ensure bounding box is within page bounds
                        table_bbox = (
                            max(prov.bbox.l, 0.0),
                            max(prov.bbox.b, 0.0),
                            min(prov.bbox.r, page_size.width),
                            min(prov.bbox.t, page_size.height),
                        )

                        # Create page tokens if not provided
                        if page_tokens is None:
                            ptokens = []
                            for ix, table_cell in enumerate(item.data.table_cells):
                                pt = PageToken(
                                    bbox=table_cell.bbox, text=table_cell.text, id=ix
                                )
                                ptokens.append(pt)
                            page_tokens = PageTokens(
                                tokens=ptokens,
                                height=prov.bbox.height,
                                width=prov.bbox.width,
                            )

                        # Predict table data
                        table_data = self._tf_predict_with_page_tokens(
                            page_image=page_image,
                            page_tokens=page_tokens,
                            table_bbox=table_bbox,
                        )

                        # Update item data
                        item.data = table_data
                        updated = True
                    except Exception as e:
                        _log.error(f"Error predicting table: {str(e)}")
                        raise

        return updated, pred_doc
