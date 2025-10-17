import importlib.metadata
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from docling.datamodel.base_models import ConversionStatus

# from docling_core.types import DoclingDocument
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.io import DocumentStream

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import PredictionFormats
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import from_pil_to_base64uri

# from docling_core.types.doc.labels import DocItemLabel

_log = logging.getLogger(__name__)


class AzureDocIntelligencePredictionProvider(BasePredictionProvider):
    """Provider that calls the Microsoft Azure Document Intelligence API for predicting the tables in document."""

    def __init__(
        self,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(
            do_visualization=do_visualization,
            ignore_missing_predictions=ignore_missing_predictions,
            true_labels=true_labels,
            pred_labels=pred_labels,
        )

        # TODO - Need a temp directory to save Azure outputs
        # Validate the required library
        try:
            from azure.ai.formrecognizer import (  # type: ignore
                AnalysisFeature,
                DocumentAnalysisClient,
            )
            from azure.core.credentials import AzureKeyCredential  # type: ignore
        except ImportError:
            raise ImportError("azure-ai-formrecognizer library is not installed..")

        # Validate the required endpoints to call the API
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not endpoint or not key:
            raise ValueError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY must be set in environment variables."
            )

        self.doc_intelligence_client = DocumentAnalysisClient(
            endpoint, AzureKeyCredential(key)
        )

    def extract_bbox_from_polygon(self, polygon):
        """Helper function to extract bbox coordinates from polygon data."""
        if not polygon or not isinstance(polygon, list):
            return {"l": 0, "t": 0, "r": 0, "b": 0}

        # Handle flat array format: [x1, y1, x2, y2, x3, y3, x4, y4]
        if len(polygon) >= 8 and all(isinstance(p, (int, float)) for p in polygon):
            return {"l": polygon[0], "t": polygon[1], "r": polygon[4], "b": polygon[5]}
        # Handle array of point objects: [{x, y}, {x, y}, ...]
        elif len(polygon) >= 4 and all(
            isinstance(p, dict) and "x" in p and "y" in p for p in polygon
        ):
            return {
                "l": polygon[0]["x"],
                "t": polygon[0]["y"],
                "r": polygon[2]["x"],
                "b": polygon[2]["y"],
            }
        else:
            return {"l": 0, "t": 0, "r": 0, "b": 0}

    def convert_azure_output_to_docling(
        self, analyze_result, record: DatasetRecord
    ) -> DoclingDocument:
        """Converts Azure Document Intelligence output to DoclingDocument format."""
        doc = DoclingDocument(name=record.doc_id)

        for page in analyze_result.get("pages", []):
            page_no = page.get("page_number", 1)

            page_width = page.get("width")
            page_height = page.get("height")

            im = record.ground_truth_page_images[page_no - 1]

            # Add page with image
            image_ref = ImageRef(
                mimetype=f"image/png",
                dpi=72,
                size=Size(width=float(im.width), height=float(im.height)),
                uri=from_pil_to_base64uri(im),
            )
            page_item = PageItem(
                page_no=page_no,
                size=Size(width=page_width, height=page_height),
                image=image_ref,
            )
            doc.pages[page_no] = page_item

            for word in page.get("words", []):
                polygon = word.get("polygon", [])
                bbox = self.extract_bbox_from_polygon(polygon)

                text_content = word.get("content", "")

                bbox_obj = BoundingBox(
                    l=bbox["l"],
                    t=bbox["t"],
                    r=bbox["r"],
                    b=bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                prov = ProvenanceItem(
                    page_no=page_no, bbox=bbox_obj, charspan=(0, len(text_content))
                )

                # TODO: This needs to be developed further. Azure responses contain full-page document information,
                #       with text and layout features,
                #       see https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout
                #       This code only adds the primitive text content, without
                #       layout labels or reading order, then all tables separately. This will work for plain
                #       table datasets only.
                doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

        for table in analyze_result.get("tables", []):
            page_no = table.get("page_range", {}).get("first_page_number", 1)
            row_count = table.get("row_count", 0)
            col_count = table.get("column_count", 0)

            table_polygon = table.get("bounding_regions", [{}])[0].get("polygon", [])
            table_bbox = self.extract_bbox_from_polygon(table_polygon)

            table_bbox_obj = BoundingBox(
                l=table_bbox["l"],
                t=table_bbox["t"],
                r=table_bbox["r"],
                b=table_bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            table_prov = ProvenanceItem(
                page_no=page_no, bbox=table_bbox_obj, charspan=(0, 0)
            )

            table_cells = []

            for cell in table.get("cells", []):

                cell_text = cell.get("content", "").strip()
                row_index = cell.get("row_index", 0)
                col_index = cell.get("column_index", 0)
                row_span = cell.get("row_span", 1)
                col_span = cell.get("column_span", 1)

                cell_polygon = cell.get("bounding_regions", [{}])[0].get("polygon", [])
                cell_bbox = self.extract_bbox_from_polygon(cell_polygon)

                table_cell = TableCell(
                    bbox=BoundingBox(
                        l=cell_bbox["l"],
                        t=cell_bbox["t"],
                        r=cell_bbox["r"],
                        b=cell_bbox["b"],
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + row_span,
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + col_span,
                    text=cell_text,
                    column_header=False,
                    row_header=False,
                    row_section=False,
                )

                table_cells.append(table_cell)

            table_data = TableData(
                table_cells=table_cells, num_rows=row_count, num_cols=col_count
            )

            doc.add_table(prov=table_prov, data=table_data, caption=None)

        return doc

    @property
    def prediction_format(self) -> PredictionFormats:
        """Get the prediction format."""
        return PredictionFormats.JSON

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Generate a prediction from Azure Doc Intelligence.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction

        Raises:
            RuntimeError: If ground truth doc is not available or if mime type is unsupported
        """

        status = ConversionStatus.SUCCESS
        result_orig = None
        assert record.original is not None

        try:
            _log.info(f"Processing file '{record.doc_id}'..")
            if record.mime_type == "application/pdf":
                if not isinstance(record.original, DocumentStream):
                    raise RuntimeError(
                        "Original document must be a DocumentStream for PDF files"
                    )
                # Call the Azure API by passing in the image for prediction
                poller = self.doc_intelligence_client.begin_analyze_document(
                    "prebuilt-layout", record.original.stream, features=[]
                )
                result = poller.result()
                result_json = result.to_dict()
                _log.info(
                    f"Successfully processed [{record.doc_id}] using Azure API..!!"
                )

            elif record.mime_type == "image/png":
                # Call the Azure API by passing in the image for prediction
                buf = BytesIO()

                # TODO do this in a loop for all page images in the doc, not just the first.
                record.ground_truth_page_images[0].save(buf, format="PNG")

                poller = self.doc_intelligence_client.begin_analyze_document(
                    "prebuilt-layout", BytesIO(buf.getvalue()), features=[]
                )
                result = poller.result()
                result_json = result.to_dict()
                _log.info(
                    f"Successfully processed [{record.doc_id}] using Azure API..!!"
                )
            else:
                raise RuntimeError(
                    f"Unsupported mime type: {record.mime_type}. AzureDocIntelligencePredictionProvider supports 'application/pdf' and 'image/png'"
                )
            # Convert the prediction to doclingDocument
            pred_doc = self.convert_azure_output_to_docling(result_json, record)
            result_orig = json.dumps(result_json)

        except Exception as e:
            _log.error(
                f"Error in TableFormer prediction for '{record.doc_id}': {str(e)}"
            )
            status = ConversionStatus.FAILURE
            if not self.ignore_missing_predictions:
                raise
            pred_doc = record.ground_truth_doc.model_copy(
                deep=True
            )  # Use copy of ground truth as fallback

        pred_record = self.create_dataset_record_with_prediction(
            record, pred_doc, result_orig
        )
        pred_record.status = status
        return pred_record

    def info(self) -> Dict:
        return {
            "asset": "Azure AI Document Intelligence",
            "version": importlib.metadata.version("azure-ai-formrecognizer"),
        }
