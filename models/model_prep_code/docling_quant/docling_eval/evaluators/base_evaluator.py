import json
import logging
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar

from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.document import (
    DoclingDocument,
    DocTagsDocument,
    DocTagsPage,
)
from pydantic import BaseModel

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import PredictionFormats

_log = logging.getLogger(__name__)


class UnitEvaluation(BaseModel):
    pass


class DatasetEvaluation(BaseModel):
    pass


UnitEvaluationType = TypeVar("UnitEvaluationType", bound=UnitEvaluation)
DatasetEvaluationType = TypeVar("DatasetEvaluationType", bound=DatasetEvaluation)


def docling_document_from_doctags(
    data_record: DatasetRecordWithPrediction,
) -> DoclingDocument:
    r""" """
    doc_id = data_record.doc_id
    doctags = data_record.original
    if not isinstance(doctags, str):
        raise RuntimeError("Invalid format of original prediction")

    page_image = (
        data_record.ground_truth_page_images[0]
        if data_record.ground_truth_page_images
        else None
    )

    doctags_page = DocTagsPage(tokens=doctags, image=page_image)
    doctags_doc = DocTagsDocument(pages=[doctags_page])
    pred_doc = DoclingDocument(name=doc_id)
    pred_doc.load_from_doctags(doctags_doc)

    return pred_doc


class BaseEvaluator(Generic[UnitEvaluationType, DatasetEvaluationType]):
    r"""
    Base class for all evaluators
    """

    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT
        ],
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT
        ],
    ):
        r"""
        Parameters
        ----------
        intermediate_evaluations_path: When True the evalution per example will be saved in a file
        """
        self._intermediate_evaluations_path = intermediate_evaluations_path

        # Validate the prediction_sources
        if set(prediction_sources) - set(supported_prediction_formats):
            msg = "Unsupported prediction_sources. "
            msg += f"It should be something out of {supported_prediction_formats}"
            raise RuntimeError(msg)
        self._prediction_sources = prediction_sources
        self._supported_prediction_sources = supported_prediction_formats

        self._accepted_status: List[ConversionStatus] = [
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
        ]

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetEvaluationType:
        r"""
        Perform the evaluation
        """
        return None  # type: ignore

    def supported_prediction_formats(self) -> List[PredictionFormats]:
        r"""
        Return the supported formats for predictions
        """
        return self._supported_prediction_sources

    def save_intermediate_evalutions(
        self,
        evaluation_name: str,
        enunumerate_id: int,
        doc_id: str,
        evaluations: List[UnitEvaluationType],
    ) -> Optional[Path]:
        r"""
        Utility method to save intermediate evaluation results
        Return immediatelly if the intermediate_evaluation_path is not set
        It returns the file Path with the intermediate results or None
        """
        if self._intermediate_evaluations_path:
            return None

        evals = [ev.model_dump() for ev in evaluations]
        evaluation_filename = f"{evaluation_name}_{enunumerate_id:05d}_{doc_id}.json"
        evaluation_fn = self._intermediate_evaluations_path / evaluation_filename  # type: ignore
        _log.info("Saving intermediate evaluations: %s", evaluation_fn)
        with open(evaluation_fn, "w") as fd:
            json.dump(evals, fd)

        return evaluation_fn
