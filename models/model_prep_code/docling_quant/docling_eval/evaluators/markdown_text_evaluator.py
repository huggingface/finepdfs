import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import evaluate
import nltk
from datasets import load_dataset
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import ContentLayer, DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from nltk import edit_distance, word_tokenize
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import (  # type: ignore
    BenchMarkColumns,
    PredictionFormats,
)
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    UnitEvaluation,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats

_log = logging.getLogger(__name__)


class PageMarkdownEvaluation(UnitEvaluation):
    doc_id: str

    true_md: str
    pred_md: str
    bleu: float
    f1_score: float
    precision: float
    recall: float
    edit_distance: float
    meteor: float


class DatasetMarkdownEvaluation(DatasetEvaluation):
    evaluations: List[PageMarkdownEvaluation]

    bleu_stats: DatasetStatistics
    f1_score_stats: DatasetStatistics
    precision_stats: DatasetStatistics
    recall_stats: DatasetStatistics
    edit_distance_stats: DatasetStatistics
    meteor_stats: DatasetStatistics


class MarkdownTextEvaluator(BaseEvaluator):
    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [],
    ):
        r""" """
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
            PredictionFormats.MARKDOWN,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        self._bleu_eval = evaluate.load("bleu")

        # Download the NLTK data
        nltk.download("popular", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        # Select which DocItemLabels should be exported to markdown
        self._labels: Set[DocItemLabel] = set(
            [
                DocItemLabel.CAPTION,
                DocItemLabel.FOOTNOTE,
                DocItemLabel.FORMULA,
                DocItemLabel.LIST_ITEM,
                DocItemLabel.PAGE_FOOTER,
                DocItemLabel.PAGE_HEADER,
                DocItemLabel.PICTURE,
                DocItemLabel.SECTION_HEADER,
                # DocItemLabel.TABLE,
                DocItemLabel.TEXT,
                DocItemLabel.TITLE,
                DocItemLabel.DOCUMENT_INDEX,
                DocItemLabel.CODE,
                DocItemLabel.CHECKBOX_SELECTED,
                DocItemLabel.CHECKBOX_UNSELECTED,
                DocItemLabel.FORM,
                DocItemLabel.KEY_VALUE_REGION,
                DocItemLabel.PARAGRAPH,
                DocItemLabel.REFERENCE,
            ]
        )

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetMarkdownEvaluation:
        r"""
        Parameters
        ----------
        ds_path: Path to load the parquet files of the dataset
        split: Split of the dataset to load
        """
        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        _log.info(f"Overview of the dataset: {ds}")
        if ds is not None:
            ds_selection = ds[split]

        evaluations: list[PageMarkdownEvaluation] = []

        # Metrics per page
        ds_metrics: dict[str, list[float]] = {
            "bleu": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "edit_distance": [],
            "meteor": [],
        }

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Markdown text evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id
            if data_record.status not in self._accepted_status:
                _log.error(
                    "Skipping record without successfull conversion status: %s", doc_id
                )
                continue

            true_doc = data_record.ground_truth_doc
            true_md = self._docling_document_to_md(true_doc)
            pred_md = self._get_pred_md(data_record)

            if pred_md is None:
                _log.error("There is no markdown prediction for doc_id=%s", doc_id)
                continue

            bleu = 0.0
            if true_md != "" and pred_md != "":
                bleu = self._compute_bleu_score(true_md, pred_md)
                ntlk_scores = self._compute_nltk_scores(true_md, pred_md)

            # Collect metrics across pages
            ds_metrics["bleu"].append(bleu)
            for score_name, score in ntlk_scores.items():
                ds_metrics[score_name].append(score)

            md_evaluation = PageMarkdownEvaluation(
                doc_id=doc_id,
                true_md=true_md,
                pred_md=pred_md,
                bleu=bleu,
                f1_score=ntlk_scores["f1_score"],
                precision=ntlk_scores["precision"],
                recall=ntlk_scores["recall"],
                edit_distance=ntlk_scores["edit_distance"],
                meteor=ntlk_scores["meteor"],
            )
            evaluations.append(md_evaluation)

            if self._intermediate_evaluations_path:
                self.save_intermediate_evalutions("MD", i, doc_id, evaluations)

        ds_md_evalutions = DatasetMarkdownEvaluation(
            evaluations=evaluations,
            bleu_stats=compute_stats(ds_metrics["bleu"]),
            f1_score_stats=compute_stats(ds_metrics["f1_score"]),
            precision_stats=compute_stats(ds_metrics["precision"]),
            recall_stats=compute_stats(ds_metrics["recall"]),
            edit_distance_stats=compute_stats(ds_metrics["edit_distance"]),
            meteor_stats=compute_stats(ds_metrics["meteor"]),
        )
        return ds_md_evalutions

    def _compute_bleu_score(self, true_txt: str, pred_txt: str) -> float:
        r"""
        Compute BLEU score with the HF evaluate and the default Tokenizer_13
        """
        result = self._bleu_eval.compute(
            predictions=[pred_txt], references=[[true_txt]]
        )
        bleu = result["bleu"]
        return bleu

    def _compute_nltk_scores(self, true_txt: str, pred_txt: str) -> dict[str, float]:
        r"""
        Returns:
        --------
        dict with keys: ["f_measure", "precision", "recall", "edit_dist"]
        """
        true_tokens = word_tokenize(true_txt)
        true_tokens_set = set(true_tokens)
        pred_tokens = word_tokenize(pred_txt)
        pred_tokens_set = set(pred_tokens)

        f1_score = f_measure(true_tokens_set, pred_tokens_set)
        precision_score = precision(true_tokens_set, pred_tokens_set)
        recall_score = recall(true_tokens_set, pred_tokens_set)
        edit_dist = edit_distance(pred_tokens, true_tokens) / max(
            len(pred_tokens), len(true_tokens)
        )
        meteor = meteor_score.meteor_score([true_tokens], pred_tokens)

        metrics: dict[str, float] = {
            "f1_score": f1_score,
            "precision": precision_score,
            "recall": recall_score,
            "edit_distance": edit_dist,
            "meteor": meteor,
        }
        return metrics

    def _docling_document_to_md(self, doc: DoclingDocument) -> str:
        r"""
        Export DoclingDocument to markdown
        """
        md = doc.export_to_markdown(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="",
            labels=self._labels,
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
        )
        return md

    def _get_pred_md(self, data_record: DatasetRecordWithPrediction) -> Optional[str]:
        r"""
        Get the predicted markdown
        """
        pred_md = None
        for prediction_format in self._prediction_sources:
            if prediction_format == PredictionFormats.DOCLING_DOCUMENT:
                pred_doc = data_record.predicted_doc
                if pred_doc is not None:
                    pred_md = self._docling_document_to_md(pred_doc)
            elif prediction_format == PredictionFormats.MARKDOWN:
                pred_md = data_record.original_prediction
            if pred_md is not None:
                break

        return pred_md
