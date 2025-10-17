import logging
from pathlib import Path
from typing import Dict, List, Optional

import nltk
from datasets import load_dataset
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import DoclingDocument, TextItem
from nltk import edit_distance, word_tokenize
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score
from nltk.translate.bleu_score import corpus_bleu
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


class BoxesTextEvaluation(UnitEvaluation):
    r"""Evaluation for matched bboxes of the same page"""

    doc_id: str

    # Ideally it is an 1-1 matching between a true/pred bbox. In practice it is many to many
    true_bboxes: Optional[List[BoundingBox]]
    pred_bboxes: Optional[List[BoundingBox]]
    true_tokens: Optional[List[str]]
    pred_tokens: Optional[List[str]]

    bleu: float
    f1_score: float
    precision: float
    recall: float
    edit_distance: float
    meteor: float


class DatasetBoxesTextEvaluation(DatasetEvaluation):
    evaluations: List[BoxesTextEvaluation]
    bleu_stats: DatasetStatistics
    f1_score_stats: DatasetStatistics
    precision_stats: DatasetStatistics
    recall_stats: DatasetStatistics
    edit_distance_stats: DatasetStatistics
    meteor_stats: DatasetStatistics


class BboxTextEvaluator(BaseEvaluator):
    r"""
    1. Starting from a true DoclingDocument and a pred DoclingDocument.
    2. Take as a pivot the document with less bboxes.
    3. For each bbox of the pivot find the corresponding bboxes of the other document.
       a. Each bbox of the other is matched to the bbox of the pivot with the max IoU.
       b. There can be also unmatched bboxes from both documents.
       c. Collect the orphan true bboxes (without matches to the pred bboxes).
    4. Tokenize the text from the pivot bbox and the joint texts from the matched "other" bboxes.
       - This creates an 1-1 mapping of the "pivot" tokens and the "other" tokens for the corresponding bboxes.
    5. The mapping of the pivot/other tokens can be also seen as a mapping between the true/pred tokens.
    6. Add to the mapping from true/pred tokens from 5 the orphan tokens from 3c as true/empty.
    7. Compute text metrics (BLEU, f1, recall, precision, meteor, edit_dist) for the pivot/other tokens.
    """

    def __init_(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [],
    ):
        r""" """
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        # Download the NLTK data
        nltk.download("popular", quiet=True)

    def __call__(
        self, ds_path: Path, split: str = "test"
    ) -> DatasetBoxesTextEvaluation:
        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        _log.info(f"oveview of dataset: {ds}")
        if ds is not None:
            ds_selection = ds[split]

        # Metrics per page
        ds_metrics: Dict[str, List[float]] = {
            "bleu": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "edit_distance": [],
            "meteor": [],
        }
        boxes_evaluations: List[BoxesTextEvaluation] = []

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Matched bboxes text evaluations",
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
            pred_doc = data_record.predicted_doc
            if pred_doc is None:
                _log.error("There is no prediction for doc_id=%s", doc_id)
                continue

            # Match the bboxes/text from the true/pred documents
            matches = self._match_bboxes(true_doc, pred_doc)

            for match in matches:
                true_tokens = match["true_tokens"]
                pred_tokens = match["pred_tokens"]
                scores = self._compute_scores(true_tokens, pred_tokens)

                for score_name, score in scores.items():
                    ds_metrics[score_name].append(score)

                boxes_evaluation = BoxesTextEvaluation(
                    doc_id=doc_id,
                    true_bboxes=match["true_bboxes"],
                    pred_bboxes=match["pred_bboxes"],
                    true_tokens=match["true_tokens"],
                    pred_tokens=match["pred_tokens"],
                    bleu=scores["bleu"],
                    f1_score=scores["f1_score"],
                    precision=scores["precision"],
                    recall=scores["recall"],
                    edit_distance=scores["edit_distance"],
                    meteor=scores["meteor"],
                )
                boxes_evaluations.append(boxes_evaluation)

        ds_evaluation = DatasetBoxesTextEvaluation(
            evaluations=boxes_evaluations,
            bleu_stats=compute_stats(ds_metrics["bleu"]),
            f1_score_stats=compute_stats(ds_metrics["f1_score"]),
            precision_stats=compute_stats(ds_metrics["precision"]),
            recall_stats=compute_stats(ds_metrics["recall"]),
            edit_distance_stats=compute_stats(ds_metrics["edit_distance"]),
            meteor_stats=compute_stats(ds_metrics["meteor"]),
        )
        return ds_evaluation

    def _match_bboxes(
        self,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        pivot: Optional[str] = None,
    ) -> List[Dict[str, List]]:
        r"""
        Parameters
        ----------
        pivot: It must be either None or one of ["true", "pred"].
               If it is None the pivot is the document with less bboxes.

        Returns
        --------
        List with matchings between for the bboxes/text between the true and pred docs.
        - Each list item is a dict with the matching between the true and the pred bboxes.
        - Each dict has the keys:
          "true_bboxes": List[BoundingBox] of true bboxes
          "true_tokens": List[str] of the tokenized text contained in the true_bboxes
          "pred_bboxes": List[BoundingBox] of pred bboxes
          "pred_tokens": List[str] of the tokenized text contained in the pred_bboxes
        """
        if pivot is not None:
            assert pivot in ["true", "pred"]

        # Collect bboxes from both documents (true, pred)
        bboxes: Dict[str, List[BoundingBox]] = {"true": [], "pred": []}
        texts: Dict[str, List[str]] = {"true": [], "pred": []}
        for doc_key, doc in {"true": true_doc, "pred": pred_doc}.items():
            for doc_item, _ in doc.iterate_items():
                if not isinstance(doc_item, TextItem):
                    continue
                assert len(doc_item.prov) == 1
                prov = doc_item.prov[0]
                bboxes[doc_key].append(prov.bbox)
                texts[doc_key].append(doc_item.text)

        # Decide which document is the pivot
        if pivot is None:
            pivot = "true" if len(bboxes["true"]) <= len(bboxes["pred"]) else "pred"
        other = "pred" if pivot == "true" else "true"

        # Map the "pivot" bboxes to the "other" bboxes
        # Keys: the indices from bboxes[pivot]. Each value: list with indices from bboxes[other]
        pivot_mappings: Dict[int, List[int]] = {}
        all_other_ids = set()
        for other_id, other_bbox in enumerate(bboxes[other]):
            max_iou = None
            max_pivot_id = None
            for pivot_id, pivot_bbox in enumerate(bboxes[pivot]):
                iou = other_bbox.intersection_over_union(pivot_bbox)
                if max_iou is None or max_iou < iou:
                    max_iou = iou
                    max_pivot_id = pivot_id
            if max_iou is not None and max_pivot_id is not None:
                if max_pivot_id not in pivot_mappings:
                    pivot_mappings[max_pivot_id] = []
                pivot_mappings[max_pivot_id].append(other_id)
                all_other_ids.add(other_id)

        # Collect the unmatched true bboxes
        orphan_trues: List[int] = []
        for true_id in range(len(bboxes["true"])):
            if pivot == "true":
                if true_id not in pivot_mappings:
                    orphan_trues.append(true_id)
            else:
                if true_id not in all_other_ids:
                    orphan_trues.append(true_id)

        # Create mapping for the text of the matched bboxes
        # Each dict has the keys:
        #  "true_bboxes": List[BoundingBox] of true bboxes
        #  "true_tokens": List[str] of the tokenized text contained in the true_bboxes
        #  "pred_bboxes": List[BoundingBox] of pred bboxes
        #  "pred_tokens": List[str] of the tokenized text contained in the pred_bboxes
        matches: list[Dict[str, list]] = []
        for pivot_id, list_other_ids in pivot_mappings.items():
            pivot_bboxes = [bboxes[pivot][pivot_id]]
            pivot_text = texts[pivot][pivot_id]
            pivot_tokens = word_tokenize(pivot_text)
            other_tokens = []
            other_bboxes = []
            for other_id in list_other_ids:
                other_text = texts[other][other_id]
                other_tokens.extend(word_tokenize(other_text))
                other_bboxes.append(bboxes[other][other_id])

            matches.append(
                {
                    f"{pivot}_bboxes": pivot_bboxes,
                    f"{pivot}_tokens": pivot_tokens,
                    f"{other}_bboxes": other_bboxes,
                    f"{other}_tokens": other_tokens,
                }
            )

        # Add the orphans_true inside the matches
        for orphan_true_id in orphan_trues:
            orphan_bboxes = [bboxes["true"][orphan_true_id]]
            orphan_text = texts["true"][orphan_true_id]
            orphan_tokens = word_tokenize(orphan_text)
            matches.append(
                {
                    "true_bboxes": orphan_bboxes,
                    "true_tokens": orphan_tokens,
                    "pred_bboxes": [],
                    "pred_tokens": [],
                }
            )

        return matches

    def _compute_scores(
        self, true_tokens: list[str], pred_tokens: list[str]
    ) -> Dict[str, float]:
        r"""
        Return
        ------
        dict with keys: ["bleu", "f_measure", "precision", "recall", "edit_dist"]
        """
        true_tokens_set = set(true_tokens)
        pred_tokens_set = set(pred_tokens)

        bleu = corpus_bleu(
            [[true_tokens]], [pred_tokens], weights=[0.25, 0.25, 0.25, 0.25]
        )
        f1_score = f_measure(true_tokens_set, pred_tokens_set)
        precision_score = precision(true_tokens_set, pred_tokens_set)
        recall_score = recall(true_tokens_set, pred_tokens_set)
        edit_dist = edit_distance(pred_tokens, true_tokens) / max(
            len(pred_tokens), len(true_tokens)
        )
        meteor = meteor_score.meteor_score([true_tokens], pred_tokens)

        # Make the metrics 0 if they are None
        metrics: Dict[str, float] = {
            "bleu": bleu,
            "f1_score": f1_score,
            "precision": precision_score,
            "recall": recall_score,
            "edit_distance": edit_dist,
            "meteor": meteor,
        }
        metrics = {k: 0.0 if v is None else float(v) for k, v in metrics.items()}

        return metrics
