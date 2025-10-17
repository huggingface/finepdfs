import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from docling_core.types.doc.document import (
    DEFAULT_EXPORT_LABELS,
    DocItem,
    DoclingDocument,
)
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    UnitEvaluation,
    docling_document_from_doctags,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats

_log = logging.getLogger(__name__)


class ClassLayoutEvaluation(BaseModel):
    r"""
    Class based layout evaluation
    """

    name: str
    label: str
    value: float  # mAP[0.5:0.05:0.95]


class ImageLayoutEvaluation(UnitEvaluation):
    r"""
    Image based layout evaluation
    """

    name: str
    value: float  # Area weighted average IoU for label-matched GT/pred bboxes for IoU thres = 0.5

    map_val: float  # mAP[0.5:0.05:0.95]
    map_50: float  # AP at IoU thres=0.5
    map_75: float  # AP at IoU thres=0.75

    # Weighted average IoU for the page bboxes with matching labels (between GT and pred)
    # The weight is the bbox size and each measurement corresponds to a different IoU threshold
    avg_weighted_label_matched_iou_50: float
    avg_weighted_label_matched_iou_75: float
    avg_weighted_label_matched_iou_90: float
    avg_weighted_label_matched_iou_95: float


class DatasetLayoutEvaluation(DatasetEvaluation):
    true_labels: Dict[str, int]
    pred_labels: Dict[str, int]
    mAP: float  # The mean AP[0.5:0.05:0.95] across all classes

    intersecting_labels: List[str]
    evaluations_per_class: List[ClassLayoutEvaluation]
    evaluations_per_image: List[ImageLayoutEvaluation]

    image_mAP_stats: (
        DatasetStatistics  # Stats for the mAP[0.5:0.05:0.95] across all images
    )

    def to_table(self) -> Tuple[List[List[str]], List[str]]:

        headers = ["label", "Class mAP[0.5:0.95]"]

        self.evaluations_per_class = sorted(
            self.evaluations_per_class, key=lambda x: x.value, reverse=True
        )

        table = []
        for i in range(len(self.evaluations_per_class)):
            table.append(
                [
                    f"{self.evaluations_per_class[i].label}",
                    f"{100.0*self.evaluations_per_class[i].value:.2f}",
                ]
            )

        return table, headers


class LayoutEvaluator(BaseEvaluator):
    def __init__(
        self,
        label_mapping: Optional[Dict[DocItemLabel, Optional[DocItemLabel]]] = None,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [],
    ):
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
            PredictionFormats.DOCTAGS,
            PredictionFormats.JSON,
            PredictionFormats.YAML,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        self.filter_labels = []
        self.label_names = {}
        self.label_mapping = label_mapping or {v: v for v in DocItemLabel}

        for i, _ in enumerate(DEFAULT_EXPORT_LABELS):
            self.filter_labels.append(_)
            self.label_names[i] = _
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

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ) -> DatasetLayoutEvaluation:
        logging.info("Loading the split '%s' from: '%s'", split, ds_path)

        # Load the dataset
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        logging.info("Files: %s", split_files)
        ds = load_dataset("parquet", data_files={split: split_files})
        logging.info("Overview of dataset: %s", ds)

        ds_selection: Dataset = ds[split]
        total_items = len(ds_selection)
        begin_index = 0
        end_index = -1
        begin, end = self.get_effective_indices(total_items, begin_index, end_index)

        # Select the split
        if begin > 0 or end < total_items:
            ds_selection = ds_selection.select(range(begin, end))

        # Select the range of documents to evaluate

        true_labels, pred_labels, intersection_labels = self._find_intersecting_labels(
            ds_selection
        )
        intersection_labels_str = "\n" + "\n".join(sorted(intersection_labels))
        logging.info(f"Intersection labels: {intersection_labels_str}")

        doc_ids = []
        ground_truths = []
        predictions = []
        mismatched_docs = 0

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Layout evaluations",
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
            pred_doc = self._get_pred_doc(data_record)
            if not pred_doc:
                _log.error("There is no prediction for doc_id=%s", doc_id)
                continue

            gts, preds = self._extract_layout_data(
                true_doc=true_doc,
                pred_doc=pred_doc,
                filter_labels=intersection_labels,
            )

            if len(gts) > 0:
                for i in range(len(gts)):
                    doc_ids.append(data[BenchMarkColumns.DOC_ID] + f"-page-{i}")

                ground_truths.extend(gts)

                if len(gts) == len(preds):
                    predictions.extend(preds)
                else:
                    mismatched_docs += 1
                    logging.error(
                        "Mismatch in len of GT (%s) vs pred (%s) in document_id '%s'.",
                        len(gts),
                        len(preds),
                        doc_id,
                    )

                    predictions.append(
                        {
                            "boxes": torch.empty(0, 4),
                            "labels": torch.empty(0),
                            "scores": torch.empty(0),
                        }
                    )

        if mismatched_docs > 0:
            logging.error(
                "Total mismatched/skipped documents: %s over %s",
                mismatched_docs,
                len(ds_selection),
            )

        assert len(doc_ids) == len(ground_truths), "doc_ids==len(ground_truths)"
        assert len(doc_ids) == len(predictions), "doc_ids==len(predictions)"

        # Initialize metric for the bboxes of the entire document
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

        # Update metric with predictions and ground truths
        metric.update(predictions, ground_truths)

        # Compute mAP and other metrics per class
        result = metric.compute()

        evaluations_per_class: List[ClassLayoutEvaluation] = []

        total_mAP = result["map"]
        if "map_per_class" in result:
            for label_idx, class_map in enumerate(result["map_per_class"]):
                evaluations_per_class.append(
                    ClassLayoutEvaluation(
                        name="Class mAP[0.5:0.95]",
                        label=intersection_labels[label_idx].value,
                        value=class_map,
                    )
                )

        # Compute mAP for each image individually
        map_values = []

        evaluations_per_image: List[ImageLayoutEvaluation] = []
        for i, (doc_id, pred, gt) in enumerate(
            zip(doc_ids, predictions, ground_truths)
        ):
            # Reset the metric for the next image
            # metric.reset()
            metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

            # Update with single image
            metric.update([pred], [gt])

            # Compute metrics
            result = metric.compute()

            # Extract mAP for this image
            map_value = float(result["map"].item())
            map_50 = float(result["map_50"].item())
            map_75 = float(result["map_75"].item())

            result = self._compute_average_iou_with_labels_across_iou(
                pred_boxes=pred["boxes"],
                pred_labels=pred["labels"],
                gt_boxes=gt["boxes"],
                gt_labels=gt["labels"],
            )
            average_iou_50 = result["average_iou_50"]
            average_iou_75 = result["average_iou_75"]
            average_iou_90 = result["average_iou_90"]
            average_iou_95 = result["average_iou_95"]

            map_values.append(map_value)
            image_evaluation = ImageLayoutEvaluation(
                name=doc_id,
                value=average_iou_50,
                map_val=map_value,
                map_50=map_50,
                map_75=map_75,
                avg_weighted_label_matched_iou_50=average_iou_50,
                avg_weighted_label_matched_iou_75=average_iou_75,
                avg_weighted_label_matched_iou_90=average_iou_90,
                avg_weighted_label_matched_iou_95=average_iou_95,
            )
            evaluations_per_image.append(image_evaluation)
            if self._intermediate_evaluations_path:
                self.save_intermediate_evalutions(
                    "Layout_image", i, doc_id, evaluations_per_image
                )

        evaluations_per_class = sorted(evaluations_per_class, key=lambda x: -x.value)
        evaluations_per_image = sorted(evaluations_per_image, key=lambda x: -x.value)

        return DatasetLayoutEvaluation(
            mAP=total_mAP,
            evaluations_per_class=evaluations_per_class,
            evaluations_per_image=evaluations_per_image,
            image_mAP_stats=compute_stats(map_values),
            true_labels=true_labels,
            pred_labels=pred_labels,
            intersecting_labels=[_.value for _ in intersection_labels],
        )

    def _get_pred_doc(
        self, data_record: DatasetRecordWithPrediction
    ) -> Optional[DoclingDocument]:
        r"""
        Get the predicted DoclingDocument
        """
        pred_doc = None
        for prediction_format in self._prediction_sources:
            if prediction_format == PredictionFormats.DOCLING_DOCUMENT:
                pred_doc = data_record.predicted_doc
            elif prediction_format == PredictionFormats.JSON:
                if data_record.original_prediction:
                    pred_doc = DoclingDocument.load_from_json(
                        data_record.original_prediction
                    )
            elif prediction_format == PredictionFormats.YAML:
                if data_record.original_prediction:
                    pred_doc = DoclingDocument.load_from_yaml(
                        data_record.original_prediction
                    )
            elif prediction_format == PredictionFormats.DOCTAGS:
                pred_doc = docling_document_from_doctags(data_record)
            if pred_doc is not None:
                break

        return pred_doc

    def _compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes."""
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        intersection = torch.max(x2 - x1, torch.tensor(0.0)) * torch.max(
            y2 - y1, torch.tensor(0.0)
        )
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def _compute_average_iou_with_labels(
        self, pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5
    ):
        """
        Compute the average IoU for label-matched detections and weight by bbox area:

        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes (N x 4).
            pred_labels (torch.Tensor): Labels for predicted boxes (N).
            gt_boxes (torch.Tensor): Ground truth bounding boxes (M x 4).
            gt_labels (torch.Tensor): Labels for ground truth boxes (M).
            iou_thresh (float): IoU threshold for a match.

        Returns:
            dict: Average IoU and unmatched ground truth information.
        """
        matched_gt = set()
        ious = []
        weights = []
        weights_sum = 0.0

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            weight = abs((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]))

            weights.append(weight)
            weights_sum += weight

            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if i not in matched_gt and pred_label == gt_label:
                    iou = self._compute_iou(pred_box, gt_box)
                    if iou >= iou_thresh:
                        matched_gt.add(i)
                        ious.append(iou.item())
                        break

        avg_iou = 0.0
        for w, v in zip(weights, ious):
            avg_iou += w * v / weights_sum

        unmatched_gt = len(gt_boxes) - len(matched_gt)  # Ground truth boxes not matched

        return {
            "average_iou": avg_iou,  # It should range in [0, 1]
            "unmatched_gt": unmatched_gt,
            "matched_gt": len(ious),
        }

    def _compute_average_iou_with_labels_across_iou(
        self, pred_boxes, pred_labels, gt_boxes, gt_labels
    ):

        res_50 = self._compute_average_iou_with_labels(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.50
        )
        res_75 = self._compute_average_iou_with_labels(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.75
        )
        res_90 = self._compute_average_iou_with_labels(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.90
        )
        res_95 = self._compute_average_iou_with_labels(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.95
        )

        return {
            "average_iou_50": res_50["average_iou"],
            "average_iou_75": res_75["average_iou"],
            "average_iou_90": res_90["average_iou"],
            "average_iou_95": res_95["average_iou"],
        }

    def _find_intersecting_labels(
        self,
        ds: Dataset,
    ) -> tuple[dict[str, int], dict[str, int], list[DocItemLabel]]:
        r"""
        Compute counters per labels for the groundtruth, prediciton and their intersections

        Returns
        -------
        true_labels: dict[label -> counter]
        pred_labels: dict[label -> counter]
        intersection_labels: list[DocItemLabel]
        """

        true_labels: Dict[str, int] = {}
        pred_labels: Dict[str, int] = {}

        for i, data in tqdm(
            enumerate(ds), desc="Layout evaluations", ncols=120, total=len(ds)
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            true_doc = data_record.ground_truth_doc
            pred_doc = self._get_pred_doc(data_record)

            for item, level in true_doc.iterate_items():
                if isinstance(item, DocItem):  # and item.label in filter_labels:
                    for prov in item.prov:
                        if item.label in [
                            self.label_mapping[v] for v in true_labels if v is not None  # type: ignore
                        ]:
                            true_labels[item.label] += 1
                        elif self.label_mapping[item.label]:
                            true_labels[self.label_mapping[item.label]] = 1  # type: ignore

            if pred_doc:
                for item, level in pred_doc.iterate_items():
                    if isinstance(item, DocItem):  # and item.label in filter_labels:
                        for prov in item.prov:
                            if item.label in [
                                self.label_mapping[v] for v in pred_labels if v is not None  # type: ignore
                            ]:
                                pred_labels[item.label] += 1
                            elif self.label_mapping[item.label] is not None:
                                pred_labels[self.label_mapping[item.label]] = 1  # type: ignore

        """
        logging.info(f"True labels:")
        for label, count in true_labels.items():
            logging.info(f" => {label}: {count}")

        logging.info(f"Pred labels:")
        for label, count in pred_labels.items():
            logging.info(f" => {label}: {count}")
        """

        intersection_labels: List[DocItemLabel] = []
        for label, count in true_labels.items():
            if label in pred_labels:
                intersection_labels.append(DocItemLabel(label))

        return true_labels, pred_labels, intersection_labels

    def _extract_layout_data(
        self,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        filter_labels: List[DocItemLabel],
    ) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        r"""
        Filter to keep only bboxes from the given labels
        Convert each bbox to top-left-origin, normalize to page size and scale 100

        Returns
        -------
        ground_truths: List of dict with keys "bboxes", "labels" and values are tensors
        predictions: List of dict with keys "bboxes", "labels", "scores" and values are tensors
        """
        assert len(true_doc.pages) == len(
            pred_doc.pages
        ), f"len(true_doc.pages)==len(pred_doc.pages) => {len(true_doc.pages)}=={len(pred_doc.pages)}"

        # page_num -> List[DocItem]
        true_pages_to_objects: Dict[int, List[DocItem]] = {}
        pred_pages_to_objects: Dict[int, List[DocItem]] = {}

        for item, level in true_doc.iterate_items():
            if (
                isinstance(item, DocItem)
                and self.label_mapping[item.label] in filter_labels
            ):
                for prov in item.prov:
                    if prov.page_no not in true_pages_to_objects:
                        true_pages_to_objects[prov.page_no] = [item]
                    else:
                        true_pages_to_objects[prov.page_no].append(item)

        for item, level in pred_doc.iterate_items():
            if (
                isinstance(item, DocItem)
                and self.label_mapping[item.label] in filter_labels
            ):
                for prov in item.prov:
                    if prov.page_no not in pred_pages_to_objects:
                        pred_pages_to_objects[prov.page_no] = [item]
                    else:
                        pred_pages_to_objects[prov.page_no].append(item)

        ground_truths = []
        predictions = []

        # DEBUG
        # true_tl_bboxes = []
        # pred_tl_bboxes = []

        for page_no, items in true_pages_to_objects.items():

            page_size = true_doc.pages[page_no].size

            page_height = page_size.height
            page_width = page_size.width

            bboxes = []
            labels = []
            for item in items:
                for prov in item.prov:
                    bbox = prov.bbox.to_top_left_origin(page_height=page_height)
                    # true_tl_bboxes.append(copy.deepcopy(bbox))

                    bbox = bbox.normalized(page_size)
                    bbox = bbox.scaled(100.0)

                    # logging.info(f"ground-truth {page_no}: {page_width, page_height} -> {item.label}, {bbox.coord_origin}: [{bbox.l}, {bbox.t}, {bbox.r}, {bbox.b}]")

                    bboxes.append([bbox.l, bbox.t, bbox.r, bbox.b])
                    labels.append(filter_labels.index(self.label_mapping[item.label]))  # type: ignore

            ground_truths.append(
                {
                    "boxes": torch.tensor(bboxes),
                    "labels": torch.tensor(labels),
                }
            )

        for page_no, items in pred_pages_to_objects.items():

            page_size = pred_doc.pages[page_no].size

            page_height = page_size.height
            page_width = page_size.width

            bboxes = []
            labels = []
            scores = []
            for item in items:
                for prov in item.prov:
                    bbox = prov.bbox.to_top_left_origin(page_height=page_height)
                    # pred_tl_bboxes.append(copy.deepcopy(bbox))

                    bbox = bbox.normalized(page_size)
                    bbox = bbox.scaled(100.0)

                    bboxes.append([bbox.l, bbox.t, bbox.r, bbox.b])
                    labels.append(filter_labels.index(self.label_mapping[item.label]))  # type: ignore
                    scores.append(1.0)  # FIXME

            predictions.append(
                {
                    "boxes": torch.tensor(bboxes),
                    "labels": torch.tensor(labels),
                    "scores": torch.tensor(scores),
                }
            )

        """
        assert len(ground_truths) == len(
            predictions
        ), f"len(ground_truths)==len(predictions) => {len(ground_truths)}=={len(predictions)}"
        """

        # Debug
        # true_tl_bboxes_str = "\n".join([json.dumps(b.model_dump(include={"b", "l", "t", "r"})) for b in true_tl_bboxes])
        # pred_tl_bboxes_str = "\n".join([json.dumps(b.model_dump(include={"b", "l", "t", "r"})) for b in pred_tl_bboxes])
        # print(f"Doc id: {doc_id}")
        # print(f"True bboxes: [{len(true_tl_bboxes)}]")
        # print(true_tl_bboxes_str)
        # print(f"Pred bboxes: [{len(pred_tl_bboxes)}]")
        # print(pred_tl_bboxes_str)

        return ground_truths, predictions
