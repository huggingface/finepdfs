import glob
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from docling_core.types.doc.document import DoclingDocument, TableItem
from docling_core.types.doc.labels import DocItemLabel
from lxml import html
from pydantic import BaseModel
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
from docling_eval.evaluators.teds import TEDScorer

_log = logging.getLogger(__name__)


class TableEvaluation(UnitEvaluation):
    filename: str = "<unknown>"
    table_id: int = -1
    TEDS: float
    is_complex: bool = False

    true_ncols: int = -1
    pred_ncols: int = -1

    true_nrows: int = -1
    pred_nrows: int = -1


class DatasetTableEvaluation(DatasetEvaluation):
    evaluations: List[TableEvaluation]

    TEDS: DatasetStatistics
    TEDS_struct: DatasetStatistics
    TEDS_simple: DatasetStatistics
    TEDS_complex: DatasetStatistics

    def save_histogram_delta_row_col(self, figname: Path):

        delta_row = {i: 0 for i in range(-10, 11)}
        delta_col = {i: 0 for i in range(-10, 11)}

        for evaluation in self.evaluations:
            if evaluation.true_nrows - evaluation.pred_nrows in delta_row:
                delta_row[evaluation.true_nrows - evaluation.pred_nrows] += 1

            if evaluation.true_ncols - evaluation.pred_ncols in delta_col:
                delta_col[evaluation.true_ncols - evaluation.pred_ncols] += 1

        x_row, y_row = [], []
        for k, v in delta_row.items():
            x_row.append(k)
            if v == 0:
                y_row.append(1.0e-6)
            else:
                y_row.append(v / float(len(self.evaluations)))

        x_col, y_col = [], []
        for k, v in delta_col.items():
            x_col.append(k)
            if v == 0:
                y_col.append(1.0e-6)
            else:
                y_col.append(v / float(len(self.evaluations)))

        fignum = int(1000 * random.random())
        plt.figure(fignum)

        plt.semilogy(x_row, y_row, "k.-", label="rows_{true} - rows_{pred}")
        plt.semilogy(x_col, y_col, "r.-", label="cols_{true} - cols_{pred}")

        plt.xlabel("delta")
        plt.ylabel("%")
        plt.legend(loc="upper right")

        logging.info(f"saving figure to {figname}")
        plt.savefig(figname)


def is_complex_table(table: TableItem) -> bool:
    r"""
    Implement the logic to check if table is complex
    """
    for cell in table.data.table_cells:
        if cell.row_span > 1 or cell.col_span > 1:
            return True
    return False


class TableEvaluator(BaseEvaluator):
    r"""
    Evaluate table predictions from HF dataset with the columns:
    """

    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        structure_only: bool = False,
        prediction_sources: List[PredictionFormats] = [],
    ):
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
            PredictionFormats.DOCTAGS,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        self._structure_only = structure_only
        self._teds_scorer = TEDScorer()
        self._stopwords = ["<i>", "</i>", "<b>", "</b>", "<u>", "</u>"]

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetTableEvaluation:
        r"""
        Load a dataset in HF format. Expected columns with DoclingDocuments
        "GTDoclingDocument"
        "PredictionDoclingDocument"
        """
        logging.info("Loading the split '%s' from: '%s'", split, ds_path)

        # Load the dataset
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        logging.info("Files: %s", split_files)
        ds = load_dataset("parquet", data_files={split: split_files})
        logging.info("Overview of dataset: %s", ds)

        # Select the split
        ds_selection: Dataset = ds[split]

        table_evaluations = []
        table_struct_evaluations = []
        evaluation_errors = 0
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Table evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id
            gt_doc = data_record.ground_truth_doc
            pred_doc = self._get_pred_doc(data_record)
            if not pred_doc:
                _log.error("There is no prediction for doc_id=%s", doc_id)
                continue

            try:
                if not self._structure_only:
                    results = self._evaluate_tables_in_documents(
                        doc_id=doc_id,
                        true_doc=gt_doc,
                        pred_doc=pred_doc,
                        structure_only=False,
                    )
                    table_evaluations.extend(results)

                    if self._intermediate_evaluations_path:
                        self.save_intermediate_evalutions(
                            "TEDs_struct_content", i, doc_id, results
                        )

                results = self._evaluate_tables_in_documents(
                    doc_id=data[BenchMarkColumns.DOC_ID],
                    true_doc=gt_doc,
                    pred_doc=pred_doc,
                    structure_only=True,
                )
                table_struct_evaluations.extend(results)
                if self._intermediate_evaluations_path:
                    self.save_intermediate_evalutions("TEDs_struct", i, doc_id, results)

            except Exception as ex:
                evaluation_errors += 1
                _log.error("Error during tables evaluation for %s", doc_id)

        _log.info(
            "Finish. %s documents were skipped due to evaluation errors",
            evaluation_errors,
        )

        # Compute TED statistics for the entire dataset
        teds_simple = []
        teds_complex = []
        teds_all = []
        if not self._structure_only:
            for te in table_evaluations:
                teds_all.append(te.TEDS)

                if te.is_complex:
                    teds_complex.append(te.TEDS)
                else:
                    teds_simple.append(te.TEDS)

        teds_struct = []
        for te in table_struct_evaluations:
            teds_struct.append(te.TEDS)

        dataset_evaluation = DatasetTableEvaluation(
            evaluations=table_evaluations,
            TEDS=compute_stats(teds_all),
            TEDS_struct=compute_stats(teds_struct),
            TEDS_simple=compute_stats(teds_simple),
            TEDS_complex=compute_stats(teds_complex),
        )
        return dataset_evaluation

    def _evaluate_tables_in_documents(
        self,
        doc_id: str,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        structure_only: bool = False,
    ) -> List[TableEvaluation]:
        r""" """
        table_evaluations = []
        true_tables = true_doc.tables
        pred_tables = pred_doc.tables
        _log.info(
            "#-true-tables: %s, #-pred-tables: %s", len(true_tables), len(pred_tables)
        )
        assert len(true_tables) == len(
            pred_tables
        ), "len(true_tables)!=len(pred_tables)"

        for table_id in range(len(true_tables)):  # , len(pred_tables)):

            # Avoid items of type DocItemLabel.DOCUMENT_INDEX
            if true_tables[table_id].label != DocItemLabel.TABLE:
                logging.warning(
                    f"Skipping table with label {true_tables[table_id].label}"
                )
                continue

            try:
                true_table = true_tables[table_id]
                pred_table = pred_tables[table_id]

                is_complex = is_complex_table(true_table)

                true_html = true_table.export_to_html(true_doc)
                pred_html = pred_table.export_to_html(pred_doc)

                # Filter out tags that may be present in GT but not in prediction to avoid penalty
                for stopword in self._stopwords:
                    predicted_html = pred_html.replace(stopword, "")
                for stopword in self._stopwords:
                    true_html = true_html.replace(stopword, "")

                true_html_obj = html.fromstring(true_html)
                pred_html_obj = html.fromstring(pred_html)

                teds = self._teds_scorer(
                    gt_table=true_html_obj,
                    pred_table=pred_html_obj,
                    structure_only=structure_only,
                )
                # logging.info(f"teds: {teds}")

                teds = round(teds, 3)
                table_evaluation = TableEvaluation(
                    TEDS=teds,
                    is_complex=is_complex,
                    filename=doc_id,
                    table_id=table_id,
                    true_ncols=true_table.data.num_cols,
                    pred_ncols=pred_table.data.num_cols,
                    true_nrows=true_table.data.num_rows,
                    pred_nrows=pred_table.data.num_rows,
                )
                table_evaluations.append(table_evaluation)
            except Exception:
                logging.error(
                    f"Table {table_id} from document {doc_id} could not be compared!"
                )

        return table_evaluations

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
            elif prediction_format == PredictionFormats.DOCTAGS:
                pred_doc = docling_document_from_doctags(data_record)
            if pred_doc is not None:
                break

        return pred_doc
