import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, Optional, Tuple

import typer
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PaginatedPipelineOptions,
    PdfPipelineOptions,
    VlmPipelineOptions,
    smoldocling_vlm_conversion_options,
    smoldocling_vlm_mlx_conversion_options,
)
from docling.document_converter import FormatOption, PdfFormatOption
from docling.models.factories import get_ocr_factory
from docling.pipeline.vlm_pipeline import VlmPipeline
from tabulate import tabulate  # type: ignore

from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
)
from docling_eval.dataset_builders.cvat_dataset_builder import CvatDatasetBuilder
from docling_eval.dataset_builders.cvat_preannotation_builder import (
    CvatPreannotationBuilder,
)
from docling_eval.dataset_builders.doclaynet_v1_builder import DocLayNetV1DatasetBuilder
from docling_eval.dataset_builders.doclaynet_v2_builder import DocLayNetV2DatasetBuilder
from docling_eval.dataset_builders.docvqa_builder import DocVQADatasetBuilder
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder
from docling_eval.dataset_builders.funsd_builder import FUNSDDatasetBuilder
from docling_eval.dataset_builders.omnidocbench_builder import (
    OmniDocBenchDatasetBuilder,
)
from docling_eval.dataset_builders.otsl_table_dataset_builder import (
    FintabNetDatasetBuilder,
    PubTables1MDatasetBuilder,
    PubTabNetDatasetBuilder,
)
from docling_eval.dataset_builders.xfund_builder import XFUNDDatasetBuilder
from docling_eval.evaluators.bbox_text_evaluator import BboxTextEvaluator
from docling_eval.evaluators.layout_evaluator import (
    DatasetLayoutEvaluation,
    LayoutEvaluator,
)
from docling_eval.evaluators.markdown_text_evaluator import (
    DatasetMarkdownEvaluation,
    MarkdownTextEvaluator,
)
from docling_eval.evaluators.readingorder_evaluator import (
    DatasetReadingOrderEvaluation,
    ReadingOrderEvaluator,
    ReadingOrderVisualizer,
)
from docling_eval.evaluators.stats import DatasetStatistics
from docling_eval.evaluators.table_evaluator import (
    DatasetTableEvaluation,
    TableEvaluator,
)
from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider
from docling_eval.prediction_providers.file_provider import FilePredictionProvider
from docling_eval.prediction_providers.tableformer_provider import (
    TableFormerPredictionProvider,
)

# Configure logging
logging.getLogger("docling").setLevel(logging.WARNING)
_log = logging.getLogger(__name__)

app = typer.Typer(
    name="docling-eval",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class PredictionProviderType(str, Enum):
    """Types of prediction providers available."""

    DOCLING = "docling"
    TABLEFORMER = "tableformer"
    FILE = "file"
    SMOLDOCLING = "smoldocling"


def log_and_save_stats(
    odir: Path,
    benchmark: BenchMarkNames,
    modality: EvaluationModality,
    metric: str,
    stats: DatasetStatistics,
    log_filename: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    For the given DatasetStatistics, generate textual table and plot.

    Args:
        odir: Output directory
        benchmark: Benchmark name
        modality: Evaluation modality
        metric: Metric name
        stats: Dataset statistics
        log_filename: Optional log filename

    Returns:
        Tuple of (log_filename, fig_filename)
    """
    log_mode = "a"
    if log_filename is None:
        log_filename = (
            odir / f"evaluation_{benchmark.value}_{modality.value}_{metric}.txt"
        )
        log_mode = "w"
    fig_filename = odir / f"evaluation_{benchmark.value}_{modality.value}_{metric}.png"
    stats.save_histogram(figname=fig_filename, name=metric)

    data, headers = stats.to_table(metric)
    content = f"{benchmark.value} {modality.value} {metric}: "
    content += "mean={:.2f} median={:.2f} std={:.2f}\n\n".format(
        stats.mean, stats.median, stats.std
    )
    content += tabulate(data, headers=headers, tablefmt="github")
    content += "\n\n\n"

    _log.info(content)
    with open(log_filename, log_mode) as fd:
        fd.write(content)
        _log.info("Saving statistics report to %s", log_filename)

    return log_filename, fig_filename


def get_dataset_builder(
    benchmark: BenchMarkNames,
    target: Path,
    split: str = "test",
    begin_index: int = 0,
    end_index: int = -1,
    dataset_source: Optional[Path] = None,
):
    """Get the appropriate dataset builder for the given benchmark."""
    common_params = {
        "target": target,
        "split": split,
        "begin_index": begin_index,
        "end_index": end_index,
    }

    if benchmark == BenchMarkNames.DPBENCH:
        return DPBenchDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.DOCLAYNETV1:
        return DocLayNetV1DatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.DOCLAYNETV2:
        if dataset_source is None:
            raise ValueError("dataset_path is required for DocLayNetV2")
        return DocLayNetV2DatasetBuilder(dataset_source=dataset_source, **common_params)  # type: ignore

    elif benchmark == BenchMarkNames.FUNSD:
        if dataset_source is None:
            raise ValueError("dataset_source is required for FUNSD")
        return FUNSDDatasetBuilder(dataset_source=dataset_source, **common_params)  # type: ignore

    elif benchmark == BenchMarkNames.XFUND:
        if dataset_source is None:
            raise ValueError("dataset_source is required for XFUND")
        return XFUNDDatasetBuilder(dataset_source=dataset_source, **common_params)  # type: ignore

    elif benchmark == BenchMarkNames.OMNIDOCBENCH:
        return OmniDocBenchDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.FINTABNET:
        return FintabNetDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.PUB1M:
        return PubTables1MDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.PUBTABNET:
        return PubTabNetDatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.DOCVQA:
        return DocVQADatasetBuilder(**common_params)  # type: ignore

    elif benchmark == BenchMarkNames.CVAT:
        assert dataset_source is not None
        return CvatDatasetBuilder(
            name="CVAT", dataset_source=dataset_source, target=target, split=split
        )

    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


def get_prediction_provider(
    provider_type: PredictionProviderType,
    file_source_path: Optional[Path] = None,
    file_prediction_format: Optional[PredictionFormats] = None,
    artifacts_path: Optional[Path] = None,
):
    pipeline_options: PaginatedPipelineOptions
    """Get the appropriate prediction provider with default settings."""
    if provider_type == PredictionProviderType.DOCLING:

        # ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
        #     kind="easyocr",
        # )

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=False,
        )

        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        if artifacts_path is not None:
            pipeline_options.artifacts_path = artifacts_path

        return DoclingPredictionProvider(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            },
            do_visualization=True,
            ignore_missing_predictions=True,
        )

    elif provider_type == PredictionProviderType.SMOLDOCLING:
        pipeline_options = VlmPipelineOptions()

        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        pipeline_options.vlm_options = smoldocling_vlm_conversion_options
        if sys.platform == "darwin":
            try:
                import mlx_vlm  # type: ignore

                pipeline_options.vlm_options = smoldocling_vlm_mlx_conversion_options

                if artifacts_path is not None:
                    pipeline_options.artifacts_path = artifacts_path

            except ImportError:
                _log.warning(
                    "To run SmolDocling faster, please install mlx-vlm:\n"
                    "pip install mlx-vlm"
                )

        pdf_format_option = PdfFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
        )

        format_options: Dict[InputFormat, FormatOption] = {
            InputFormat.PDF: pdf_format_option,
            InputFormat.IMAGE: pdf_format_option,
        }

        return DoclingPredictionProvider(
            format_options=format_options,
            do_visualization=True,
            ignore_missing_predictions=True,
        )

    elif provider_type == PredictionProviderType.TABLEFORMER:
        return TableFormerPredictionProvider(
            do_visualization=False,
            ignore_missing_predictions=True,
        )

    elif provider_type == PredictionProviderType.FILE:
        if file_prediction_format is None:
            raise ValueError("file_prediction_format is required for File provider")
        if file_source_path is None:
            raise ValueError("file_source_path is required for File provider")

        return FilePredictionProvider(
            prediction_format=file_prediction_format,
            source_path=file_source_path,
            do_visualization=True,
            ignore_missing_predictions=True,
            ignore_missing_files=True,
            use_ground_truth_page_images=False,
        )

    else:
        raise ValueError(f"Unsupported prediction provider: {provider_type}")


def evaluate(
    modality: EvaluationModality,
    benchmark: BenchMarkNames,
    idir: Path,
    odir: Path,
    split: str = "test",
):
    """
    Evaluate predictions against ground truth.

    Args:
        modality: Evaluation modality
        benchmark: Benchmark name
        idir: Input directory with dataset
        odir: Output directory for results
        split: Dataset split
        begin_index: Begin index
        end_index: End index
    """
    if not os.path.exists(idir):
        _log.error(f"Benchmark directory not found: {idir}")
        return

    os.makedirs(odir, exist_ok=True)

    # Save the evaluation
    save_fn = odir / f"evaluation_{benchmark.value}_{modality.value}.json"

    if modality == EvaluationModality.END2END:
        _log.error("END2END evaluation not supported. ")

    elif modality == EvaluationModality.LAYOUT:
        layout_evaluator = LayoutEvaluator()
        layout_evaluation = layout_evaluator(
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(layout_evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.TABLE_STRUCTURE:
        table_evaluator = TableEvaluator()
        table_evaluation = table_evaluator(
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(table_evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.READING_ORDER:
        readingorder_evaluator = ReadingOrderEvaluator()
        readingorder_evaluation = readingorder_evaluator(
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(
                readingorder_evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    elif modality == EvaluationModality.MARKDOWN_TEXT:
        md_evaluator = MarkdownTextEvaluator()
        md_evaluation = md_evaluator(
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(
                md_evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    elif modality == EvaluationModality.BBOXES_TEXT:
        bbox_evaluator = BboxTextEvaluator()
        bbox_evaluation = bbox_evaluator(
            idir,
            split=split,
        )

        with open(save_fn, "w") as fd:
            json.dump(
                bbox_evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    else:
        _log.error(f"Unsupported modality for evaluation: {modality}")
        return

    _log.info(f"The evaluation has been saved in '{save_fn}'")


def visualize(
    modality: EvaluationModality,
    benchmark: BenchMarkNames,
    idir: Path,
    odir: Path,
    split: str = "test",
):
    """
    Visualize evaluation results.

    Args:
        modality: Visualization modality
        benchmark: Benchmark name
        idir: Input directory with dataset
        odir: Output directory for visualizations
        split: Dataset split
        begin_index: Begin index
        end_index: End index
    """
    if not os.path.exists(idir):
        _log.error(f"Input directory not found: {idir}")
        return

    os.makedirs(odir, exist_ok=True)
    metrics_filename = odir / f"evaluation_{benchmark.value}_{modality.value}.json"

    if not os.path.exists(metrics_filename):
        _log.error(f"Metrics file not found: {metrics_filename}")
        _log.error("You need to run evaluation first before visualization")
        return

    if modality == EvaluationModality.END2END:
        _log.error("END2END visualization not supported")

    elif modality == EvaluationModality.LAYOUT:
        try:
            with open(metrics_filename, "r") as fd:
                layout_evaluation = DatasetLayoutEvaluation.model_validate_json(
                    fd.read()
                )

            # Save layout statistics for mAP
            log_filename, _ = log_and_save_stats(
                odir,
                benchmark,
                modality,
                "mAP_0.5_0.95",
                layout_evaluation.image_mAP_stats,
            )

            # Append to layout statistics, the AP per classes
            data, headers = layout_evaluation.to_table()
            content = "\n\n\nAP[0.5:0.05:0.95] per class (reported as %):\n\n"
            content += tabulate(data, headers=headers, tablefmt="github")

            # Append to layout statistics, the mAP
            content += "\n\nTotal mAP[0.5:0.05:0.95] (reported as %): {:.2f}".format(
                100.0 * layout_evaluation.mAP
            )
            _log.info(content)
            with open(log_filename, "a") as fd:
                fd.write(content)
        except Exception as e:
            _log.error(f"Error processing layout evaluation: {str(e)}")

    elif modality == EvaluationModality.TABLE_STRUCTURE:
        try:
            with open(metrics_filename, "r") as fd:
                table_evaluation = DatasetTableEvaluation.model_validate_json(fd.read())

            figname = (
                odir
                / f"evaluation_{benchmark.value}_{modality.value}-delta_row_col.png"
            )
            table_evaluation.save_histogram_delta_row_col(figname=figname)

            # TEDS struct-with-text
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "TEDS_struct-with-text",
                table_evaluation.TEDS,
            )

            # TEDS struct-only
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "TEDS_struct-only",
                table_evaluation.TEDS_struct,
            )
        except Exception as e:
            _log.error(f"Error processing table evaluation: {str(e)}")

    elif modality == EvaluationModality.READING_ORDER:
        try:
            with open(metrics_filename, "r") as fd:
                ro_evaluation = DatasetReadingOrderEvaluation.model_validate_json(
                    fd.read()
                )

            # ARD
            log_and_save_stats(
                odir, benchmark, modality, "ARD_norm", ro_evaluation.ard_stats
            )

            # Weighted ARD
            log_and_save_stats(
                odir, benchmark, modality, "weighted_ARD", ro_evaluation.w_ard_stats
            )

            # Generate visualizations of the reading order across the GT and the prediction
            ro_visualizer = ReadingOrderVisualizer()
            ro_visualizer(
                idir,
                metrics_filename,
                odir,
                split=split,
            )
        except Exception as e:
            _log.error(f"Error processing reading order evaluation: {str(e)}")

    elif modality == EvaluationModality.MARKDOWN_TEXT:
        try:
            with open(metrics_filename, "r") as fd:
                md_evaluation = DatasetMarkdownEvaluation.model_validate_json(fd.read())

            # Log stats for all metrics in the same file
            log_filename = odir / f"evaluation_{benchmark.value}_{modality.value}.txt"
            with open(log_filename, "w") as fd:
                fd.write(
                    f"{benchmark.value} size: {len(md_evaluation.evaluations)}\n\n"
                )

            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "BLEU",
                md_evaluation.bleu_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "F1",
                md_evaluation.f1_score_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "precision",
                md_evaluation.precision_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "recall",
                md_evaluation.recall_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "edit_distance",
                md_evaluation.edit_distance_stats,
                log_filename=log_filename,
            )
            log_and_save_stats(
                odir,
                benchmark,
                modality,
                "meteor",
                md_evaluation.meteor_stats,
                log_filename=log_filename,
            )
        except Exception as e:
            _log.error(f"Error processing markdown text evaluation: {str(e)}")

    else:
        _log.error(f"Unsupported modality for visualization: {modality}")


@app.command()
def create_cvat(
    output_dir: Annotated[Path, typer.Option(help="Output directory")],
    gt_dir: Annotated[Path, typer.Option(help="Dataset source path")],
    bucket_size: Annotated[int, typer.Option(help="Size of CVAT tasks")] = 20,
):
    builder = CvatPreannotationBuilder(
        dataset_source=gt_dir, target=output_dir, bucket_size=bucket_size
    )
    builder.prepare_for_annotation()


@app.command()
def create_gt(
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")],
    dataset_source: Annotated[
        Optional[Path], typer.Option(help="Dataset source path")
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    begin_index: Annotated[int, typer.Option(help="Begin index (inclusive)")] = 0,
    end_index: Annotated[
        int, typer.Option(help="End index (exclusive), -1 for all")
    ] = -1,
):
    """Create ground truth dataset only."""
    gt_dir = output_dir / "gt_dataset"

    try:
        dataset_builder = get_dataset_builder(
            benchmark=benchmark,
            target=gt_dir,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
            dataset_source=dataset_source,
        )

        # Retrieve and save the dataset
        if dataset_builder.must_retrieve:
            dataset_builder.retrieve_input_dataset()
        dataset_builder.save_to_disk(chunk_size=80)

        _log.info(f"Ground truth dataset created at {gt_dir}")
    except ValueError as e:
        _log.error(f"Error creating dataset builder: {str(e)}")


@app.command()
def create_eval(
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    output_dir: Annotated[Path, typer.Option(help="Output directory.")],
    prediction_provider: Annotated[
        PredictionProviderType, typer.Option(help="Type of prediction provider to use")
    ],
    gt_dir: Annotated[
        Optional[Path], typer.Option(help="Input directory for GT")
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    begin_index: Annotated[int, typer.Option(help="Begin index (inclusive)")] = 0,
    end_index: Annotated[
        int, typer.Option(help="End index (exclusive), -1 for all")
    ] = -1,
    # File provider required options
    file_prediction_format: Annotated[
        Optional[str],
        typer.Option(
            help="Prediction format for File provider (required if using FILE provider)"
        ),
    ] = None,
    file_source_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Source path for File provider (required if using FILE provider)"
        ),
    ] = None,
    artifacts_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory for local model artifacts. Will only be passed to providers supporting this."
        ),
    ] = None,
):
    """Create evaluation dataset from existing ground truth."""
    gt_dir = gt_dir or output_dir / "gt_dataset"
    pred_dir = output_dir / "eval_dataset"

    # Check if ground truth exists
    if not gt_dir.exists():
        _log.error(f"Ground truth directory not found: {gt_dir}")
        _log.error(
            "Cannot create eval dataset without ground truth. Run create_gt first."
        )
        return

    try:
        # Convert string option to enum value
        file_format = (
            PredictionFormats(file_prediction_format)
            if file_prediction_format
            else None
        )

        # Create the appropriate prediction provider
        provider = get_prediction_provider(
            provider_type=prediction_provider,
            file_source_path=file_source_path,
            file_prediction_format=file_format,
            artifacts_path=artifacts_path,
        )

        # Get the dataset name from the benchmark
        dataset_name = f"{benchmark.value}"

        # Create predictions
        provider.create_prediction_dataset(
            name=dataset_name,
            gt_dataset_dir=gt_dir,
            target_dataset_dir=pred_dir,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        _log.info(f"Evaluation dataset created at {pred_dir}")
    except ValueError as e:
        _log.error(f"Error creating prediction provider: {str(e)}")


@app.command()
def create(
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")],
    dataset_source: Annotated[
        Optional[Path], typer.Option(help="Dataset source path")
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    begin_index: Annotated[int, typer.Option(help="Begin index (inclusive)")] = 0,
    end_index: Annotated[
        int, typer.Option(help="End index (exclusive), -1 for all")
    ] = -1,
    prediction_provider: Annotated[
        Optional[PredictionProviderType],
        typer.Option(help="Type of prediction provider to use"),
    ] = None,
    file_prediction_format: Annotated[
        Optional[str], typer.Option(help="Prediction format for File provider")
    ] = None,
    file_source_path: Annotated[
        Optional[Path], typer.Option(help="Source path for File provider")
    ] = None,
):
    """Create both ground truth and evaluation datasets in one step."""
    # First create ground truth
    create_gt(
        benchmark=benchmark,
        output_dir=output_dir,
        dataset_source=dataset_source,
        split=split,
        begin_index=begin_index,
        end_index=end_index,
    )

    # Then create evaluation if provider specified
    if prediction_provider:
        create_eval(
            benchmark=benchmark,
            output_dir=output_dir,
            prediction_provider=prediction_provider,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
            file_prediction_format=file_prediction_format,
            file_source_path=file_source_path,
        )
    else:
        _log.info(
            "No prediction provider specified, skipping evaluation dataset creation"
        )


@app.command(name="evaluate")
def evaluate_cmd(
    modality: Annotated[EvaluationModality, typer.Option(help="Evaluation modality")],
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    output_dir: Annotated[Path, typer.Option(help="Base output directory")],
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
):
    """Evaluate predictions against ground truth."""
    # Derive input and output paths based on the directory structure in test_dataset_builder.py
    input_dir = output_dir / "eval_dataset"
    eval_output_dir = output_dir / "evaluations" / modality.value

    # Create output directory
    os.makedirs(eval_output_dir, exist_ok=True)

    # Call our self-contained evaluation function
    evaluate(
        modality=modality,
        benchmark=benchmark,
        idir=input_dir,
        odir=eval_output_dir,
        split=split,
    )


@app.command(name="visualize")
def visualize_cmd(
    modality: Annotated[
        EvaluationModality, typer.Option(help="Visualization modality")
    ],
    benchmark: Annotated[BenchMarkNames, typer.Option(help="Benchmark name")],
    output_dir: Annotated[Path, typer.Option(help="Base output directory")],
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    begin_index: Annotated[int, typer.Option(help="Begin index (inclusive)")] = 0,
    end_index: Annotated[
        int, typer.Option(help="End index (exclusive), -1 for all")
    ] = -1,
):
    """Visualize evaluation results."""
    # Derive input and output paths based on the directory structure in test_dataset_builder.py
    input_dir = output_dir / "eval_dataset"
    eval_output_dir = output_dir / "evaluations" / modality.value

    # Create output directory
    os.makedirs(eval_output_dir, exist_ok=True)

    # Call our self-contained visualization function
    visualize(
        modality=modality,
        benchmark=benchmark,
        idir=input_dir,
        odir=eval_output_dir,
        split=split,
    )


@app.callback()
def main():
    """Docling Evaluation CLI for benchmarking document processing tasks."""
    pass


if __name__ == "__main__":
    app()
