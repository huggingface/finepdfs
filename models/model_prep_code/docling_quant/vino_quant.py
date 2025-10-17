import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).parent
sys.path.append(CURRENT_DIR / "docling-eval")
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.document import InputDocument
from docling.datamodel.base_models import InputFormat
import sys
import numpy as np # Keep if needed by ObjectDetectionDataReader or other parts
import nncf
import openvino as ov
import json # Keep if potentially needed for config or future extensions
import os
import argparse
import shutil # Keep for potential directory operations
import shutil
import numpy as np
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import os
from docling.document_converter import PdfFormatOption
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.evaluators.layout_evaluator import LayoutEvaluator
from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider
import json

# Assuming ObjectDetectionDataReader is correctly defined and imported
# It might depend on some docling imports, ensure they are available
from aux.data_reader import ObjectDetectionDataReader
from pathlib import Path


input_folder = CURRENT_DIR / "data/pdfs"

def convert_to_pngs():
    from pathlib import Path
    pdf_files = list(Path(input_folder).glob("*.pdf"))
    print(len(pdf_files))
    import os
    os.makedirs(input_folder.parent / "pngs", exist_ok=True)
    for file in pdf_files:
        file_name = file.stem
        input_doc = InputDocument(file, InputFormat.PDF, DoclingParseV4DocumentBackend)
        try:
            doc = DoclingParseV4DocumentBackend(input_doc, path_or_stream=file)
        except Exception as e:
            continue
        for idx, page_i in enumerate(range(doc.page_count())):
            page_i = doc.load_page(page_i)
            image = page_i.get_page_image()
            image.save(input_folder.parent / "pngs" / f"{file_name}_{idx}.png")


def run_create_eval(benchmark, output_dir, end_index, begin_index, model, preprocess_in_vino=False):
    # Create the evaluation dataset
    gt_dir = output_dir / "gt_dataset"
    pred_dir = output_dir / "eval_dataset"



    # Configure Docling Prediction Provider
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        images_scale=2.0,
        generate_page_images=True,
        generate_picture_images=True,
    )
    # Monkeypatch the model in docling
    provider = DoclingPredictionProvider(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        },
        do_visualization=True, # Keep visualization for potential debugging
        ignore_missing_predictions=True,
    )
    provider.doc_converter.initialize_pipeline(InputFormat.PDF)
    provider.doc_converter._get_pipeline(InputFormat.PDF).build_pipe[2].layout_predictor.init_vino_model(model, preprocess_in_vino)

    # Get the dataset name from the benchmark
    dataset_name = benchmark

    # Create predictions
    provider.create_prediction_dataset(
        name=dataset_name,
        gt_dataset_dir=gt_dir,
        target_dataset_dir=pred_dir,
        split="test",
        begin_index=begin_index,
        end_index=end_index,
    )

    # Actually evaluate the predictions


def run_evaluate(
    benchmark: BenchMarkNames,
    output_dir: Path,
    split: str = "test",
    begin_index: int = 0,
    end_index: int = -1,
):
    """Evaluates the layout modality."""
    input_dir = output_dir / "eval_dataset"
    eval_output_dir = output_dir / "evaluations" / EvaluationModality.LAYOUT.value
    modality = EvaluationModality.LAYOUT

    if not input_dir.exists():
        return False

    os.makedirs(eval_output_dir, exist_ok=True)
    save_fn = eval_output_dir / f"evaluation_{benchmark}_{modality.value}.json"

    layout_evaluator = LayoutEvaluator()
    layout_evaluation = layout_evaluator(
        input_dir,
        split=split,
        begin_index=begin_index,
        end_index=end_index,
    )

    with open(save_fn, "w") as fd:
        # Use model_dump_json for direct serialization if available and preferred
        # otherwise fall back to model_dump and json.dump
        if hasattr(layout_evaluation, 'model_dump_json'):
            fd.write(layout_evaluation.model_dump_json(indent=2))
        else:
            import json
            json.dump(layout_evaluation.model_dump(), fd, indent=2, sort_keys=True)

def validation_function(model, data_loader, output_dir: Path | None = None, preprocess_in_vino=False):
    if not output_dir:
        delete_output_dir = True
        output_dir = CURRENT_DIR / "tmp/quant_with_validation"
    else:
        delete_output_dir = False
        if not (output_dir / "gt_dataset").exists():
            shutil.copytree(CURRENT_DIR / "tmp/gt_dataset", output_dir / "gt_dataset")
    benchmark = "DocLayNetV1"
    # Run create_eval
    data_loader_mat = list(data_loader)
    begin_index = min(data_loader_mat)
    end_index = max(data_loader_mat)
    run_create_eval(benchmark=benchmark, output_dir=output_dir, end_index=end_index, begin_index=begin_index, model=model, preprocess_in_vino=preprocess_in_vino)
    # Run evaluate
    run_evaluate(benchmark=benchmark, output_dir=output_dir, split="test", begin_index=begin_index, end_index=end_index)

    # Open the evaluation results
    with open(output_dir / "evaluations" / EvaluationModality.LAYOUT.value / f"evaluation_{benchmark}_{EvaluationModality.LAYOUT.value}.json", "r") as fd:
        evaluation = json.load(fd)

    per_image_map = [x["map_val"] for x in evaluation["evaluations_per_image"]]
    mean_map = np.mean(per_image_map)

    # Delete the new results
    if delete_output_dir:
        shutil.rmtree(output_dir / "evaluations")
        shutil.rmtree(output_dir / "eval_dataset")
    return float(mean_map)

def quantize_model(base_model: ov.Model, calibration_dataset: nncf.Dataset, args: argparse.Namespace) -> ov.Model:
    """
    Quantizes an OpenVINO model using NNCF based on provided arguments.
    Accuracy control is NOT used in this version.

    Args:
        base_model: The original OpenVINO model to quantize.
        calibration_dataset: The NNCF dataset for calibration.
        args: Parsed command-line arguments containing quantization settings.

    Returns:
        The quantized OpenVINO model.
    """
    ignored = None
    if args.ignore_scope:
         # Define the scope to ignore during quantization, adjust node names as needed
        ignored = nncf.IgnoredScope(
            subgraphs=[
                nncf.Subgraph(
                    inputs=["aten::sigmoid/Sigmoid"], # Example input node name
                    outputs=["aten::gather/GatherElements", "aten::topk/TopK"], # Example output node names
                )
            ]
        )
        print("Ignoring specified subgraph during quantization.")

    preset = nncf.QuantizationPreset.PERFORMANCE # Default to PERFORMANCE
    if args.precision == "mixed":
        preset = nncf.QuantizationPreset.MIXED
        print("Using MIXED precision preset.")
    else:
         print("Using PERFORMANCE precision preset.")


    quant_args = {
        "model": base_model,
        "calibration_dataset": calibration_dataset,
        "target_device": nncf.TargetDevice.CPU, # Assuming CPU target
        "subset_size": args.subset_size,
        "fast_bias_correction": args.fast_bias_correction,
        "preset": preset,
        "ignored_scope": ignored,
    }

    print("Quantizing without Accuracy Control...")
    quantized_model = nncf.quantize(**quant_args)

    return quantized_model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=CURRENT_DIR / "models/rtdetr_layout_openvino.xml")
    parser.add_argument("--precision", type=str, default="mixed")
    parser.add_argument("--ignore-scope", default=False, action="store_true")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--eval-subset-size", type=int, default=1000)
    parser.add_argument("--fast-bias-correction", default=False, action="store_true")
    parser.add_argument("--use-accuracy-control", default=False, action="store_true")
    parser.add_argument("--just-eval", default=False, action="store_true")
    parser.add_argument("--preprocess-in-vino", default=False, action="store_true")
    parser.add_argument("--override-name", type=str, default=None)

    args = parser.parse_args()
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{args.precision}_subset_{args.subset_size}_fast_bias_{args.fast_bias_correction}_accuracy_control_{args.use_accuracy_control}_precision_{args.precision}_ignore_scope_{args.ignore_scope}_date_{date}"
    if args.override_name:
        name = args.override_name
    output_dir = Path(f"{CURRENT_DIR}/models/{name}")
    if not args.just_eval:
        input_reader = ObjectDetectionDataReader(CURRENT_DIR / "data/pngs")
        calibration_dataset = nncf.Dataset(input_reader)
        model = ov.Core().read_model(model=args.base_model)
        quantized_model = quantize_model(model, calibration_dataset, args)
        # Save the quantized model
        ov.save_model(quantized_model, output_dir / f"{name}.xml", compress_to_fp16=False)
    else:
        quantized_model = ov.Core().read_model(model=args.base_model)
    import time
    start_time = time.time()
    validation_function(quantized_model, range(args.eval_subset_size + 1), output_dir, preprocess_in_vino=True)
    end_time = time.time()
    with open(output_dir / "validation_time.txt", "w") as fd:
        fd.write(f"{end_time - start_time} seconds")


if __name__ == "__main__":
    main()