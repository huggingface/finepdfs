"""
Basic inference pipeline example without chunking.

This example shows how to run inference on documents using the InferenceRunner
without chunking. Documents are processed and saved to a simple output structure.
"""

import asyncio
import argparse
from typing import Any
from datatrove.data import Document
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner, InferenceResult
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.executor.local import LocalPipelineExecutor


async def rollout_teacher(document: Document, generate: Any, **kwargs) -> Any:
    import re

    FW_EDU = """\
Below is an extract from a PDF file. Evaluate whether the extract has a high educational
value and could be useful in an educational setting for teaching from primary school to
grade school levels using the additive 5-point scoring system described below. Points are
accumulated based on the satisfaction of each criterion:
- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and
promotional material.
- Add another point if the extract addresses certain elements pertinent to education but
does not align closely with educational standards. It might mix educational content with
non-educational material, offering a superficial overview of potentially useful topics, or
presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key
concepts relevant to school curricula. It is coherent though it may not be comprehensive
or could include some extraneous information. It may resemble an introductory section of
a textbook or a basic tutorial that is suitable for learning but has notable limitations like
treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes
for a level not higher than grade school, exhibiting a clear and consistent writing style. It
could be similar to a chapter from a textbook or a tutorial, offering substantial educational
content, including exercises and solutions, with minimal irrelevant information, and the
concepts aren’t too advanced for grade school students. The content is coherent, focused,
and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for
teaching either at primary school or grade school. It follows detailed reasoning, the writing
style is easy to follow and offers profound and thorough insights into the subject matter,
devoid of any non-educational or complex content.
The extract: {example}.
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score: <total points>"\
"""

    def parse_fw_edu_score(result: InferenceResult):
        if not isinstance(result, InferenceResult):
            return None
        match = re.search(r"Educational score:\s*(\d+)", result.text)
        if match:
            return int(match.group(1))
        return None

    chunks = document.metadata["chunks"]
    tasks = []
    for chunk in chunks:
        prompt = FW_EDU.format(example=chunk)
        tasks.append(generate({
            "messages": [
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            "max_tokens": 512,
        }))
    
    # generate returns a list of results for each request
    all_inference_results = await asyncio.gather(*tasks)
    
    # all_inference_results is a list (one per chunk) of lists (one per request, here 1)
    fw_edu_scores = []
    for chunk_results in all_inference_results:
        # Each chunk_results is a list of InferenceResult | InferenceError
        chunk_score = parse_fw_edu_score(chunk_results[0]) if chunk_results else None
        fw_edu_scores.append(chunk_score)
    
    document.metadata["fw_edu_scores"] = [fw_edu_scores]
    return document



INPUT_DIR = "./finepdfs/data/exact_dedup/per_language/output"
OUTPUT_DIR = "./finepdfs/data/classification_only_top_bottom_300k/teacher"


def run_pipeline_for_tokens():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--languages", type=str, default="eng_Latn")
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    model_name = args.model_name
    model_kwargs = {}
    # model_kwargs = {
    #     "limit-mm-per-prompt.image": 0,
    #     "limit-mm-per-prompt.video": 0,
    # }
    if "oss" not in model_name and "FP8" not in model_name:
        model_kwargs["quantization"] = "fp8"

    if "mistral" in model_name:
        model_kwargs["tokenizer-mode"] = "mistral"

    sampling_rate = 1.0 # Compute so that you get ~300k samples
    

    config = InferenceConfig(
        server_type="vllm",
        model_name_or_path=model_name,
        default_generation_params={"temperature": 0.0},
        model_max_context=16384,
        max_concurrent_generations=400,
        model_kwargs=model_kwargs,
        tp=args.gpus,
    )

    languages = args.languages.split(",")

    for lang in languages:
        pipeline=[
            JsonlReader(
                data_folder=f"{INPUT_DIR}/{lang}",
                glob_pattern=f"*.jsonl.gz",
                shuffle_files=True,
                doc_progress=True,

            ),
            #
            SamplerFilter(
                rate=sampling_rate,
            ),
            InferenceRunner(
                rollout_fn=rollout_teacher,
                config=config,
                output_writer=JsonlWriter(
                    output_folder=f"{OUTPUT_DIR}/{lang}/{args.model_name.replace('/', '_')}",
                    output_filename="${rank}_chunk_${chunk_index}.jsonl.gz",  # Chunked filename pattern
                ),
            ),
        ]
        executor = LocalPipelineExecutor(pipeline)
        executor.run()
    

if __name__ == "__main__":
    run_pipeline_for_tokens()




