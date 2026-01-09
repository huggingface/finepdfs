"""
Classify documents with Gemma and mark correctness.

Runs as a Slurm pipeline: reads the first --limit documents, performs Gemma-based
language check, and writes documents back with metadata["language_correct"]
set to True/False based on Gemma classification.
"""

import argparse
from typing import Any, AsyncGenerator, Dict
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceRunner,
    InferenceResult,
)
from langcodes import Language


INPUT_DIR = "./finepdfs/data/glotlid/per_language"
OUTPUT_DIR = "./finepdfs/data/gemma_classified"
DEFAULT_MODEL_NAME = "google/gemma-3-27b-it"



async def rollout_gemma(document: Document, generate: Any, **kwargs) -> Any:
    import re
    language = kwargs.get("language")
    model_name = kwargs.get("model_name")
    
    lang = Language.get(language.replace("-", "_"))
    language_name = lang.display_name()
    language_script = lang.script_name()

    pages_list = []
    start = 0
    for i, page_offset in enumerate(document.media[0].metadata["page_offsets"]):
        if document.metadata["best_page_languages"][i] == language:
            pages_list.append(document.text[start:page_offset])
        start = page_offset
    pages = "\n\n".join(pages_list)

    # Remove the tables
    table_pattern = r"^\s*\|.*\|\s*$"
    pages = re.sub(table_pattern, "", pages, re.MULTILINE)
    # Truncate all \n{2,} to \n\n
    pages = re.sub(r"\n{3,}", "\n\n", pages)

    # Use a tokenizer from kwargs or load it once
    tokenizer = kwargs.get("tokenizer")
    if tokenizer is None:
        from transformers import AutoTokenizer
        # This is a bit slow to do every time, but rollout_fn can receive a shared tokenizer via shared_context
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.encode(pages, add_special_tokens=False)
    tokens = tokens[:1024]
    text_to_analyze = tokenizer.decode(tokens)

    prompt = f"""\
## Task

Determine if the given text is written in {language_name} ({language}).

## Instructions

* If any part of the text is written in {language_name} using the {language_script} script, respond with: `YES`
* If no part of the text is in {language_name} using the {language_script} script, respond with: `NO`
* If the text is a creole, dialect, or related language to {language_name}, respond with: `NO` — even if it uses the {language_script} script.

## Output format

Respond with only: `YES` or `NO`

## Text to analyze

```
{text_to_analyze}
```"""

    results = await generate({
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "max_tokens": 3,
        "temperature": 0.0,
    })

    for result in results:
        if isinstance(result, InferenceResult):
            response = (result.text or "").strip().upper()
            if document.metadata is None:
                document.metadata = {}
            if response.startswith("YES"):
                document.metadata["language_correct"] = True
                return document
            elif response.startswith("NO"):
                document.metadata["language_correct"] = False
                return document
            else:
                document.metadata["language_correct"] = None
    return document


def build_pipeline(language: str, limit: int, model_name: str, output_folder: str):
    reader = JsonlReader(
        f"{INPUT_DIR}/{language}",
        glob_pattern="*.jsonl.gz",
        # Ideally this should be well shuffled, so that you don't take just docling or just ocr documents
        limit=limit,
    )

    SEQS = 64
    from transformers import AutoTokenizer
    inference = InferenceRunner(
        rollout_fn=rollout_gemma,
        shared_context={
            "language": language,
            "model_name": model_name,
            "tokenizer": AutoTokenizer.from_pretrained(model_name),
        },
        config=InferenceConfig(
            server_type="vllm",
            model_name_or_path=model_name,
            default_generation_params={"temperature": 1.0},
            model_max_context=2048+3,
            max_concurrent_generations=500,
            metric_interval=30,
            model_kwargs={"limit-mm-per-prompt.image": 0, "limit-mm-per-prompt.video": 0, "max-num-seqs": SEQS, "max-num-batched-tokens": SEQS*1203},
        ),
        output_writer=JsonlWriter(output_folder),
    )

    return [reader, inference]


def main():
    parser = argparse.ArgumentParser(description="Classify documents with Gemma and mark language correctness")
    parser.add_argument("--limit", type=int, required=True, help="Number of documents to read and classify per language")
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Model name or path for Gemma (default: google/gemma-3-27b-it)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        help="Comma-separated list of language codes to process (e.g., 'eng_Latn,spa_Latn'). If not provided, uses all languages from CSV.",
    )
    args = parser.parse_args()

    limit = args.limit
    model_name = args.model_name

    languages = [lang.strip() for lang in args.languages.split(',') if lang.strip()]

    for language in languages:
        output_folder = f"{OUTPUT_DIR}_{limit}/{language}"

        pipeline = build_pipeline(
            language=language,
            limit=limit,
            model_name=model_name,
            output_folder=output_folder,
        )
        compute = LocalPipelineExecutor(
            pipeline=pipeline,
        )
        compute.run()


if __name__ == "__main__":
    main()
