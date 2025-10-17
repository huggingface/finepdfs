import argparse
import os
from typing import Any, AsyncGenerator, Optional
from pipeline_utils.language import SelectBestLanguage

# --- Shared third-party imports used across steps ---
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.media.filters.mime_filter import MimeTypeFilter
from datatrove.pipeline.media.media_readers.warc import WarcReaderFast
from datatrove.pipeline.media.media_readers.zstd import ZstdThreadedReader
from datatrove.pipeline.media.readers.http_fetch import HTTPFetchReader
from datatrove.pipeline.media.media_writers.zstd import BinaryZstdWriter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens.counter import TokensCounter
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceRunner,
    InferenceSuccess,
)
from datatrove.pipeline.dedup.exact_dedup import (
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    ExactDedupConfig,
)
from datatrove.pipeline.dedup import (
    MinhashDedupBuckets,
    MinhashDedupFilter,
    MinhashDedupSignature,
    MinhashConfig,
    MinhashDedupCluster,
)
# Docling branch postprocessing
from postprocessing.page_numbers import PostprocessPageNumbers
from postprocessing.boilerplate import TagBoilerplateFormatter
from postprocessing.language import LanguageTagger
from postprocessing.tables import CleanTables
from postprocessing.normalize import Normalize
from postprocessing.remove_image_annots import RemoveImageAnnotationsByRatio
from datatrove.utils.hashing import HashConfig
# --- Project-specific imports ---
from blocks.extractors.docling import DoclingExtractor
from blocks.predictor.ocr_predictor import PDFScannedPredictor
from blocks.readers.warc_reparse import WarcIndexReprocess
from blocks.utils import MIME_TYPES, index_adapter, filter_non_pdf, filter_non_truncated
from classification.label_utils import AddTextChunks

# Utilities split into separate modules to keep this file short
from pipeline_utils.extract_utils import async_query_builder_extract, postprocess_extract
from pipeline_utils.postprocess_utils import (
    AddMetadata,
    RemoveDoclingMetadata,
    DropFailedDocuments,
    CoallesceFailedPages,
    async_query_builder_postprocess,
    postprocess_postprocess,
)
from pipeline_utils.push_utils import push_adapter

# =====================================================================================
# Constants aggregated from original scripts
# =====================================================================================

CC_INDEX_INPUT_TEMPLATE = "s3://commoncrawl/cc-index/table/cc-main/warc/crawl={crawl_id}/subset=warc"
CC_PATHS_TEMPLATE = "s3://commoncrawl/crawl-data/{crawl_id}/warc.paths.gz"
SPLIT_TRUNCATION_DIR = "./finepdfs/data/split_truncation/{prefix}"
PDF_SAVE_DIR = "./finepdfs/data/pdf"

BYTE_CONTENT_DEDUPLICATION_DIR_OUTPUT = "./finepdfs/data/byte_content_deduplication/{prefix}"
OCR_CLASSIFICATION_DIR_OUTPUT = "./finepdfs/data/ocr_classification/{prefix}"
DEDUP_OUTPUT_DIR = "./finepdfs/data/content_dedup/{prefix}"
PDF_SCANNED_MODEL_PATH = "./models/xgb_ocr_classifier/xgb_classifier.ubj"

INPUT_DIR_EXTRACT = "./finepdfs/data/content_dedup/{prefix}"
OUTPUT_OCR_DIR = "./finepdfs/data/ocr_docs_extracted/{prefix}"
OUTPUT_NON_OCR_DIR = "./finepdfs/data/non_ocr_docs_extracted/{prefix}"

DOCLING_INPUT_DIR = "./finepdfs/data/non_ocr_docs_extracted"
OCR_INPUT_DIR = "./finepdfs/data/ocr_docs_extracted"
SAVE_DOCLING_DIR = "./finepdfs/data/postprocessed/output_docling"
SAVE_OCR_DIR = "./finepdfs/data/postprocessed/output_ocr"
FAILED_PAGES_OCR_DIR = "./finepdfs/data/postprocessed/ocr_failed"
EMPTY_PAGES_DOCLING_DIR = "./finepdfs/data/postprocessed/empty"

TH_VALUES_FILE = "./thresholds/th_values.json"
PER_LANGUAGE_DIR_EXACT = "./finepdfs/data/glotlid/per_language"
OUTPUT_DIR_EXACT = "./finepdfs/data/exact_dedup/per_language"

INPUT_DIR_MODEL = "./finepdfs/data/exact_dedup/per_language/output"
OUTPUT_DIR_MODEL = "./finepdfs/data/model_labeling/per_language"

PER_LANGUAGE_DIR_MINHASH = "./finepdfs/data/model_labeling/per_language"
OUTPUT_DIR_MINHASH = "./finepdfs/data/minhash/per_language"

PUSH_INPUT_DIR = "./finepdfs/data/minhash/per_language/"

# Adjust as needed, low for demo purposes (for real run we didn't limit)
LIMIT = 10000

# =====================================================================================
# Step 1: filter_pdfs_and_refetch
# =====================================================================================

def run_filter_pdfs_and_refetch(crawl_ids: list[str]):
    for crawl_id in crawl_ids:
        if crawl_id < "CC-MAIN-2019-47":
            index_reader = WarcIndexReprocess(
                data_folder=f"s3://commoncrawl",
                limit=LIMIT,
                paths_file=CC_PATHS_TEMPLATE.format(crawl_id=crawl_id),
            )
        else:
            index_reader = ParquetReader(
                data_folder=CC_INDEX_INPUT_TEMPLATE.format(crawl_id=crawl_id),
                glob_pattern="*.parquet",
                doc_progress=True,
                adapter=index_adapter,
                limit=LIMIT,
            )

        pipeline = [
            index_reader,
            LambdaFilter(filter_non_pdf),
            LambdaFilter(
                filter_non_truncated,
                exclusion_writer=JsonlWriter(
                    output_folder=SPLIT_TRUNCATION_DIR.format(prefix="non_truncated"),
                ),
            ),
            # For production purposes, it's not wise to to run this in one pipeline for good resource allocation.
            # We recommend separting the httpfetch reader to a separate pipeline to maximize resource utilization.
            HTTPFetchReader(workers=15, max_retries=3, timeout=(60, 60), download_timeout=60 * 10),
            MimeTypeFilter(mime_types=MIME_TYPES["pdf"]),
            BinaryZstdWriter(
                max_file_size=100 * 1024 * 1024 * 1024,
                output_folder=PDF_SAVE_DIR,
                output_filename=f"{crawl_id.replace('-', '_')}_${{rank}}.zstd",
            ),
            LambdaFilter(
                lambda x: x.media[0].media_bytes is not None,
                exclusion_writer=JsonlWriter(
                    output_folder=SPLIT_TRUNCATION_DIR.format(prefix="failed_pdf_fetch"),
                ),
            ),
            JsonlWriter(
                output_folder=SPLIT_TRUNCATION_DIR.format(prefix="truncated"),
            ),
        ]

        LocalPipelineExecutor(pipeline).run()


# =====================================================================================
# Step 2: content_dedup_ocr_organize
# =====================================================================================

def _get_media_bytes(doc: Document) -> bytes:
    return doc.media[0].media_bytes if doc.media[0].media_bytes else b""


CONTENT_DEDUP_CONFIG = ExactDedupConfig(content_getter=_get_media_bytes)


def _filter_ocr(x: Document):
    meta = x.media[0].metadata or {}
    # See the training notebook why we decided for this threshold
    return not (
        meta.get("ocr_prob", 0) >= 0.2 or meta.get("garbled_text_ratio", 0) > 0.0
    )


def run_content_dedup_ocr_organize():
    for truncation in ["truncated", "non_truncated"]:
        if truncation == "non_truncated":
            reader = WarcReaderFast(
                data_folder="s3://commoncrawl",
                preserve_order=True,
                workers=5,
            )
        else:
            reader = ZstdThreadedReader(
                data_folder=PDF_SAVE_DIR,
                workers=2,
                preserve_order=True,
            )

        # 2.1 Signatures
        pipeline1 = [
            JsonlReader(
                data_folder=SPLIT_TRUNCATION_DIR.format(prefix=truncation),
                glob_pattern="**/*.jsonl.gz",
                doc_progress=True,
            ),
            reader,
            ExactDedupSignature(
                config=CONTENT_DEDUP_CONFIG,
                output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/sigs"),
                finder_workers=100,
            ),
        ]

        # 2.2 Find duplicates
        pipeline2 = [
            ExactFindDedups(
                config=CONTENT_DEDUP_CONFIG,
                data_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/sigs"),
                output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/dups"),
            )
        ]

        # 2.3 Filter dups, OCR predicate, split
        pipeline3 = [
            JsonlReader(
                data_folder=SPLIT_TRUNCATION_DIR.format(prefix=truncation),
                glob_pattern="**/*.jsonl.gz",
                doc_progress=True,
            ),
            ExactDedupFilter(
                config=CONTENT_DEDUP_CONFIG,
                data_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/dups"),
                exclusion_writer=JsonlWriter(
                    output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/removed")
                ),
            ),
            reader,
            PDFScannedPredictor(
                path_to_model=PDF_SCANNED_MODEL_PATH,
                exclusion_writer=JsonlWriter(
                    output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/failed_ocr")
                ),
                exclude_failed=True,
            ),
            LambdaFilter(_filter_ocr, exclusion_writer=JsonlWriter(output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/ocr"))),
            JsonlWriter(output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/non_ocr")),
        ]

        LocalPipelineExecutor(pipeline1).run()
        LocalPipelineExecutor(pipeline2, tasks=100, workers=1).run()
        LocalPipelineExecutor(pipeline3).run()


# =====================================================================================
# Step 3: extract
# =====================================================================================

def run_extract(gpus: int=1):
    # Non-OCR extraction (Docling)
    # Use can play around with docling imports and remove torch deps as our quant needs just open vino
    # This allowed us to run the pipeline on 1 cpu with 2GB or RAM. We highly recommend AVX512 support for optimal performance.
    for truncation in ["truncated", "non_truncated"]:
        if truncation == "non_truncated":
            reader = WarcReaderFast(
                data_folder="s3://commoncrawl",
                preserve_order=True,
                workers=5,
            )
        else:
            reader = ZstdThreadedReader(
                data_folder=PDF_SAVE_DIR,
                workers=2,
                preserve_order=True,
            )
        pipeline_docling = [
            JsonlReader(
                data_folder=INPUT_DIR_EXTRACT.format(prefix=f"{truncation}/non_ocr"),
                glob_pattern="**/*.jsonl.gz",
                doc_progress=True,
            ),
            reader,
            DoclingExtractor(
                timeout=10 * 60,
                exclusion_writer=JsonlWriter(
                    output_folder=OUTPUT_NON_OCR_DIR.format(prefix=f"{truncation}/failed")
                ),
            ),
            JsonlWriter(output_folder=OUTPUT_NON_OCR_DIR.format(prefix=f"{truncation}/extracted")),
        ]
        LocalPipelineExecutor(pipeline_docling).run()

    # OCR extraction via InferenceRunner
    # For production environment, this is too slow as you should asynchronously fetch PDFs from bucket inside the query preparation step.
    # We use synchronous fetching here for simplicity.
    # On h100 you should be able to see ~5 pages/s per worker
    for truncation in ["truncated", "non_truncated"]:
        if truncation == "non_truncated":
            reader = WarcReaderFast(
                data_folder="s3://commoncrawl",
                preserve_order=True,
                workers=5,
            )
        else:
            reader = ZstdThreadedReader(
                data_folder=PDF_SAVE_DIR,
                workers=2,
                preserve_order=True,
            )

        runner = InferenceRunner(
            config=InferenceConfig(
                model_name_or_path="reducto/RolmOCR",
                temperature=0.0,
                max_concurrent_tasks=50 if truncation == "truncated" else 300,
                server_type="vllm",
                metric_interval=100,
                dp=gpus,
            ),
            query_builder=async_query_builder_extract,
            postprocess_fn=postprocess_extract,
            output_writer=JsonlWriter(output_folder=OUTPUT_OCR_DIR.format(prefix=f"{truncation}/extracted")),
        )

        pipeline = [
            JsonlReader(
                data_folder=INPUT_DIR_EXTRACT.format(prefix=f"{truncation}/ocr"),
                glob_pattern="**/*.jsonl.gz",
                doc_progress=True,
            ),
            reader,
            runner,
        ]
        LocalPipelineExecutor(pipeline).run()


# =====================================================================================
# Step 4: postprocess
# =====================================================================================

def run_postprocess():
    language_tagger = LanguageTagger(language_threshold=0.01, label_only=True, backend="glotlid")
    # OCR branch postprocessing
    for truncation in ["truncated", "non_truncated"]:
        if truncation == "non_truncated":
            reader = WarcReaderFast(
                data_folder="s3://commoncrawl",
                preserve_order=True,
                workers=5,
            )
        else:
            reader = ZstdThreadedReader(
                data_folder=PDF_SAVE_DIR,
                workers=2,
                preserve_order=True,
            )

        pipeline_ocr = [
            JsonlReader(
                data_folder=OCR_INPUT_DIR,
                glob_pattern=f"{truncation}/extracted/*.jsonl.gz",
            ),
            reader,
            AddMetadata(is_docling=False, is_truncated=truncation == "truncated"),
            DropFailedDocuments(EMPTY_PAGES_DOCLING_DIR),
            CoallesceFailedPages(FAILED_PAGES_OCR_DIR),
            TagBoilerplateFormatter(is_ocr=True, drop=True),
            Normalize(is_from_docling=False),
            language_tagger,
            TokensCounter(batch_size=1000, tokenizer_name_or_path="hynky/Llama-3.2-1B-no-bos"),
            # Removes hallucinations caused by blank pages
            # This is very expensive the way we are doing so, better option
            # would be to finetuned ViT model to classify blank pages, however
            # this we didn't have time for it.
            InferenceRunner(
                config=InferenceConfig(
                    model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
                    temperature=0.0,
                    max_concurrent_tasks=50,
                    server_type="vllm",
                    metric_interval=100,
                ),
                query_builder=async_query_builder_postprocess,
                postprocess_fn=postprocess_postprocess,
                output_writer=JsonlWriter(
                    output_folder=SAVE_OCR_DIR.format(prefix=f"extracted")
                ),
            ),
        ]
        LocalPipelineExecutor(pipeline_ocr).run()



    pipeline_docling = [
        JsonlReader(data_folder=DOCLING_INPUT_DIR, glob_pattern="non_truncated/extracted/*.jsonl.gz"),
        AddMetadata(is_docling=True, is_truncated=False),
        JsonlReader(data_folder=DOCLING_INPUT_DIR, glob_pattern="truncated/extracted/*.jsonl.gz"),
        AddMetadata(is_docling=True, is_truncated=True),
        RemoveDoclingMetadata(),
        DropFailedDocuments(EMPTY_PAGES_DOCLING_DIR),
        PostprocessPageNumbers(),
        CleanTables(),
        RemoveImageAnnotationsByRatio(ratio_threshold=0.8),
        TagBoilerplateFormatter(is_ocr=False, drop=True),
        Normalize(is_from_docling=True),
        language_tagger,
        TokensCounter(batch_size=1000, tokenizer_name_or_path="hynky/Llama-3.2-1B-no-bos"),
        JsonlWriter(output_folder=SAVE_DOCLING_DIR),
    ]
    LocalPipelineExecutor(pipeline_docling).run()

# =====================================================================================
# Language filter (glotlid) -> splits into per-language shards
# =====================================================================================



def run_language_filter(languages: Optional[list[str]] = None):
    import json

    with open(TH_VALUES_FILE, "r") as f:
        language_thresholds_dict = {k: max(float(v), 0.05) for k, v in json.load(f).items()}

    # Always allow zxx_* to pass if they are top-1,
    # Tus zxx is never rerouted
    language_thresholds_dict["zxx_Latn"] = -1
    language_thresholds_dict["zxx_Zzzz"] = -1
    language_thresholds_dict["zxx_Arab"] = -1

    pipeline: list[PipelineStep] = [
        JsonlReader(
            data_folder=SAVE_DOCLING_DIR,
            glob_pattern="*.jsonl.gz",
        ),
        JsonlReader(
            data_folder=SAVE_OCR_DIR,
            glob_pattern="*.jsonl.gz",
        ),
        SelectBestLanguage(language_thresholds_dict=language_thresholds_dict),
    ]

    # Optionally restrict to selected languages
    if languages:
        allowed = set(languages)

        def _keep_selected(doc: Document) -> bool:
            return doc.metadata.get("language_bucket") in allowed

        pipeline.append(LambdaFilter(_keep_selected))

    pipeline.append(
        JsonlWriter(
            output_folder=PER_LANGUAGE_DIR_EXACT,
            output_filename="${language_bucket}/${rank}.jsonl.gz",
        )
    )

    LocalPipelineExecutor(pipeline).run()

# =====================================================================================
# Step 5: exact_dedup
# =====================================================================================

def create_content_getter_exact():
    import re as _re

    remove_spaces_regex = _re.compile(r"\s+")

    def content_getter(doc: Document) -> str:
        return remove_spaces_regex.sub("", doc.text)

    return content_getter


EXACT_CONFIG = ExactDedupConfig(content_getter=create_content_getter_exact())


def run_exact_dedup(languages: Optional[list[str]] = None, tasks: int = 100):
    if not languages:
        languages = [
            lang
            for lang in get_datafolder(PER_LANGUAGE_DIR_EXACT).list_files(
                include_directories=True, recursive=False
            )
            if lang
        ]

    for language in languages:
        worker_tasks = max(tasks // 2, 1)

        pipeline1 = [
            JsonlReader(data_folder=f"{PER_LANGUAGE_DIR_EXACT}/{language}"),
            ExactDedupSignature(
                config=EXACT_CONFIG,
                output_folder=f"{OUTPUT_DIR_EXACT}/sigs/{language}",
                finder_workers=worker_tasks,
            ),
        ]
        pipeline2 = [
            ExactFindDedups(
                config=EXACT_CONFIG,
                data_folder=f"{OUTPUT_DIR_EXACT}/sigs/{language}",
                output_folder=f"{OUTPUT_DIR_EXACT}/dups/{language}",
            )
        ]
        output_writer = JsonlWriter(output_folder=f"{OUTPUT_DIR_EXACT}/output/{language}")
        exclude_writer = JsonlWriter(output_folder=f"{OUTPUT_DIR_EXACT}/removed/{language}")
        pipeline3 = [
            JsonlReader(data_folder=f"{PER_LANGUAGE_DIR_EXACT}/{language}"),
            ExactDedupFilter(
                config=EXACT_CONFIG,
                data_folder=f"{OUTPUT_DIR_EXACT}/dups/{language}",
                exclusion_writer=exclude_writer,
            ),
            # Adds chunk for model classification
            AddTextChunks(tokenizer_name="answerdotai/ModernBERT-large" if language == "eng_Latn" else "mmbert-colab/mmBERT-base"),
            output_writer,
        ]

        LocalPipelineExecutor(pipeline1).run()
        LocalPipelineExecutor(pipeline2, tasks=worker_tasks).run()
        LocalPipelineExecutor(pipeline3).run()


# =====================================================================================
# Step 6: model_classification
# =====================================================================================

def model_exists(repo_id: str) -> bool:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.model_info(repo_id)
        return True
    except Exception:
        return False


async def async_query_builder_model(runner: InferenceRunner, document: Document) -> AsyncGenerator[dict[str, Any], None]:
    for chunk in document.metadata["chunks"]:
        yield {"input": chunk}


def make_postprocess_fn_model(output_fields_in_order: list[str]):
    def postprocess(document: Document):
        results = document.metadata.get("inference_results", [])
        per_field_scores: dict[str, list[float | None]] = {field: [] for field in output_fields_in_order}
        for chunk_result in results:
            if isinstance(chunk_result, InferenceSuccess):
                try:
                    values = [float(part.strip()) for part in chunk_result.text.split(",") if part.strip() != ""]
                except Exception:
                    values = []
            else:
                values = []
            for idx, field in enumerate(output_fields_in_order):
                value = values[idx] if idx < len(values) else None
                per_field_scores[field].append(value)
        for field, series in per_field_scores.items():
            document.metadata[field] = series
        document.metadata.pop("inference_results", None)
        document.metadata.pop("chunks", None)
        return document

    return postprocess


def run_model_classification(languages: Optional[list[str]] = None, gpus: int = 1):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if not languages:
        languages = [
            lang
            for lang in get_datafolder(INPUT_DIR_MODEL).list_files(
                include_directories=True, recursive=False
            )
            if lang
        ]

    for language in languages:
        edu_model = f"HuggingFaceFW/finepdfs_edu_classifier_{language}"
        dclm_model = f"HuggingFaceFW/finepdfs_dclm_classifier_{language}"
        present_models: list[str] = []
        output_fields_in_order: list[str] = []
        if model_exists(edu_model):
            present_models.append(edu_model)
            output_fields_in_order.append("fw_edu_scores")
        if model_exists(dclm_model):
            present_models.append(dclm_model)
            output_fields_in_order.append("dclm_scores")

        data_folder = f"{INPUT_DIR_MODEL}/{language}"
        output_folder = f"{OUTPUT_DIR_MODEL}/{language}"

        if not present_models:
            pipeline = [
                JsonlReader(data_folder=data_folder, glob_pattern="*.jsonl.gz"),
                JsonlWriter(output_folder=output_folder),
            ]
            LocalPipelineExecutor(pipeline=pipeline).run()
            continue

        model_kwargs = {
            "server_script": f"{CURRENT_DIR}/blocks/classification/tf_batching.py",
            "batch-size": 256,
            "batch-timeout": 10.0,
            "max-context": 2048,
            "model-name-or-path": ";".join(present_models),
            "host": "0.0.0.0",
        }

        config = InferenceConfig(
            server_type="custom",
            model_name_or_path=";".join(present_models),
            max_concurrent_requests=1024,
            max_concurrent_tasks=2048,
            model_kwargs=model_kwargs,
            dp=gpus,
            use_chat=False,
            server_log_folder="./server_logs",
        )

        pipeline = [
            JsonlReader(data_folder=data_folder, glob_pattern="*.jsonl.gz"),
            InferenceRunner(
                query_builder=async_query_builder_model,
                config=config,
                output_writer=JsonlWriter(output_folder=output_folder),
                postprocess_fn=make_postprocess_fn_model(output_fields_in_order),
            ),
        ]
        LocalPipelineExecutor(pipeline=pipeline).run()


# =====================================================================================
# Step 7: minhash
# =====================================================================================

def create_content_getter_minhash():
    import re

    def content_getter(doc: Document) -> str:
        max_len = 1_000_000
        text = doc.text
        if len(text) > max_len:
            # Find the next whitespace (any \s) after max_len using regex
            match = re.search(r"\s", text[max_len:])
            if match is None:
                # No whitespace found, just truncate at max_len
                text = text[:max_len]
            else:
                text = text[:max_len + match.start()]
        return text

    return content_getter


MINHASH_CONFIG = MinhashConfig(
    hash_config=HashConfig(hash_fc="xxhash", precision=64),
    num_buckets=32,
    hashes_per_bucket=10,
    n_grams=5,
)


def run_minhash(languages: Optional[list[str]] = None):
    if not languages:
        languages = [
            lang
            for lang in get_datafolder(PER_LANGUAGE_DIR_MINHASH).list_files(
                include_directories=True, recursive=False
            )
            if lang
        ]

    for language in languages:
        tasks = len(get_datafolder(f"{PER_LANGUAGE_DIR_MINHASH}/{language}").list_files(glob_pattern="*.jsonl.gz"))
        WORKERS = max(tasks // 2, MINHASH_CONFIG.num_buckets) // MINHASH_CONFIG.num_buckets * MINHASH_CONFIG.num_buckets

        output_folder = f"{OUTPUT_DIR_MINHASH}/{language}"
        if language == "eng_Latn":
            input_block = [JsonlReader(data_folder=f"{PER_LANGUAGE_DIR_MINHASH}/{language}"), LambdaFilter(lambda x: max(x.metadata.get("fw_edu_scores", [0])) >= 0.5)]
        else:
            input_block = [JsonlReader(data_folder=f"{PER_LANGUAGE_DIR_MINHASH}/{language}")]

        # Tokenizers map as some languages are too slow/don't have a tokenizer
        tokenizers_map = {
            "lat_Latn": "ita_Latn",
            "kaz_Cyrl": "rus_Cyrl",
            "cym_Latn": "eng_Latn",
            "glg_Latn": "por_Latn",
            "und_Hyra": "jpn_Jpan",
            "unknown": "eng_Latn",
        }

        tokenizer_name = tokenizers_map.get(language, language)

        pipeline1 = [
            *input_block,
            MinhashDedupSignature(output_folder=f"{output_folder}/signatures", config=MINHASH_CONFIG, language=tokenizer_name),
        ]
        pipeline2 = [
            MinhashDedupBuckets(
                input_folder=f"{output_folder}/signatures",
                output_folder=f"{output_folder}/buckets",
                config=MINHASH_CONFIG,
                lines_to_buffer=20000,
            ),
        ]
        pipeline3 = [
            MinhashDedupCluster(
                input_folder=f"{output_folder}/buckets",
                output_folder=f"{output_folder}/clusters",
                config=MINHASH_CONFIG,
                save_cluster_id=True,
                save_cluster_size=True,
            ),
        ]
        pipeline4 = [
            *input_block,
            MinhashDedupFilter(
                input_folder=f"{output_folder}/clusters",
                exclusion_writer=JsonlWriter(output_folder=f"{output_folder}/removed"),
                load_cluster_ids=True,
                load_cluster_sizes=True,
            ),
            JsonlWriter(output_folder=f"{output_folder}/output"),
        ]

        LocalPipelineExecutor(pipeline1, tasks=tasks).run()
        LocalPipelineExecutor(pipeline2, tasks=WORKERS).run()
        LocalPipelineExecutor(pipeline3, tasks=1).run()
        LocalPipelineExecutor(pipeline4, tasks=tasks).run()


def run_push_to_hub(languages: Optional[list[str]] = None):
    if not languages:
        languages = [
            lang
            for lang in get_datafolder(PUSH_INPUT_DIR).list_files(
                include_directories=True, recursive=False
            )
            if lang
        ]

    for lang in languages:
        pipeline = [
            JsonlReader(
                data_folder=f"{PUSH_INPUT_DIR}/{lang}/output",
                glob_pattern=f"*.jsonl.gz",
            ),
            HuggingFaceDatasetWriter(
                dataset="HuggingFaceFW/finepdfs_subset",
                local_working_dir=f"./output_{lang}/${{rank}}",
                output_filename=f"data/{lang}/train/${{rank}}.parquet",
                adapter=push_adapter,
            ),
            LambdaFilter(lambda x: max(x.metadata.get("fw_edu_scores", [0])) >= 0.5),
            HuggingFaceDatasetWriter(
                dataset="HuggingFaceFW/finepdfs_fw_edu_subset",
                local_working_dir=f"./output_edu_{lang}/${{rank}}",
                output_filename=f"data/{lang}/train/${{rank}}.parquet",
                adapter=push_adapter,
            ),
        ]
        LocalPipelineExecutor(pipeline=pipeline).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinePDFs pipeline sequentially")
    parser.add_argument("--crawl-ids", type=str, required=True, help="Comma-separated CommonCrawl crawl IDs for step 1")
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated list of languages for steps 5-8")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for model classification (step 6)")
    return parser.parse_args()


def main():
    args = parse_args()
    languages = [s.strip() for s in args.languages.split(",")] if args.languages else None
    crawl_ids = [s.strip() for s in args.crawl_ids.split(",")] if args.crawl_ids else []

    # Sequentially run all steps
    run_filter_pdfs_and_refetch(crawl_ids)
    run_content_dedup_ocr_organize()
    run_extract(args.gpus)
    run_postprocess()
    run_language_filter(languages=languages)
    run_exact_dedup(languages=languages)
    run_model_classification(languages=languages, gpus=args.gpus)
    run_minhash(languages=languages)
    run_push_to_hub(languages=languages)


if __name__ == "__main__":
    main()


