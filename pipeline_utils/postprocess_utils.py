import asyncio
import io
import re
from typing import Any, AsyncGenerator, Iterable

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceSuccess
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.media import iter_pages
from datatrove.utils.typeshelper import StatHints

try:
    import pymupdf
except Exception:
    pymupdf = None

from .extract_utils import render_page_to_base64png_pymupdf


PAGE_TYPE_DETECTOR_PROMPT = """\
### Task
You are a page-type detector.  
Given **one scanned or rendered page image**, decide whether the page contains extractable body text.

### How to decide
• **TEXT** - The page has at least one full line of running text (sentences, paragraphs, bullet points, tables, captions, etc.).  
• **NO_TEXT** - The page is blank **or** contains only non-textual content such as:
- full-page photographs, illustrations, patterns, divider pages, gradients, background art  
- incidental text (page numbers, logos, watermarks)  
- cover/title pages with just a big graphic and no body text

Ignore page numbers, headers/footers, signatures when making your decision.

### Output format
Respond with exactly one of the two strings (no extra words, no punctuation):

TEXT
NO_TEXT\
"""


class AddMetadata(PipelineStep):
    name: str = "🔧 AddMetadata"
    type: str = "🔧 Filter"

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for document in data:
            document.metadata = {**self.kwargs, **document.metadata}
            yield document


class RemoveDoclingMetadata(PipelineStep):
    name: str = "🗑️ RemoveUselessMetadata"
    type: str = "🔧 Filter"

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for document in data:
            if document.media and document.media[0].metadata and "docling_doc_dict" in document.media[0].metadata:
                document.media[0].metadata.pop("docling_doc_dict")
            yield document


class DropFailedDocuments(PipelineStep):
    name: str = "🗂️ DropFailedDocuments"
    type: str = "🔧 Filter"

    def __init__(self, empty_pages_output_dir: str):
        super().__init__()
        self.empty_pages_output_dir = empty_pages_output_dir

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        with JsonlWriter(output_folder=self.empty_pages_output_dir) as writer:
            for document in data:
                if not document.text or not document.media[0].metadata.get("page_offsets"):
                    self.stat_update(StatHints.dropped)
                    writer.write(document, rank=rank, world_size=world_size)
                else:
                    self.stat_update(StatHints.forwarded)
                    yield document


class CoallesceFailedPages(PipelineStep):
    name: str = "CoallesceFailedPages"
    type: str = "Formatter"

    def __init__(self, failed_pages_output_dir: str):
        super().__init__()
        self.failed_pages_output_dir = failed_pages_output_dir

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        corrupted_pages_regex = re.compile(
            r"(^The \d+-\d+ season)|(^The following is a list of the most common types of data breaches and how)"
        )
        failed_pages_regex = re.compile(r"(<--- failed_to_process_page --->)|(<--- stop_reason_.*? --->)", re.DOTALL)
        with JsonlWriter(output_folder=self.failed_pages_output_dir) as writer:
            for document in data:
                pages: list[str] = []
                start = 0
                success_pages = 0
                for offset in document.media[0].metadata.get("page_offsets", []):
                    page_text = document.text[start:offset]
                    start = offset
                    is_corrupted = corrupted_pages_regex.search(page_text) is not None
                    is_failed = failed_pages_regex.search(page_text) is not None
                    if is_failed or is_corrupted:
                        pages.append("<failed_page></failed_page>\n")
                    else:
                        success_pages += 1
                        pages.append(page_text.strip() + "\n")

                if len(pages) == 0 or success_pages == 0:
                    self.stat_update(StatHints.dropped)
                    writer.write(document, rank=rank, world_size=world_size)
                else:
                    self.stat_update(StatHints.forwarded)
                    document.text = "".join(pages)
                    import numpy as np
                    document.media[0].metadata["page_offsets"] = [
                        int(l) for l in np.cumsum([len(page) for page in pages])
                    ]
                    yield document


def prepare_requests_postprocess(media_bytes: bytes | None, document: Document) -> list[tuple[dict[str, Any], int]]:
    from loguru import logger as _logger  # use local logger if available

    if media_bytes is None or pymupdf is None:
        _logger.error("No media bytes or pymupdf missing")
        return []
    try:
        pymupdf_doc = pymupdf.open(None, io.BytesIO(media_bytes))
    except Exception as e:
        _logger.error(f"Error opening PDF: {e}")
        return []

    requests: list[tuple[dict[str, Any], int]] = []
    potential_halucinated_pages = [
        (page_i, pdf_i)
        for page_i, (pdf_i, page_text, page_language) in enumerate(
            zip(
                document.media[0].metadata.get("page_indices", []),
                iter_pages(document),
                document.metadata.get("best_page_languages", []),
            )
        )
        if isinstance(page_text, str) and page_text.startswith("The") and page_language == "eng_Latn"
    ]

    for page_index, pdf_index in potential_halucinated_pages:
        try:
            page = pymupdf_doc[pdf_index]
            image_base64 = render_page_to_base64png_pymupdf(
                page,
                resize_longest_side_pixels=896,
                max_visual_tokens=32 * 32,
            )
            requests.append(
                (
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                                        "max_pixels": 2048 * 28 * 28,
                                        "min_pixels": 56 * 28 * 28,
                                    },
                                    {"type": "text", "text": PAGE_TYPE_DETECTOR_PROMPT},
                                ],
                            }
                        ],
                        "temperature": 0.0,
                        "repetition_penalty": 1.05,
                    },
                    page_index,
                )
            )
        except Exception as e:
            _logger.error(f"Error preparing page request: {e}")

    return requests


async def async_query_builder_postprocess(runner: InferenceRunner, document: Document) -> AsyncGenerator[dict[str, Any], None]:
    if not hasattr(runner, "process_pool"):
        from concurrent.futures import ProcessPoolExecutor
        import atexit

        runner.process_pool = ProcessPoolExecutor(max_workers=4)
        runner.process_pool.__enter__()
        atexit.register(runner.process_pool.__exit__, None, None, None)

    from .postprocess_utils import prepare_requests_postprocess as _prep

    requests_tuple = await asyncio.get_event_loop().run_in_executor(
        runner.process_pool, _prep, document.media[0].media_bytes, document
    )
    request_ids = [i for _, i in requests_tuple]
    document.metadata["request_page_indices"] = request_ids
    for request, _ in requests_tuple:
        yield request


def postprocess_postprocess(document: Document) -> Document:
    page_texts = list(iter_pages(document))
    page_changed = False
    for page_result, request_page_index in zip(
        document.metadata.get("inference_results", []), document.metadata.get("request_page_indices", [])
    ):
        if not isinstance(page_result, InferenceSuccess):
            continue
        if page_result.text != "TEXT":
            page_texts[request_page_index] = None
            page_changed = True
    if "inference_results" in document.metadata:
        del document.metadata["inference_results"]
    if "request_page_indices" in document.metadata:
        del document.metadata["request_page_indices"]

    if page_changed:
        valid_indices = [i for i, page_text in enumerate(page_texts) if page_text is not None]
        valid_page_texts = [page_texts[i] for i in valid_indices]
        import numpy as np

        document.media[0].metadata["page_offsets"] = np.cumsum([len(page_text) for page_text in valid_page_texts]).tolist()
        document.media[0].metadata["page_indices"] = [document.media[0].metadata["page_indices"][i] for i in valid_indices]
        document.text = "".join(valid_page_texts)
    return document








