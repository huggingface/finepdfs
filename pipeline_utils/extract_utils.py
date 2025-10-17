import asyncio
import base64
import io
from typing import Any, AsyncGenerator, Literal, Optional

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import InferenceRunner, InferenceSuccess
import pymupdf



class RepetitionChecker:
    """Detect repeated lines/sentences/characters in a stream of text."""

    def __init__(
        self,
        min_line_repetitions: int = 3,
        min_sentence_repetitions: int = 5,
        min_char_repetition: int = 350,
        check_lines: bool = True,
        check_sentences: bool = True,
        check_chars: bool = True,
        min_sentence_length: int = 50,
        min_line_length: int = 8,
    ):
        self.min_line_repetitions = min_line_repetitions
        self.min_sentence_repetitions = min_sentence_repetitions
        self.min_char_repetition = min_char_repetition
        self.min_sentence_length = min_sentence_length
        self.min_line_length = min_line_length

        self.do_check_lines = check_lines
        self.do_check_sentences = check_sentences
        self.do_check_chars = check_chars

        self._current_line_buffer: list[str] = []
        self._current_sentence_buffer: list[str] = []
        self._last_sentence_content = ""
        self._last_line_content = ""
        self._last_char_content = ""

        self._line_reps = 0
        self._sentence_reps = 0
        self._char_reps = 0

        self._sentence_terminators = {".", "?", "!"}

    def add_char(self, unigram: str) -> Literal["sentence", "line", "char"] | None:
        detected_repetition_type = None

        self._current_line_buffer.append(unigram)
        self._current_sentence_buffer.append(unigram)

        # Sentence check
        if self.do_check_sentences and unigram in self._sentence_terminators:
            sentence_content = "".join(self._current_sentence_buffer)

            if (
                len(sentence_content) >= self.min_sentence_length
                and sentence_content == self._last_sentence_content
            ):
                self._sentence_reps += 1
                if self._sentence_reps >= self.min_sentence_repetitions:
                    detected_repetition_type = "sentence"
            else:
                self._sentence_reps = 0

            self._current_sentence_buffer = []
            self._last_sentence_content = sentence_content

        # Char repetition check
        if self.do_check_chars:
            if unigram == self._last_char_content:
                self._char_reps += 1
                if self._char_reps >= self.min_char_repetition:
                    detected_repetition_type = "char"
            else:
                self._char_reps = 0
            self._last_char_content = unigram

        # Line check
        if (
            self.do_check_lines
            and unigram == "\n"
            and len(self._current_line_buffer) > 2
            and self._current_line_buffer[-2] != "\n"
        ):
            line_content = "".join(self._current_line_buffer)
            if line_content and line_content == self._last_line_content:
                self._line_reps += 1
                if (
                    self._line_reps >= self.min_line_repetitions
                    and ("|" not in line_content or self._line_reps >= 3 * self.min_line_repetitions)
                    and (
                        len(line_content) >= self.min_line_length
                        or self._line_reps >= 4 * self.min_line_repetitions
                    )
                ):
                    detected_repetition_type = "line"
            else:
                self._line_reps = 0

            self._current_line_buffer = []
            self._last_line_content = line_content

        return detected_repetition_type


def render_page_to_base64png_pymupdf(page, resize_longest_side_pixels: Optional[int], max_visual_tokens: int) -> str:
    page.remove_rotation()
    rect = page.rect
    page_width = rect.width
    page_height = rect.height

    if page_width == 0 or page_height == 0:
        zoom = 0
    else:
        IMAGE_FACTOR = 28
        max_total_pixels = max_visual_tokens * IMAGE_FACTOR * IMAGE_FACTOR

        if resize_longest_side_pixels is not None:
            scale_factor = resize_longest_side_pixels / max(page_width, page_height)
            aligned_width = page_width * scale_factor if scale_factor > 1 else page_width
            aligned_height = page_height * scale_factor if scale_factor > 1 else page_height
        else:
            aligned_width = page_width
            aligned_height = page_height

        aligned_width = max(IMAGE_FACTOR, round(aligned_width / IMAGE_FACTOR) * IMAGE_FACTOR)
        aligned_height = max(IMAGE_FACTOR, round(aligned_height / IMAGE_FACTOR) * IMAGE_FACTOR)

        if aligned_width * aligned_height > max_total_pixels:
            scale_factor = (max_total_pixels / (aligned_width * aligned_height)) ** 0.5
            aligned_width = max(IMAGE_FACTOR, int(aligned_width * scale_factor // IMAGE_FACTOR) * IMAGE_FACTOR)
            aligned_height = max(IMAGE_FACTOR, int(aligned_height * scale_factor // IMAGE_FACTOR) * IMAGE_FACTOR)

        zoom_x = aligned_width / page_width
        zoom_y = aligned_height / page_height
        zoom = min(zoom_x, zoom_y)

    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")


def postprocess_extract(document: Document) -> Document | None:
    page_results = document.metadata.get("inference_results", [])
    if len(page_results) == 0:
        return None

    page_dict = {index: "<--- failed_to_process_page --->" for index in range(document.metadata.get("num_pages", 0))}
    for i, page_result in enumerate(page_results):
        if not isinstance(page_result, InferenceSuccess):
            continue

        stop_reason = page_result.finish_reason
        checker = RepetitionChecker()
        for char in page_result.text:
            repetition_type = checker.add_char(char)
            if repetition_type is not None:
                stop_reason = repetition_type
                break

        if stop_reason == "stop":
            content = page_result.text
        else:
            content = f"<--- stop_reason_{stop_reason} --->"
        page_dict[i] = content

    offsets = [0]
    running = 0
    for page in page_dict.values():
        running += len(page)
        offsets.append(running)
    offsets = offsets[1:]
    document.text = "\n".join(page_dict.values())

    extraction_metadata = {
        "page_offsets": offsets,
        "extracted_pages": len(page_results),
        "total_pages": len(page_dict),
    }
    document.media[0].metadata = document.media[0].metadata | extraction_metadata
    if "inference_results" in document.metadata:
        del document.metadata["inference_results"]
    return document


def prepare_requests_extract(media_bytes: bytes | None) -> tuple[list[dict[str, Any]] | None, int]:
    if media_bytes is None:
        return None, 0
    try:
        pymupdf_doc = pymupdf.open(None, io.BytesIO(media_bytes))
    except Exception:
        return None, 0

    requests: list[dict[str, Any]] = []
    for page in pymupdf_doc:
        image_base64 = render_page_to_base64png_pymupdf(
            page,
            resize_longest_side_pixels=1280,
            max_visual_tokens=2048,
        )
        requests.append(
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
                            {
                                "type": "text",
                                "text": "Return the plain text representation of this document as if you were reading it naturally.\n",
                            },
                        ],
                    }
                ],
                "temperature": 0.0,
                "repetition_penalty": 1.05,
            }
        )
    return requests, pymupdf_doc.page_count


async def async_query_builder_extract(runner: InferenceRunner, document: Document) -> AsyncGenerator[dict[str, Any], None]:
    if not hasattr(runner, "process_pool"):
        from concurrent.futures import ProcessPoolExecutor
        import atexit

        runner.process_pool = ProcessPoolExecutor(max_workers=4)
        runner.process_pool.__enter__()
        atexit.register(runner.process_pool.__exit__, None, None, None)

    requests, n_pages = await asyncio.get_event_loop().run_in_executor(
        runner.process_pool, prepare_requests_extract, document.media[0].media_bytes
    )
    document.metadata["num_pages"] = n_pages
    if requests is None:
        return
    for request in requests:
        yield request








