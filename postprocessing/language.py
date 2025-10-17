from collections import defaultdict
from typing import Literal, Iterable
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from typing import Callable

class LanguageTagger(PipelineStep):
    name = "🌍 Per-page Language ID"
    type = "Filter"

    _requires_dependencies = [("fasttext", "fasttext-numpy2-wheel"), "fasteners"]

    def __init__(
        self,
        languages: list[str] | str | None = None,
        language_threshold: float = 0.65,
        backend: Literal["ft176", "glotlid"] = "ft176",
        label_only: bool = False,
        keep_top_pairs_threshold: float = -1,
        tags_function: Callable | None = None,
        min_alpha_length_bytes: int = 50,
        min_alpha_ratio: float = 0.2,
    ):
        """
        Tags documents with language information using two modes:
        1) Per-page language detection with averaging
        2) Full document language detection

        Args:
            languages: list of languages to keep. None for all
            language_threshold: language_threshold minimum score to accept a document
            keep_top_pairs_threshold: keep a list of all language pairs with at least this score. -1 to disable
            backend: language detection backend to use
        """
        self.language_threshold = language_threshold
        self.tags_function = tags_function
        if isinstance(languages, str):
            languages = [languages]
        self.languages = languages
        self.backend = backend
        from datatrove.utils.lid import FT176LID, GlotLID
        self.model = FT176LID(languages, k=1000) if backend == "ft176" else GlotLID(languages, k=1000)
        self.label_only = label_only
        self.keep_top_pairs_threshold = keep_top_pairs_threshold
        self.min_alpha_length_bytes = min_alpha_length_bytes
        self.min_alpha_ratio = min_alpha_ratio
        super().__init__()

    def _get_alpha_length(self, text: str) -> int:
        return len("".join([ch for ch in text if ch.isalpha()]).encode("utf-8"))

    def _alpha_ratio(self, text: str) -> float:
        alpha_length = self._get_alpha_length(text)
        all_non_space_chars = len(text.replace(" ", "").replace("\n", "").encode("utf-8"))
        if all_non_space_chars == 0:
            return 0.0
        return alpha_length / all_non_space_chars


    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        import re
        table_pattern = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
        """Process documents and add language tagging information"""
        for doc in data:
            with self.track_time():
                page_offsets = doc.media[0].metadata["page_offsets"]  # type: ignore
                # Collect all pages
                pages = []
                start = 0
                for offset in page_offsets:
                    page_text = doc.text[start:offset]
                    # Remove tables
                    page_text = table_pattern.sub("", page_text, re.MULTILINE)
                    page_text = page_text.replace("|", "").replace("-", "").replace("*", "").replace("#", "")
                    # Replace all multiple spaces with single space
                    page_text = re.sub(r"\s+", " ", page_text)
                    start = offset
                    pages.append(page_text)
                

                # Compute it per page
                page_language_scores = defaultdict(float)
                page_top_languages: list[tuple[str, float]] = []  # [(top_lang, top_score)]
                
                for page_text in pages:
                    alpha_length = self._get_alpha_length(page_text)
                    alpha_ratio = self._alpha_ratio(page_text)
                    if alpha_length < self.min_alpha_length_bytes or alpha_ratio < self.min_alpha_ratio:
                        page_best_lang_pair, page_lang_pairs = ("unknown", 0), {}
                    else:
                        page_best_lang_pair, page_lang_pairs = self.model.predict(Document(text=page_text, id="0"))

                    for lang_key, score in page_lang_pairs.items():
                        page_language_scores[lang_key] += score
                    page_top_languages.append(page_best_lang_pair)

                # Normalize language scores
                for lang_key, score in page_language_scores.items():
                    page_language_scores[lang_key] = score / len(pages)

                # Handle edge case where no language scores exist
                if page_language_scores:
                    best_lang_pair = max(page_language_scores.items(), key=lambda x: x[1])
                else:
                    best_lang_pair = ("unknown", 0.0)

                # Add metadata
                doc.metadata["best_page_scores"] = [float(score) for _, score in page_top_languages]
                doc.metadata["best_page_languages"] = [lang for lang, _ in page_top_languages]
                doc.metadata["best_page_average_language"] = best_lang_pair[0]
                doc.metadata["best_page_average_score"] = float(best_lang_pair[1])
                for lang, score in {lang: score for lang, score in page_language_scores.items() if score > self.language_threshold}.items():
                    doc.metadata[f"page_average_language_{lang}_score"] = float(score)

                # Mode 2: Full document language detection
                full_doc = "".join(pages)[:40000]

                if self._get_alpha_length(full_doc) < self.min_alpha_length_bytes or self._alpha_ratio(full_doc) < self.min_alpha_ratio:
                    doc.metadata["language"] = "unknown"
                    doc.metadata["language_score"] = 0.0
                    doc.metadata["top_language_unknown_score"] = 0.0
                else:
                    # For each page keep all language pairs where score is above threshold
                    best_lang_pair, lang_pairs = self.model.predict(Document(text=full_doc, id="0"))
                    doc.metadata["language"] = best_lang_pair[0]
                    doc.metadata["language_score"] = float(best_lang_pair[1])
                    lang_pars = {
                        lang: score for lang, score in lang_pairs.items() if score > self.language_threshold
                    }
                    for lang, score in lang_pars.items():
                        doc.metadata[f"top_language_{lang}_score"] = float(score)
            yield doc
