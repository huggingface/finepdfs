from typing import Iterable
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep


class SelectBestLanguage(PipelineStep):
    name: str = "SelectBestLanguage"
    type: str = "Filter"

    def __init__(self, language_thresholds_dict: dict[str, float]):
        self.language_thresholds_dict = language_thresholds_dict
        super().__init__()

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            available_languages = sorted(
                {
                    k[len("page_average_language_") : -len("_score")]: float(v)
                    for k, v in document.metadata.items()
                    if k.startswith("page_average_language_")
                }.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            # choose the first language above its threshold
            for i, (language, score) in enumerate(available_languages):
                # don't redirect from zxx_* if it's the top detected
                if i == 0 and language.startswith("zxx_"):
                    document.metadata["language_bucket"] = language
                    break

                if score > self.language_thresholds_dict.get(language, 10000):
                    document.metadata["language_bucket"] = language
                    break
            else:
                # fallback to highest available or "unknown"
                highest_lang = list(available_languages)[0][0] if available_languages else "unknown"
                if highest_lang in self.language_thresholds_dict:
                    document.metadata["language_bucket"] = f"{highest_lang}_removed"
                else:
                    document.metadata["language_bucket"] = highest_lang

            yield document