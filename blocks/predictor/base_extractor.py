from abc import abstractmethod
from multiprocessing import Pipe, Process
import contextlib

from datatrove.data import DocumentsPipeline, Media
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.data import MediaType
from datatrove.utils.typeshelper import StatHints
from datatrove.pipeline.extractors.base import ExtractorSandbox
from typing import Optional
from typing import Iterable
from datatrove.data import Document
from datatrove.pipeline.writers.disk_base import DiskWriter

class BaseMediaExtractor(PipelineStep):
    """Base Extractor module. Extractors extract text from html or other non-plain text formats"""

    type = "🛢 - EXTRAC"

    @abstractmethod
    def __init__(self, timeout: float = 60, exclusion_writer: Optional[DiskWriter] = None, exclude_failed: bool = True, keep_original_text: bool = False):
        """

        Args:
            timeout: the timeout for extraction, per document, in seconds
        """
        super().__init__()
        self.timeout = timeout
        self._warned_error = False
        self.keep_original_text = keep_original_text
        self.exclusion_writer = exclusion_writer
        self.exclude_failed = exclude_failed

    @abstractmethod
    def extract(self, media_bytes: bytes, document_metadata: dict) -> tuple[str, dict]:
        """abstract method that actually implements the extraction, e.g. trafilatura.

        Args:
          text: str: non-plain text

        Returns: extracted plain text

        """
        pass

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Iterates through each document in data and calls `timeout_extract` on it.

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)
        """
        with (ExtractorSandbox(timeout=self.timeout, wamup_text=(None, None)) as extractor, self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as exclusion_writer):
            for doc in data:
                self.stat_update(StatHints.total)
                with self.track_time():
                    texts = []
                    for media in doc.media:
                        try:
                            # text, metadata = extractor.process_document((media.media_bytes, media.metadata), self.extract)
                            text, metadata = self.extract(media.media_bytes, media.metadata)
                            if metadata.get("extraction_error"):
                                logger.warning(f"❌ Error {metadata['extraction_error']} while cleaning record text. Skipping record.")
                                media.metadata["extraction_failed"] = metadata["extraction_error"]
                                continue

                            media.metadata = metadata | media.metadata
                            texts.append(text)
                            self.stat_update("extracted")
                        except TimeoutError:
                            self.stat_update("timeout")
                            logger.warning("⏰ Timeout while cleaning record text. Skipping record.")
                            media.metadata["extraction_failed"] = "timeout"
                            continue
                        except EOFError:
                            # Process died unexpectedly
                            self.stat_update("broken_process")
                            logger.warning("Process died unexpectedly, will create new process for next document")
                            media.metadata["extraction_failed"] = "broken_process"
                            continue
                        except Exception as e:
                            self.stat_update("clean_error")
                            if not self._warned_error:
                                logger.warning(
                                    f'❌ Error "{e}" while cleaning record text. Skipping record. '
                                    f"This message will only appear once."
                                )
                                self._warned_error = True
                            # Add this to metadata of the media
                            media.metadata["extraction_failed"] = str(e)
                            continue

                    if not self.keep_original_text:
                        doc.text = "".join(texts)

                if all(media.metadata.get("extraction_failed", False) for media in doc.media):
                    for media in doc.media:
                        logger.warning(f"❌ Media {media.id} extraction failed with error {media.metadata['extraction_failed']}")
                    self.stat_update(StatHints.dropped)
                    if exclusion_writer:
                        exclusion_writer.write(doc, rank=rank, world_size=world_size)

                    if not self.exclude_failed:
                        yield doc
                else:
                    self.stat_update(StatHints.forwarded)
                    for media in doc.media:
                        self.update_media_stats(media)
                    yield doc

