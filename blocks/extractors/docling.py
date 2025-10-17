from pathlib import Path
from datatrove.data import Document, Media
from blocks.predictor.base_extractor import BaseMediaExtractor
import io
from loguru import logger
import numpy as np
import warnings
import logging
from dataclasses import dataclass
from typing import Optional
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from blocks.loggers import LoggerStream
from docling_code.custom_code.text_doc_serializer import TextDocSerializer
from datatrove.pipeline.writers.disk_base import DiskWriter
from docling_code.custom_code.postprocessing.cells_merge import CellsMerger
from docling_code.custom_code.postprocessing.para_merge import ParagraphMerger
from docling_code.custom_code.postprocessing.list_marker_normalizer import ListItemMarkerProcessor
from docling_code.custom_code.postprocessing.page_num_tagger import PageNumberRemover
from docling_code.custom_code.backend.pymupdf_backend import PyMuPdfDocumentBackend

@dataclass(frozen=True)
class DoclingPostProcessingOptions:
    use_markdown: bool = False
    use_picture: bool = False
    use_file_path: bool = False
    fix_lists: bool = True
    fix_paragraphs: bool = True
    fix_reading_order: bool = True
    fix_page_numbers: bool = True

class DoclingExtractor(BaseMediaExtractor):
    def __init__(self, timeout: int = 10*60, post_processing_options: DoclingPostProcessingOptions = DoclingPostProcessingOptions(), exclusion_writer: Optional[DiskWriter] = None):
        docling_timeout = timeout - 2 if timeout > 2 else None
        pipeline_options = PdfPipelineOptions(
            do_table_structure=False,
            do_ocr=False,
            document_timeout=docling_timeout,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=1)
        )
        self.logger_stream = LoggerStream([logging.getLogger("docling"), logging.getLogger("pymupdf")])
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyMuPdfDocumentBackend)
            }
        )
        self.use_picture = post_processing_options.use_picture
        self.use_markdown = post_processing_options.use_markdown
        self.reading_order_fixer = CellsMerger()
        self.paragraph_fixer = ParagraphMerger()
        self.list_item_marker_processor = ListItemMarkerProcessor()
        self.page_number_remover = PageNumberRemover()
        super().__init__(timeout=timeout, exclusion_writer=exclusion_writer)


    # def extract(self, path: str | None) -> tuple[str, dict]:
    def extract(self, media_bytes: bytes | None, document_metadata: dict) -> tuple[str, dict]:
        if media_bytes is None:
            return "", {
                "extraction_error": "Media bytes are None"
            }

        from docling.datamodel.settings import settings
        from docling_core.types.io import DocumentStream
        from docling.datamodel.base_models import ConversionStatus
        from docling_core.transforms.serializer.markdown import MarkdownParams
        with self.logger_stream as log_output:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                with io.BytesIO(media_bytes) as stream:
                    document_stream = DocumentStream(name="test.pdf", stream=stream)
                    converted = self.doc_converter.convert(document_stream, raises_on_error=True)
                logger_content = log_output.value()

        document_initial = converted.document.export_to_dict()
        document_postprocessed = self.reading_order_fixer.process_document(converted.document, allow_multi_prov=True)
        document_postprocessed = self.paragraph_fixer.process_document(document_postprocessed, allow_multi_prov=True)
        document_postprocessed = self.list_item_marker_processor.process_document(document_postprocessed)
        document_postprocessed = self.page_number_remover.process_document(document_postprocessed)
        page_break_placeholder = "<--- page break --->"

        serializer = TextDocSerializer(doc=document_postprocessed,
                                        params=MarkdownParams(
                                        page_break_placeholder=page_break_placeholder,
                                        image_placeholder="<docling_image></docling_image>",
                                        escape_underscores=False,
                                        escape_html=False,
                        ))


        full_text = serializer.serialize().text
        page_list = full_text.split(page_break_placeholder)
        # Remove empty strings
        metadata = {
            "num_pages": len(page_list),
            "page_offsets": np.cumsum([len(t) for t in page_list]).tolist(),
            "docling_doc_dict": document_initial,
            "logs": logger_content.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"),
            "version": "2.0",
            "conversion_status": converted.status.value,
        }
        if converted.status not in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
            errors = ", ".join([e.model_dump(mode="json") for e in converted.errors])
            metadata["extraction_error"] = f"Conversion failed with status {converted.status} and errors {errors} and logger content: {logger_content}".encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        return "".join(page_list), metadata