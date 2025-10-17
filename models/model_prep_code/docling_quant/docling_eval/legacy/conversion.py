import warnings
from pathlib import Path
from typing import List, Optional

# from docling.cli.main import OcrEngine
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    VlmPipelineOptions,
    smoldocling_vlm_conversion_options,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.factories import get_ocr_factory
from docling.pipeline.vlm_pipeline import VlmPipeline

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")

ocr_factory = get_ocr_factory()


def create_pdf_docling_converter(
    page_image_scale: float = 2.0,
    do_ocr: bool = False,
    ocr_lang: List[str] = ["en"],
    ocr_engine: str = EasyOcrOptions.kind,
    timings: bool = True,
    artifacts_path: Optional[Path] = None,
):
    force_ocr: bool = True
    ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
        kind=ocr_engine,
        force_full_page_ocr=force_ocr,
    )

    if ocr_lang is not None:
        ocr_options.lang = ocr_lang

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        ocr_options=ocr_options,
        do_table_structure=True,
        artifacts_path=artifacts_path,
    )

    pipeline_options.table_structure_options.do_cell_matching = True  # do_cell_matching
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    pipeline_options.images_scale = page_image_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = timings

    return doc_converter


def create_image_docling_converter(
    do_ocr: bool = False,
    ocr_lang: List[str] = ["en"],
    ocr_engine: str = EasyOcrOptions.kind,
    timings: bool = True,
):

    force_ocr: bool = True
    ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
        kind=ocr_engine,
        force_full_page_ocr=force_ocr,
    )

    if ocr_lang is not None:
        ocr_options.lang = ocr_lang

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        ocr_options=ocr_options,
        do_table_structure=True,
    )

    pipeline_options.table_structure_options.do_cell_matching = True  # do_cell_matching
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # pipeline_options.images_scale = page_image_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = timings

    return doc_converter


def create_smol_docling_converter(
    timings: bool = True,
):
    pipeline_options = VlmPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.accelerator_options.cuda_use_flash_attention2 = True
    pipeline_options.vlm_options = smoldocling_vlm_conversion_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = timings

    return converter
