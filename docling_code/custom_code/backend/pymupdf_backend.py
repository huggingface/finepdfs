import logging
import math
import uuid
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Union

import fitz
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import BoundingRectangle, SegmentedPdfPage, TextCell
from PIL import Image
from pymupdf import Page as PyMuPDFPage

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.document import InputDocument
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)

_log = logging.getLogger(__name__)
fitz.set_messages(pylogging=True)

def get_pdf_page_geometry(
    ppage: fitz.Page,
    angle: float = 0.0,
    boundary_type: PdfPageBoundaryType = PdfPageBoundaryType.CROP_BOX,
) -> PdfPageGeometry:
    """
    Create PdfPageGeometry from a pypdfium2 PdfPage object.

    Args:
        ppage: pypdfium2 PdfPage object
        angle: Page rotation angle in degrees (default: 0.0)
        boundary_type: The boundary type for the page (default: CROP_BOX)

    Returns:
        PdfPageGeometry with all the different bounding boxes properly set
    """
    # Get the main bounding box (intersection of crop_box and media_box)
    bbox = BoundingBox.from_tuple((ppage.rect.x0, ppage.rect.y0, ppage.rect.x1, ppage.rect.y1), CoordOrigin.TOPLEFT)

    # Get all the different page boxes from pypdfium2
    media_box = BoundingBox.from_tuple((ppage.mediabox.x0, ppage.mediabox.y0, ppage.mediabox.x1, ppage.mediabox.y1), CoordOrigin.TOPLEFT)
    crop_box = BoundingBox.from_tuple((ppage.cropbox.x0, ppage.cropbox.y0, ppage.cropbox.x1, ppage.cropbox.y1), CoordOrigin.TOPLEFT)
    art_box = BoundingBox.from_tuple((ppage.artbox.x0, ppage.artbox.y0, ppage.artbox.x1, ppage.artbox.y1), CoordOrigin.TOPLEFT)
    bleed_box = BoundingBox.from_tuple((ppage.bleedbox.x0, ppage.bleedbox.y0, ppage.bleedbox.x1, ppage.bleedbox.y1), CoordOrigin.TOPLEFT)
    trim_box = BoundingBox.from_tuple((ppage.trimbox.x0, ppage.trimbox.y0, ppage.trimbox.x1, ppage.trimbox.y1), CoordOrigin.TOPLEFT)

    # Convert to BoundingBox objects using existing from_tuple method
    # pypdfium2 returns (x0, y0, x1, y1) in PDF coordinate system (bottom-left origin)
    # Use bbox as fallback when specific box types are not defined

    return PdfPageGeometry(
        angle=angle,
        rect=BoundingRectangle.from_bounding_box(bbox),
        boundary_type=boundary_type,
        art_bbox=art_box,
        bleed_bbox=bleed_box,
        crop_bbox=crop_box,
        media_bbox=media_box,
        trim_bbox=trim_box,
    )


def blocks_to_cells(blocks: list[dict], page_height:float, ignore_empty: bool = True):
    cell_counter = 0
    cells = []
    for b in blocks:
        for l in b.get("lines", []):  # noqa: E741
            for s in l["spans"]:
                text = s["text"]
                bbox = s["bbox"]
                x0, y0, x1, y1 = bbox
                cos, sin = l.get("dir", [1, 0])
                line_angle = math.degrees(math.atan2(sin, cos))

                if ignore_empty and (abs(x1 - x0) == 0 or abs(y1 - y0) == 0):
                    continue

                cells.append(
                    TextCell(
                        index=cell_counter,
                        text=text,
                        orig=text,
                        from_ocr=False,
                        rect=BoundingRectangle.from_bounding_box(
                            BoundingBox(
                                l=x0,
                                b=y0,
                                r=x1,
                                t=y1,
                            )
                        ),
                        info={
                            **s,
                            "line_bbox": BoundingBox(l=x0, t=y0, r=x1, b=y1, coord_origin=CoordOrigin.TOPLEFT).to_bottom_left_origin(page_height),
                            "line_angle": line_angle
                        }
                    )
                )
                cell_counter += 1

    return cells

class PyMuPdfPageBackend(PdfPageBackend):
    def __init__(self, doc_obj: fitz.Document, document_hash: str, page_no: int):
        super().__init__()
        self.valid = True

        try:
            self._fpage: fitz.Page = doc_obj.load_page(page_no)
            self._fpage.remove_rotation()
        except Exception as e:
            _log.info(
                f"An exception occured when loading page {page_no} of document {document_hash}.",
                exc_info=True,
            )
            self.valid = False

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        if not self.valid:
            return ""

        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            bbox = bbox.to_top_left_origin(self.get_size().height)

        rect = fitz.Rect(*bbox.as_tuple())
        text_piece = self._fpage.get_text("text", clip=rect)

        return text_piece

    def get_pymupdf_page(self) -> PyMuPDFPage:
        return self._fpage

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        if not self.valid:
            return None

        text_cells = list(self.get_text_cells())

        dimension = get_pdf_page_geometry(self._fpage)

        # Create SegmentedPdfPage
        return SegmentedPdfPage(
            dimension=dimension,
            textline_cells=text_cells,
            char_cells=[],
            word_cells=[],
            has_words=False,
            has_chars=False,
        )

    def get_text_cells(self) -> Iterable[TextCell]:
        cells = []

        if not self.valid:
            return cells

        blocks = self._fpage.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_CID_FOR_UNKNOWN_UNICODE)["blocks"]
        return blocks_to_cells(blocks, page_height=self.get_size().height)

    def get_bitmap_rects(self, float: int = 1) -> Iterable["BoundingBox"]:
        AREA_THRESHOLD = 32 * 32

        images = self._fpage.get_image_info()

        for im in images:
            cropbox = BoundingBox.from_tuple(im["bbox"], origin=CoordOrigin.TOPLEFT)
            if cropbox.area() > AREA_THRESHOLD:
                cropbox = cropbox.scaled(scale=float)

                yield cropbox

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        if not self.valid:
            return None

        if not cropbox:
            pix = self._fpage.get_pixmap(matrix=fitz.Matrix(scale, scale))
        else:
            page_height = self.get_size().height
            cropbox = cropbox.to_top_left_origin(page_height)
            pix = self._fpage.get_pixmap(
                matrix=fitz.Matrix(scale, scale), clip=cropbox.as_tuple()
            )

        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples_mv)
        return image

    def get_size(self) -> Size:
        return Size(width=self._fpage.cropbox.width, height=self._fpage.cropbox.height)

    def is_valid(self) -> bool:
        return self.valid

    def unload(self):
        self._fpage = None


class PyMuPdfDocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        success = False
        if isinstance(self.path_or_stream, Path):
            self._fdoc: fitz.Document = fitz.open(str(self.path_or_stream))
            success = True
        elif isinstance(self.path_or_stream, BytesIO):
            self._fdoc: fitz.Document = fitz.open(
                filename=str(uuid.uuid4()), filetype="pdf", stream=path_or_stream
            )
            success = True

        if not success:
            raise RuntimeError(
                f"PyMuPdf could not load document with hash {self.document_hash}."
            )

    def page_count(self) -> int:
        return self._fdoc.page_count

    def load_page(self, page_no: int) -> PyMuPdfPageBackend:
        return PyMuPdfPageBackend(self._fdoc, self.document_hash, page_no)

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        self._fdoc.close()
        self._fdoc = None