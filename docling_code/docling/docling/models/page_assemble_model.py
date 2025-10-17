import logging
import os
import re
from collections.abc import Iterable
from typing import List
import os

import numpy as np
from pydantic import BaseModel
from docling_core.types.doc import DocItemLabel, TextCell, BoundingBox, CoordOrigin
import re
from docling.datamodel.base_models import (
    AssembledUnit,
    ContainerElement,
    FigureElement,
    Page,
    PageElement,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_model import BasePageModel
from docling.models.layout_model import LayoutModel
from docling.utils.profiling import TimeRecorder
from docling_code.custom_code.table import _extract_table_from_bbox

_log = logging.getLogger(__name__)


def sanitize_cells(cells: List[TextCell]):
    return sanitize_mineru_cells(cells)



class PageAssembleOptions(BaseModel):
    pass

FAULTY_UNICODE_CHARS = {
    '⁄': '/',
    '‘': "'",
    '’': "'",
    '”': '"',
    '“': '"',
    '\u0002': '-',
}
# Compile the regex pattern once for efficiency
fault_unicode_pattern = re.compile('|'.join(map(re.escape, FAULTY_UNICODE_CHARS.keys())))
def replace_faulty_unicode_chars(text: str):
    """Replace faulty unicode characters with their correct representations.
    
    Args:
        text: The text to process
        
    Returns:
        The text with unicode characters replaced
    """
    return fault_unicode_pattern.sub(lambda m: FAULTY_UNICODE_CHARS[m.group()], text)

def strip_all_ws(text: str):
    return text.strip()

def ends_with_ws(text: str):
    return text.endswith(" ") or text.endswith("\n") or text.endswith("\t") or text.endswith("\r")

def starts_with_ws(text: str):
    return text.startswith(" ") or text.startswith("\n") or text.startswith("\t") or text.startswith("\r")

def is_broken_header(text: str):
    # Check if the header resembled this case H E L L O (so 1 spaces uppercase of at least 4 characters)
    # Pattern: An uppercase letter followed by a space, repeated at least 3 times, ending with an uppercase letter.
    # This matches strings like "A B C D", "H E L L O", etc.

    NO_WS_UPPER = "[^ \t\n\r\f\va-z]"


    pattern = rf"^({NO_WS_UPPER} +){{3,}}{NO_WS_UPPER}$"
    if bool(re.fullmatch(pattern, text.strip())):
        return True
    return False



def get_y_overlap(cell1: TextCell, cell2: TextCell):
    overlap_y = max(0, min(cell1.rect.r_y2, cell2.rect.r_y2) - max(cell1.rect.r_y0, cell2.rect.r_y0))
    height_1 = cell1.rect.r_y2 - cell1.rect.r_y0
    height_2 = cell2.rect.r_y2 - cell2.rect.r_y0
    min_height = min(height_1, height_2)
    return overlap_y / min_height if min_height > 0 else 0

def cells_overlap(cell1: TextCell, cell2: TextCell, threshold: float = 0.8):
    x_left = max(cell1.rect.r_x0, cell2.rect.r_x0)
    x_right = min(cell1.rect.r_x1, cell2.rect.r_x1)
    max_width = min(cell1.rect.r_x1 - cell1.rect.r_x0, cell2.rect.r_x1 - cell2.rect.r_x0)
    overlap = (x_right - x_left) / max_width if max_width > 0 else 0
    if overlap > threshold:
        return True
    return False

def sanitize_mineru_cells(cells: List[TextCell], ignore_rotated: bool = False):
    cells = [cell for cell in cells if cell.text != ""]
    if ignore_rotated:
        removed_cells = [cell for cell in cells if abs(cell.info["line_angle"]) >= 5]
        cells = [cell for cell in cells if abs(cell.info["line_angle"]) < 5]
        if len(removed_cells) > 0:
            _log.warning(f"Removed {len(removed_cells)} cells with angle >= 5")


    if len(cells) == 0:
        return "", None, None 

    last_line_bbox = cells[-1].info["line_bbox"]

    all_cell_widths = [(cell.rect.r_x1 - cell.rect.r_x0)/ len(cell.text) for cell in cells for _ in range(len(cell.text))]
    median_char_width = float(np.median(all_cell_widths))

    if len(cells) <= 1:
        return replace_faulty_unicode_chars(" ".join([strip_all_ws(cell.text) for cell in cells])), median_char_width, last_line_bbox

    # Remove empty cells

    sanitized_text = strip_all_ws(cells[0].text)
    prev_cell = cells[0]
    for cell in cells[1:]:
        text_line = strip_all_ws(cell.text)
        # If the cells overlap and have same exact text, skip the current cell
        if prev_cell.text == cell.text and cells_overlap(prev_cell, cell):
            continue

        # Add no space if the cells are too close to each other and they are on same line

        is_super_script = bool(cell.info["flags"] & 1)
        last_is_super_script = bool(prev_cell.info["flags"] & 1)
        overlap_y = get_y_overlap(prev_cell, cell)

        # If they are on diff lines and the last char is a hyphen, remove the hyphen
        if sanitized_text.endswith("-") and overlap_y <= 0.8:
            prev_words = re.findall(r"\b[\w]+\b", sanitized_text)
            line_words = re.findall(r"\b[\w]+\b", text_line)

            if (
                len(prev_words)
                and len(line_words)
                and prev_words[-1].isalnum()
                and line_words[0].isalnum()
            ):
                sanitized_text = sanitized_text[:-1] + text_line
            else:
                if ends_with_ws(sanitized_text):
                    sanitized_text += text_line
                else:
                    sanitized_text += " " + text_line
        elif (abs(cell.rect.r_x0 - prev_cell.rect.r_x1) > median_char_width * 0.25 or (
            ends_with_ws(prev_cell.text) or starts_with_ws(cell.text)
            or overlap_y <= 0.8 or is_super_script or last_is_super_script
        )) and not ends_with_ws(sanitized_text) and not starts_with_ws(text_line):
            # If it's empty and it ends with
            # Print more info about why we are adding space
            sanitized_text += " " + text_line
        else:
            sanitized_text += text_line

        prev_cell = cell

    # Text normalization
    sanitized_text = replace_faulty_unicode_chars(sanitized_text).strip()

    return sanitized_text, median_char_width, last_line_bbox




def sanitize_cells_docling(cells: List[TextCell]):
    lines = [
        cell.text.replace("\x02", "-").strip()
        for cell in cells
        if len(cell.text.strip()) > 0
    ]
    if len(lines) <= 1:
        return " ".join(lines)

    for ix, line in enumerate(lines[1:]):
        prev_line = lines[ix]

        if prev_line.endswith("-"):
            prev_words = re.findall(r"\b[\w]+\b", prev_line)
            line_words = re.findall(r"\b[\w]+\b", line)

            if (
                len(prev_words)
                and len(line_words)
                and prev_words[-1].isalnum()
                and line_words[0].isalnum()
            ):
                lines[ix] = prev_line[:-1]
        else:
            lines[ix] += " "

    sanitized_text = "".join(lines)

    # Text normalization
    sanitized_text = sanitized_text.replace("⁄", "/")  # noqa: RUF001
    sanitized_text = sanitized_text.replace("’", "'")  # noqa: RUF001
    sanitized_text = sanitized_text.replace("‘", "'")  # noqa: RUF001
    sanitized_text = sanitized_text.replace("“", '"')
    sanitized_text = sanitized_text.replace("”", '"')
    sanitized_text = sanitized_text.replace("•", "·")

    return sanitized_text.strip()  # Strip any leading or trailing whitespace

class PageAssembleModel(BasePageModel):
    def __init__(self, options: PageAssembleOptions):
        self.options = options


    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "page_assemble"):
                    assert page.predictions.layout is not None

                    # assembles some JSON output page by page.

                    elements: List[PageElement] = []
                    headers: List[PageElement] = []
                    body: List[PageElement] = []

                    for cluster in page.predictions.layout.clusters:
                        # _log.info("Cluster label seen:", cluster.label)
                        if cluster.label in LayoutModel.TEXT_ELEM_LABELS:
                            text, median_char_width, last_line_bbox = sanitize_cells(cluster.cells)

                            if cluster.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE] and len(cluster.cells) == 1 and is_broken_header(text):
                                text = re.sub(r" (?! )", "", text)
                                text = re.sub(r" +", " ", text)

                                
                            text_el = TextElement(
                                label=cluster.label,
                                id=cluster.id,
                                text=text,
                                page_no=page.page_no,
                                cluster=cluster,
                                media_char_width=median_char_width,
                                last_line_bbox=last_line_bbox,
                            )
                            elements.append(text_el)

                            if cluster.label in LayoutModel.PAGE_HEADER_LABELS:
                                headers.append(text_el)
                            else:
                                body.append(text_el)
                        elif cluster.label in LayoutModel.TABLE_LABELS:
                            tbl = None
                            if page.predictions.tablestructure:
                                tbl = page.predictions.tablestructure.table_map.get(
                                    cluster.id, None
                                )
                            if not tbl:  # fallback: add table without structure, if it isn't present
                                tbl = Table(
                                    label=cluster.label,
                                    id=cluster.id,
                                    text="",
                                    otsl_seq=[],
                                    table_cells=[],
                                    cluster=cluster,
                                    page_no=page.page_no,
                                )
                                if cluster.label == DocItemLabel.TABLE and page._backend and page._backend.is_valid():
                                    try:
                                        table_data = _extract_table_from_bbox(page._backend._fpage, cluster.bbox.to_top_left_origin(page._backend.get_size().height))
                                        if table_data:
                                            tbl.num_rows = table_data.num_rows
                                            tbl.num_cols = table_data.num_cols
                                            tbl.table_cells = table_data.table_cells
                                    except Exception as e:
                                        _log.error(f"Error getting table data for cluster {cluster.id}: {e}")
                                else:
                                    pass

                            elements.append(tbl)
                            body.append(tbl)
                        elif cluster.label == LayoutModel.FIGURE_LABEL:
                            fig = None
                            if page.predictions.figures_classification:
                                fig = page.predictions.figures_classification.figure_map.get(
                                    cluster.id, None
                                )
                                
                            if not fig:  # fallback: add figure without classification, if it isn't present
                                fig = FigureElement(
                                    label=cluster.label,
                                    id=cluster.id,
                                    text="",
                                    data=None,
                                    cluster=cluster,
                                    page_no=page.page_no,
                                )
                            elements.append(fig)
                            body.append(fig)
                        elif cluster.label in LayoutModel.CONTAINER_LABELS:
                            container_el = ContainerElement(
                                label=cluster.label,
                                id=cluster.id,
                                page_no=page.page_no,
                                cluster=cluster,
                            )
                            elements.append(container_el)
                            body.append(container_el)

                    page.assembled = AssembledUnit(
                        elements=elements, headers=headers, body=body
                    )

                yield page
