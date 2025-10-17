"""
List Item Marker Processor for Docling Documents

This module provides a rule-based model to identify list item markers and
merge marker-only TextItems with their content to create proper ListItems.
"""

import logging
import re
from typing import List, Optional, Tuple, Union

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    GroupLabel,
    ListItem,
    OrderedList,
    ProvenanceItem,
    RefItem,
    TextItem,
    UnorderedList,
    DocItemLabel,
)
from docling_code.custom_code.postprocessing.para_merge import merge_elements

_log = logging.getLogger(__name__)

LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；')
LINE_FIX_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', ':', '：', ';', '；')
from docling.models.page_assemble_model import strip_all_ws, ends_with_ws, starts_with_ws


def starts_with_line_fix_flag(text: str):
    return text.startswith(LINE_FIX_FLAG)

def is_on_same_line(item1: ProvenanceItem, item2: ProvenanceItem, threshold: float = 0.8):
    if item1.last_line_bbox is None:
        return False
    overlap_y = max(0, min(item1.last_line_bbox.t, item2.bbox.t) - max(item1.last_line_bbox.b, item2.bbox.b))
    height_1 = item1.last_line_bbox.height
    height_2 = item2.bbox.height
    min_height = max(height_1, height_2)
    # print(f"Overlap y: {overlap_y}, min_height: {min_height}, height_1: {height_1}, height_2: {height_2}, threshold: {threshold}")
    return (overlap_y / min_height if min_height > 0 else 0) >= threshold

def distance_in_chars(item1: ProvenanceItem, item2: ProvenanceItem):
    # Get average char width
    median_char_width = item1.media_char_width
    return abs((item2.bbox.l - item1.bbox.r) / median_char_width) if (median_char_width is not None and median_char_width > 0) else float('inf')


class CellsMerger:
    def process_document(self, doc: DoclingDocument, allowed_labels: List[DocItemLabel] = [DocItemLabel.TEXT, DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED, DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE, DocItemLabel.TITLE, DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER], allow_multi_prov: bool = False) -> DoclingDocument:

        """
        Process the entire document to identify and convert list markers.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """

        # Iterate over all items in the document, let's not merge over pages
        deleted_items = []
        for page in range(doc.num_pages()):
            # print(f"Processing page {page}")
            prev_item: Optional[TextItem] = None
            prev_level = 0
            for item, level in doc.iterate_items(page_no=page+1, with_groups=False):
                prev_level = level
                is_text_item = isinstance(item, TextItem)
                # print(f"Item: {item.text if hasattr(item, 'text') else None}, label: {item.label}, prev_item: {prev_item.text if prev_item else None}, prev_item_label: {prev_item.label if prev_item else None}")
                if not prev_item or not is_text_item or (len(item.prov) > 1 and not allow_multi_prov) or level != prev_level or item.label != prev_item.label or item.label not in allowed_labels:
                    if is_text_item and (len(item.prov) == 1 or allow_multi_prov):
                        prev_item = item
                    else:
                        prev_item = None
                    continue

                distance = distance_in_chars(prev_item.prov[-1], item.prov[0])
                # print(f"Distance: {distance}, prev_item: {prev_item.text}, item: {item.text}, bbox1: {prev_item.prov[-1].bbox}, bbox2: {item.prov[0].bbox}")
                if ends_with_ws(item.text):
                    text = strip_all_ws(item.text) + " "
                else:
                    text = strip_all_ws(item.text)
                if is_on_same_line(prev_item.prov[-1], item.prov[0]) and (distance < 6 or (starts_with_line_fix_flag(text) and distance < 30)):
                    # _log.warning(f"Merging {prev_item.text} and {item.text} because distance {distance} and {starts_with_line_fix_flag(text)} and overlap {get_overlap_y(prev_item.prov[-1], item.prov[0])}")
                    add_space = (distance > 0.25 or prev_item.text[-1] in LINE_STOP_FLAG) and not starts_with_line_fix_flag(text) and not ends_with_ws(prev_item.text) and not starts_with_ws(item.text)
                    # _log.warning(f"Merged {prev_item.text} and {item.text} with add_space {add_space}")
                    prev_item = merge_elements(prev_item, item, " " if add_space else "")
                    deleted_items.append(item)
                else:
                    prev_item = item
        if len(deleted_items) > 0:
            doc.delete_items(node_items=deleted_items)
        return doc