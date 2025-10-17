"""
Paragraph Fix Processor for Docling Documents

This module provides a processor to merge text blocks that should be part of the same paragraph
based on layout and content analysis.
"""

import logging
import re
from docling.models.page_assemble_model import ends_with_ws, starts_with_ws
from typing import List, Optional

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    TextItem,
    PageNumberItem,
)
from docling_core.types.doc.base import BoundingBox, Size


ALLOWD_LABEL_PAIRS = {
    DocItemLabel.LIST_ITEM: [DocItemLabel.TEXT],
    DocItemLabel.TEXT: [DocItemLabel.TEXT],
}

_log = logging.getLogger(__name__)

LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '"', ':', '：', ';', '；')

def should_merge_blocks(block1: TextItem, block2: TextItem) -> bool:
    """
    Determine if two text blocks should be merged based on the original logic.
    
    Args:
        block1: The previous text block (equivalent to block2 in original function)
        block2: The current text block (equivalent to block1 in original function)
        
    Returns:
        True if blocks should be merged, False otherwise
    """
    # Get bounding boxes
    block1_bbox = block1.prov[-1].bbox
    block2_bbox = block2.prov[0].bbox
    
    # Calculate dimensions
    median_char_width_block2 = block1.prov[-1].media_char_width
    median_char_width_block1 = block2.prov[0].media_char_width

    if not median_char_width_block1 or not median_char_width_block2 or (
        abs(median_char_width_block1 - median_char_width_block2) / 
        max(median_char_width_block1, median_char_width_block2) > 0.1
    ):

        if median_char_width_block1 and median_char_width_block2:
            # print(f"Skipping {block1.text} and {block2.text} because median char width is different: {median_char_width_block1} and {median_char_width_block2} -> {abs(median_char_width_block1 - median_char_width_block2) / max(median_char_width_block1, median_char_width_block2)}")
            pass
        return False

    last_line_bbox = block1.prov[-1].last_line_bbox
    if not last_line_bbox:
        return False

    # Check placement on x
    if abs(block1_bbox.l - block2_bbox.l) / median_char_width_block1 > 5 or abs(block1_bbox.r - block2_bbox.r) / median_char_width_block1 > 5:
        # print(f"Skipping {block1.text} and {block2.text} because x placement is different: {block1_bbox.l} and {block2_bbox.l} -> {abs(block1_bbox.l - block2_bbox.l) / median_char_width_block1}, {block1_bbox.r} and {block2_bbox.r} -> {abs(block1_bbox.r - block2_bbox.r) / median_char_width_block1}")
        return False

    # Ensure the blocks are close in case they are on same page
    if block1.prov[-1].page_no == block2.prov[0].page_no and abs(block1_bbox.b - block2_bbox.t) / median_char_width_block1 > 1:
        # print(f"Skipping {block1.text} and {block2.text} because y placement is different: {block1_bbox.b} and {block2_bbox.t} -> {abs(block1_bbox.b - block2_bbox.t) / median_char_width_block1}")
        return False


    if abs(block1_bbox.r - last_line_bbox.r) / median_char_width_block1 >= 2:
        # print(f"Skipping {block1.text} and {block2.text} because line-x placement is different: {block1_bbox.r} and {last_line_bbox.r} -> {abs(block1_bbox.r - last_line_bbox.r) / median_char_width_block1}")
        return False

    
    
    # Get text content
    block1_text = block1.text.strip()
    block2_text = block2.text.strip()
    
    if not block1_text or not block2_text:
        return False
        
    # Check if first character is digit or uppercase
    span_start_with_num = block2_text[0].isdigit()
    span_start_with_big_char = block2_text[0].isupper()
    
    # Check all merge conditions
    merge_conditions = [
        # last line right bbox should be aligned with the total bbox
        # Previous block doesn't end with line stop flag
        not block1_text.endswith(LINE_STOP_FLAG),
        
        not span_start_with_num,
        
        # Next block shouldn't start with uppercase letter
        not span_start_with_big_char
    ]
    
    return all(merge_conditions)

def merge_elements(item: TextItem, merge_item: TextItem, sep: str = " "):
    item_text_len = len(item.text)
    # Update charspans of merged item
    for prov in merge_item.prov:
        prov.charspan = (
            item_text_len + len(sep) + prov.charspan[0],
            item_text_len + len(sep) + prov.charspan[1]
        )
        prov.page_no = item.prov[-1].page_no
    item.text += f"{sep}{merge_item.text}"
    item.prov.extend(merge_item.prov)
    return item




class ParagraphMerger:
    """
    A processor that merges text blocks that should be part of the same paragraph.
    """

    def process_document(self, doc: DoclingDocument, allow_multi_prov: bool = False) -> DoclingDocument:
        """
        Process the entire document to identify and merge paragraph blocks.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """
        
        # Process each page separately
        deleted_items = []
        
        prev_item: Optional[TextItem] = None
        prev_level = 0
        for page in range(doc.num_pages()):
            _log.debug(f"Processing page {page + 1}")
            
            for item, level in doc.iterate_items(page_no=page + 1, with_groups=False):
                if isinstance(item, PageNumberItem):
                    continue
                prev_level = level
                is_text_item = isinstance(item, TextItem)
                
                # Only process TextItems with single provenance
                if not prev_item or not is_text_item or (len(item.prov) > 1 and not allow_multi_prov) or level != prev_level or (item.label not in ALLOWD_LABEL_PAIRS.get(prev_item.label, [])):
                    if is_text_item and len(item.prov) == 1:
                        prev_item = item
                    else:
                        prev_item = None
                    continue
                
                # Check if we should merge prev_item (block2) with current item (block1)
                if should_merge_blocks(prev_item, item):
                    _log.debug(f"PARA MERGE: Merging blocks: '{prev_item.text[:50]}...' and '{item.text[:50]}...' {prev_item.label} and {item.label}")

                    sep = " "
                    if prev_item.text.endswith("-"):
                        prev_words = re.findall(r"\b[\w]+\b", prev_item.text)
                        line_words = re.findall(r"\b[\w]+\b", item.text)

                        if (
                            len(prev_words)
                            and len(line_words)
                            and prev_words[-1].isalnum()
                            and line_words[0].isalnum()
                        ):
                            prev_item.text = prev_item.text[:-1]
                            # Update its charspan
                            prev_item.prov[-1].charspan = (
                                prev_item.prov[-1].charspan[0],
                                prev_item.prov[-1].charspan[1] - 1
                            )
                            sep = ""

                    if ends_with_ws(prev_item.text) or starts_with_ws(item.text):
                        sep = ""

                    # _log.warning(f"Merged {prev_item.text} and {item.text} with sep {sep}")
                    prev_item = merge_elements(prev_item, item, sep)
                    deleted_items.append(item)
                else:
                    prev_item = item
        
        # Delete merged items
        if deleted_items:
            doc.delete_items(node_items=deleted_items)
            # _log.debug(f"Merged {len(deleted_items)} text blocks")
        
        return doc
    