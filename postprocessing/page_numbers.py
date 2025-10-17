from collections.abc import Iterable
from typing import List
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
import re
import numpy as np


def apply_page_number_tags(
    original_page_lines: List[str],
    top_common_lines: int, bottom_common_lines: int,
    drop: bool = False
) -> str:
    """
    Apply <docling_boilerplate_top/bottom> tags around detected boilerplate for a single page.
    
    Args:
        original_page_lines: Lines from the original page (with existing tags)
        top_common_lines: Number of common lines detected at top
        bottom_common_lines: Number of common lines detected at bottom
        drop: Whether to drop boilerplate instead of tagging it
    
    Returns:
        Modified page text with boilerplate wrapped in tags or removed
    """
    # Determine delimiter based on line split method
    delimiter = '\n'
    
    if len(original_page_lines) <= top_common_lines + bottom_common_lines:
        if drop:
            return ""  # Drop the entire page if it's all boilerplate
        else:
            original_text = delimiter.join(original_page_lines)
            return f"<docling_boilerplate_top>{original_text}</docling_boilerplate_top>"

    
    # Select only lines that are not common
    text = delimiter.join(original_page_lines[top_common_lines:(-bottom_common_lines if bottom_common_lines > 0 else len(original_page_lines))])
    
    if drop:
        return text
    else:
        # When tagging, wrap boilerplate in tags
        if top_common_lines > 0:
            top_text = f"<docling_boilerplate_top>{delimiter.join(original_page_lines[:top_common_lines])}</docling_boilerplate_top>"
            text = top_text + delimiter + text
        if bottom_common_lines > 0:
            bottom_text = f"<docling_boilerplate_bottom>{delimiter.join(original_page_lines[-bottom_common_lines:])}</docling_boilerplate_bottom>"
            text = text + delimiter + bottom_text

    return text

class PostprocessPageNumbers(PipelineStep):
    name: str = "🔢 PostprocessPageNumbers"
    type: str = "🔧 Formatter"
    def __init__(self):
        """
        Postprocess <docling_page_number>content</docling_page_number> tags, to remove them if the content contains only digits and is preceeded or followed by \\s*\\d+.
        """
        super().__init__()

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1) -> Iterable[Document]:
        compiled_pattern = re.compile(r'<docling_page_number>(.*?)</docling_page_number>')
        for document in data:
            with self.track_time():
                def replace_function(match):
                    content = match.group(1)
                    # Check if content contains only digits and whitespace
                    if not re.match(r'^\s*\d+\s*$', content):
                        return match.group(0)
                    
                    # Get the full text and position of the match
                    full_text = document.text
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Check what comes before the tag
                    before_text = full_text[:start_pos]
                    after_text = full_text[end_pos:]
                    
                    # Check if preceded by whitespace and digits
                    preceded_by_digits = bool(re.search(r'\s*\d+\s*$', before_text))
                    
                    # Check if followed by whitespace and digits
                    followed_by_digits = bool(re.search(r'^\s*\d+', after_text))
                    
                    self.stat_update("page_numbers_processed", 1, unit="document")
                    if preceded_by_digits or followed_by_digits:
                        self.stat_update("removed_page_numbers", 1, unit="document")
                        return content  # Remove tags but keep content
                    else:
                        return match.group(0)  # Keep original with tags
                    
                # Split original text into pages
                with self.track_time():
                    original_text = document.text
                    pages = []
                    start = 0
                    for offset in document.media[0].metadata["page_offsets"]:  # type: ignore
                        modified_page = compiled_pattern.sub(replace_function, original_text[start:offset])
                        pages.append(modified_page)
                        start = offset

                    document.text = "".join(pages)
                    document.media[0].metadata["page_offsets"] = [int(offset) for offset in np.cumsum([len(page) for page in pages])]  # type: ignore
            yield document