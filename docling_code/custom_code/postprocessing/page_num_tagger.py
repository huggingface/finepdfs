"""
Page Number Remover Processor for Docling Documents

This module provides a processor to remove page numbers from Docling Documents.
"""

import logging
import re
from typing import List, Optional, Pattern
from functools import lru_cache

from docling_core.types.doc.document import (
    DoclingDocument,
    PageNumberItem,
    ProvenanceItem,
    TextItem,
    PictureItem,
    DocItemLabel,
    BoundingBox,
    Size
)

_log = logging.getLogger(__name__)


def _build_page_patterns():
    """Build comprehensive page number patterns with reduced repetition."""
    
    # Define pattern for legitimate page numbers (0 or 1-9999, no leading zeros)
    page_num = r'(?:0|[1-9]\d{0,3})'
    
    # Language-specific page words and their connectors (50 most popular languages)
    page_data = {
        'en': {'words': ['Page', 'p', 'pp'], 'connectors': ['of']},
        'zh': {'words': ['页'], 'connectors': ['的']},  # Simplified Chinese
        'hi': {'words': ['पृष्ठ', 'पेज'], 'connectors': ['का', 'से']},  # Hindi
        'es': {'words': ['Página', 'Pág'], 'connectors': ['de']},
        'fr': {'words': ['Page', 'p'], 'connectors': ['sur', 'de']},
        'ar': {'words': ['صفحة'], 'connectors': ['من']},
        'bn': {'words': ['পৃষ্ঠা', 'পাতা'], 'connectors': ['এর']},  # Bengali
        'ru': {'words': ['Страница', 'Стр'], 'connectors': ['из']},
        'pt': {'words': ['Página', 'Pág'], 'connectors': ['de']},
        'id': {'words': ['Halaman', 'Hal'], 'connectors': ['dari']},  # Indonesian
        'ur': {'words': ['صفحہ'], 'connectors': ['کا']},  # Urdu
        'de': {'words': ['Seite', 'S'], 'connectors': ['von']},
        'ja': {'words': ['ページ', 'P'], 'connectors': ['の']},
        'sw': {'words': ['Ukurasa', 'Uk'], 'connectors': ['wa']},  # Swahili
        'mr': {'words': ['पान', 'पृष्ठ'], 'connectors': ['चे']},  # Marathi
        'te': {'words': ['పేజీ', 'పుట'], 'connectors': ['లో']},  # Telugu
        'tr': {'words': ['Sayfa', 'S'], 'connectors': ['den']},  # Turkish
        'ta': {'words': ['பக்கம்', 'பக்'], 'connectors': ['இல்']},  # Tamil
        'vi': {'words': ['Trang', 'Tr'], 'connectors': ['của']},  # Vietnamese
        'ko': {'words': ['페이지', '쪽'], 'connectors': ['의']},
        'it': {'words': ['Pagina', 'Pag'], 'connectors': ['di']},
        'th': {'words': ['หน้า'], 'connectors': ['ของ']},  # Thai
        'gu': {'words': ['પાનું', 'પેજ'], 'connectors': ['નું']},  # Gujarati
        'pl': {'words': ['Strona', 'Str'], 'connectors': ['z']},
        'uk': {'words': ['Сторінка', 'Стор'], 'connectors': ['з']},  # Ukrainian
        'kn': {'words': ['ಪುಟ', 'ಪೇಜ್'], 'connectors': ['ರ']},  # Kannada
        'ml': {'words': ['പേജ്', 'താൾ'], 'connectors': ['ന്റെ']},  # Malayalam
        'or': {'words': ['ପୃଷ୍ଠା', 'ପେଜ୍'], 'connectors': ['ର']},  # Odia
        'pa': {'words': ['ਪੰਨਾ', 'ਪੇਜ'], 'connectors': ['ਦਾ']},  # Punjabi
        'ro': {'words': ['Pagina', 'Pag'], 'connectors': ['din']},  # Romanian
        'nl': {'words': ['Pagina', 'Pag', 'p'], 'connectors': ['van']},
        'hu': {'words': ['oldal', 'o'], 'connectors': ['ból']},
        'el': {'words': ['Σελίδα', 'Σελ'], 'connectors': ['από']},  # Greek
        'cs': {'words': ['Strana', 'Str'], 'connectors': ['z']},
        'be': {'words': ['Старонка', 'Стар'], 'connectors': ['з']},  # Belarusian
        'he': {'words': ['עמוד', 'עמ'], 'connectors': ['מתוך']},  # Hebrew
        'sv': {'words': ['Sida', 'S'], 'connectors': ['av']},
        'az': {'words': ['Səhifə', 'Səh'], 'connectors': ['dan']},  # Azerbaijani
        'bg': {'words': ['Страница', 'Стр'], 'connectors': ['от']},  # Bulgarian
        'ms': {'words': ['Muka surat', 'Ms'], 'connectors': ['daripada']},  # Malay
        'uz': {'words': ['Sahifa', 'Sah'], 'connectors': ['dan']},  # Uzbek
        'ne': {'words': ['पृष्ठ', 'पेज'], 'connectors': ['को']},  # Nepali
        'si': {'words': ['පිටුව', 'පි'], 'connectors': ['හි']},  # Sinhala
        'kk': {'words': ['Бет', 'Б'], 'connectors': ['дан']},  # Kazakh
        'am': {'words': ['ገጽ'], 'connectors': ['ከ']},  # Amharic
        'ka': {'words': ['გვერდი', 'გვ'], 'connectors': ['დან']},  # Georgian
        'no': {'words': ['Side', 'S'], 'connectors': ['av']},
        'da': {'words': ['Side', 'S'], 'connectors': ['af']},
        'fi': {'words': ['Sivu', 'S'], 'connectors': ['ja']},  # Finnish
        'sk': {'words': ['Stránka', 'Str'], 'connectors': ['z']},  # Slovak
        'hr': {'words': ['Stranica', 'Str'], 'connectors': ['od']},  # Croatian
    }
    
    # Special case languages with unique formatting
    special_patterns = [
        # Traditional Chinese
        rf'^\s*第\s*{page_num}\s*页\s*$',                    # 第1页
        rf'^\s*第\s*{page_num}\s*页\s*/\s*{page_num}\s*$',         # 第1页/10
        rf'^\s*第\s*{page_num}\s*页\s*共\s*{page_num}\s*页\s*$',   # 第1页共10页
        rf'^\s*第\s*{page_num}\s*頁\s*$',                    # 第1頁 (Traditional)
        rf'^\s*第\s*{page_num}\s*頁\s*/\s*{page_num}\s*$',         # 第1頁/10
        
        # Japanese special formats
        rf'^\s*{page_num}\s*ページ\s*$',                     # 1ページ
        rf'^\s*{page_num}\s*ページ\s*/\s*{page_num}\s*$',          # 1ページ/10
        rf'^\s*{page_num}\s*/\s*{page_num}\s*ページ\s*$',          # 1/10ページ
        rf'^\s*P\.\s*{page_num}\s*$',                        # P.1
        
        # Korean special formats
        rf'^\s*{page_num}\s*페이지\s*$',                     # 1페이지
        rf'^\s*{page_num}\s*페이지\s*/\s*{page_num}\s*$',          # 1페이지/10
        rf'^\s*{page_num}\s*/\s*{page_num}\s*페이지\s*$',          # 1/10페이지
        rf'^\s*{page_num}\s*쪽\s*$',                         # 1쪽
        
        # Hungarian special format
        rf'^\s*{page_num}\.\s*oldal\s*$',                    # 1. oldal
        rf'^\s*{page_num}\.\s*oldal\s*/\s*{page_num}\s*$',         # 1. oldal/10
        
        # Arabic/Urdu RTL formats
        rf'^\s*{page_num}\s*صفحة\s*$',                       # 1 صفحة
        rf'^\s*{page_num}\s*صفحہ\s*$',                       # 1 صفحہ (Urdu)
        
        # Thai special format
        rf'^\s*หน้า\s*{page_num}\s*$',                       # หน้า 1
        rf'^\s*หน้า\s*{page_num}\s*/\s*{page_num}\s*$',            # หน้า 1/10
        
        # Hebrew RTL format
        rf'^\s*עמוד\s*{page_num}\s*$',                       # עמוד 1
        rf'^\s*עמ\'\s*{page_num}\s*$',                       # עמ' 1
    ]
    
    patterns = []
    
    # Simple number patterns (no prefix)
    patterns.extend([
        rf'^\s*{page_num}\s*$',                              # Just a number: 1
        rf'^\s*{page_num}\s*/\s*{page_num}\s*$',                   # Format: 1/10, 1 / 10
    ])
    
    # Add special case patterns
    patterns.extend(special_patterns)
    
    # Build patterns for standard languages
    for lang_data in page_data.values():
        words = lang_data['words']
        # Use both language-specific connectors AND English "of"
        connectors = lang_data['connectors'] + ['of']
        
        for word in words:
            if '.' in word or len(word) <= 2 or word.islower():
                # Abbreviated forms (p., Pág., S., etc.)
                patterns.extend([
                    rf'^\s*{re.escape(word)}\s*{page_num}\s*$',                        # p. 1
                    rf'^\s*{re.escape(word)}\s*{page_num}\s*/\s*{page_num}\s*$',             # p. 1/10
                    rf'^\s*{re.escape(word)}\s*{page_num}\s*{page_num}[a-z\/]{page_num}\s*$',       # p. 1of2, p. 1/2
                ])
            else:
                # Full words (Page, Página, etc.)
                patterns.extend([
                    rf'^\s*{re.escape(word)}\s+{page_num}\s*$',                        # Page 1
                    rf'^\s*{re.escape(word)}\s+{page_num}\s*/\s*{page_num}\s*$',             # Page 1/10
                    rf'^\s*{re.escape(word)}\s+{page_num}\s*{page_num}[a-z\/]{page_num}\s*$',       # Page 1of2, Page 1/2
                ])
            
            # Add explicit connector patterns (e.g., "Page 1 of 2", "Strana 1 of 2")
            for connector in connectors:
                if '.' in word or len(word) <= 2 or word.islower():
                    patterns.append(rf'^\s*{re.escape(word)}\s*{page_num}\s+{re.escape(connector)}\s+{page_num}\s*$')
                else:
                    patterns.append(rf'^\s*{re.escape(word)}\s+{page_num}\s+{re.escape(connector)}\s+{page_num}\s*$')
            
            # Add colon variants
            patterns.extend([
                rf'^\s*{re.escape(word)}:\s*{page_num}\s*$',                        # Pagina: 18
                rf'^\s*{re.escape(word)}:\s*{page_num}\s*/\s*{page_num}\s*$',             # Pagina: 18/24 or Pagina: 18 / 24
            ])
    
    return patterns


@lru_cache(maxsize=1)
def _get_compiled_patterns() -> List[Pattern[str]]:
    """Get compiled regex patterns with caching for performance."""
    patterns = _build_page_patterns()
    compiled_patterns = []
    
    for pattern in patterns:
        try:
            compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            _log.debug(f"Failed to compile regex pattern '{pattern}': {e}")
            continue
    
    _log.debug(f"Compiled {len(compiled_patterns)} page number patterns")
    return compiled_patterns


class PageNumberRemover:
    """
    A processor that removes page numbers from document headers and footers.
    """

    def __init__(self):
        """Initialize the processor and compile regex patterns."""
        self._compiled_patterns = _get_compiled_patterns()

    def process_document(self, doc: DoclingDocument) -> DoclingDocument:
        """
        Process the document to remove page numbers.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """
        page_numbers = []
        for page in range(doc.num_pages()):
            
            # Get first and last items on page
            items = list(doc.iterate_items(page_no=page + 1, with_groups=False))
            if not items:
                continue
                
            # Find first and last TextItem
            first_text_item = None
            last_text_item = None
            
            for item, _ in items:
                if isinstance(item, TextItem):
                    if first_text_item is None:
                        first_text_item = item
                    last_text_item = item
            
            if not first_text_item or not last_text_item:
                continue
                
            # Check if items are text and have valid labels
            valid_labels = {DocItemLabel.TITLE, DocItemLabel.FOOTNOTE, DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER}
            
            # Avoid adding the same item twice when first and last are identical
            candidates = [first_text_item]
            if last_text_item is not first_text_item:
                candidates.append(last_text_item)
            
            for item in candidates:
                if item.label not in valid_labels:
                    continue
                    
                text = item.text.strip()
                
                # Check if text matches any page number pattern using compiled patterns
                is_page_num = any(pattern.match(text) for pattern in self._compiled_patterns)
                
                if is_page_num:
                    _log.debug(f"Found page number: '{text}'")
                    page_numbers.append(item)
        
        # Delete page number items
        if page_numbers:
            for to_delete_page_number in page_numbers:
                page_number = PageNumberItem(
                    self_ref="#",
                    text=to_delete_page_number.text,
                    orig=to_delete_page_number.orig,
                    label=DocItemLabel.PAGE_NUMBER,
                    prov=to_delete_page_number.prov,
                )
                doc.insert_item_after_sibling(
                    new_item=page_number, sibling=to_delete_page_number
                )
            doc.delete_items(node_items=page_numbers)
            _log.debug(f"Removed {len(page_numbers)} page numbers")
            
        return doc