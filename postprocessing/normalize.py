import re
import unicodedata
import ftfy
from datatrove.pipeline.base import PipelineStep

ftfy_config = ftfy.TextFixerConfig(
    unescape_html="auto",
    remove_terminal_escapes=True,
    fix_encoding=True,
    restore_byte_a0=True,
    replace_lossy_sequences=True,
    decode_inconsistent_utf8=True,
    fix_c1_controls=True,
    fix_latin_ligatures=False,
    fix_character_width=False,
    uncurl_quotes=False,
    fix_line_breaks=True,
    fix_surrogates=True,
    remove_control_chars=True,
    normalization=None,
)

# Start-of-line "split words": chars separated by single spaces; words separated by 2+ spaces
PREFIX_RE = re.compile(
    r'^(?P<lead>[ \t]*)'
    r'(?P<prefix>(?:\S(?: \S)*)(?: {2,}\S(?: \S)*)*:?)'
)
JOIN_SINGLE = re.compile(r'(?<=\S) (?=\S)')  # remove exactly one space between non-spaces

TABLE_RE = re.compile(r'^\s*(<docling_table>)?\s*\|.*\|\s*(</docling_table>)?\s*$')

PUNCTUATION = "\"!”:【，】；“„';∶`．？！：〉,&?。…." + "".join(
    map(
        chr,
        (x for a, b in ((0, 9), (11, 13), (13, 32), (127, 160)) for x in range(a, b)),
    )
)

DASHES = {'-', '‐', '‒', '–', '—'}

DOCLING_TAGS = re.compile(r'<docling_formula>|</docling_formula>|<docling_image>|</docling_image>|<docling_picture_annotation(?:_non_text)?>|</docling_picture_annotation(?:_non_text)?>|<docling_page_number>|</docling_page_number>|<docling_table>|</docling_table>')
FAILED_PAGE_RE = re.compile(r'<failed_page>.*?</failed_page>', re.DOTALL)
CORRUPTED_PAGE_RE = re.compile(r'<corrupted_page>.*?</corrupted_page>', re.DOTALL)


TERMINAL_PUNCTUATION = {
    "᪩","？","⁈","꩞","﹗","፧","𑅂","꡶","⁉","࠾","᪨","𑊩","𑱂",
    "᱿","𖩮","᥅","﹒","𑈹","𑈸","܂","؞","꛳","𑗍",
    "𐩖","꩟","᠉","𑗗","᰼","𑻸","؟","𑪜","꧉","𑗉","𐽙","𖫵","𖬷","܀","꓿",
    "᜵","𑗏","𑁇","𑗓","𑥄","៖","𑥆","𑗑","𑗒","꯫","۔","𐩗","꡷","｡","៕",
    "߹","⸮",".","𑇅","࠹","𛲟","꫰","꤯","𐽗","᭞","𑜼","፨","𑃁","꣏","𑇟","𖬸","𑜾","࠷",
    "?","𑃀","𑗃","！","։","꣎","॥","𑗖","᭛","᠃","!","၊","𖺘","⁇","𑗌","𑑋","𖭄","᭟","𑅁","𑙁",
    "⸼","꩝","𑗋","。","꧈","꫱","𑜽","𐽖","𑂿","᙮","។","꛷","៚","᥄","𑗕","𑗎","᪪",
    "᭚","࠽","𑇞","𑗊","𐽘","𑗔","𖩯","𑇍","𑻷","𐽕","𑩃","।","𑗂","𑇆","𑁈","။","᱾",
    "𑱁","꘏","܁","᜶","‼","𑈻","‽","᪫","﹖","𑑌","𑈼","𑗐","៙","᰻",
}

# Build a set for fast membership checks
_PUNCT_SET = set(PUNCTUATION) | set(TERMINAL_PUNCTUATION)

# Compiled regex patterns
_WEIRD_SPACES_RE = re.compile(r"\t\r \xa0[^\S\n]*")
_BMP_PUA_RE = re.compile(r'[\uE000-\uF8FF]')
_SUPP_PUA_RE = re.compile(r'[\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]')
_REPLACEMENT_CHAR_RE = re.compile(r'�')
_WHITESPACE_RE = re.compile(r"\s+")
_TWO_PLUS_SPACES_RE = re.compile(r" {2,}")
_TWO_PLUS_NEWLINES_RE = re.compile(r"\n{2,}")

def _word_char_ratio_ok(chunk: str, threshold_8: float = 0.65, threshold_0: float = 1.0) -> bool:
    """Return True iff >= threshold of non-space chars are \\w."""
    no_spaces = _WHITESPACE_RE.sub("", chunk)
    if not no_spaces:
        return False

    alpha_count = sum(c.isalpha() for c in no_spaces)
    alpha_ratio = alpha_count / len(no_spaces)
    if len(no_spaces) <= 8:
        return alpha_ratio >= threshold_0
    else:
        return alpha_ratio >= threshold_8
    

def fix_split_any_at_line_start(
    text: str,
    min_word_letters: int = 4,   # 'largest >= ...'
    min_total_letters: int = 10  # 'total >= ...'
) -> str:
    out = []
    for line in text.splitlines():
        if TABLE_RE.match(line):
            out.append(line)
            continue

        m = PREFIX_RE.match(line)
        if not m:
            out.append(line)
            continue

        raw_prefix = m.group("prefix")
        # words inside prefix are separated by 2+ spaces
        words = [w for w in _TWO_PLUS_SPACES_RE.split(raw_prefix.strip()) if w]
        counts = [len(w.split(" ")) for w in words]
        largest = max(counts) if counts else 0
        total_letters = sum(counts)

        # Gate: largest >= min_word_letters OR (total >= min_total_letters AND >= 2 words)
        if not ((largest >= min_word_letters or (total_letters >= min_total_letters and len(words) >= 2)) and _word_char_ratio_ok(raw_prefix)):
            out.append(line)
            continue

        # Join only inside the prefix; preserve 2+ spaces as word gaps (collapsed to one)
        fixed_prefix = JOIN_SINGLE.sub("", raw_prefix)
        fixed_prefix = _TWO_PLUS_SPACES_RE.sub(" ", fixed_prefix)

        # Keep the original remainder exactly as-is…
        rest = line[m.end():]
        # …but GUARANTEE a space boundary if the remainder starts with a non-space
        if rest and not rest[0].isspace() and fixed_prefix and not fixed_prefix[-1].isspace() and not fixed_prefix.endswith(":"):
            rest  = " " + fixed_prefix[-1] + rest
            fixed_prefix = fixed_prefix[:-1]

        s = m.group("lead") + fixed_prefix + rest
        out.append(s)

    return "\n".join(out)


def _is_punct_only(s: str) -> bool:
    s = s.strip()
    if len(s) == 0:
        return False
    return s in _PUNCT_SET

def fix_ocr_punctuation_lines(text: str) -> str:
    """
    Join lines that contain only punctuation with the previous and following lines.
    Uses the provided PUNCTUATION and TERMINAL_PUNCTUATION inventories.
    
    Example:
        "This is the reason\\n:  ->  "This is the reason: "
    """
    lines = text.splitlines()
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if _is_punct_only(line):
            # Merge consecutive punctuation-only lines into a single block
            if out:
                out_i = len(out) - 1
                # Remove empty lnes until we find a non-empty line
                while out_i >= 0 and len(out[out_i].strip()) == 0:
                    out_i -= 1

                if out_i >= 0 and (out[out_i][-1].isspace() or out[out_i][-1].isalpha() or out[out_i][-1].isdigit()):
                    out = out[:out_i+1]
                    out[-1] = out[-1] + line
                else:
                    out.append(line)
            else:
                out.append(line)
            i += 1
        else:
            out.append(line)
            i += 1

    return "\n".join(out)

def normalize_weird_spaces(text: str) -> str:
    text = _WEIRD_SPACES_RE.sub(" ", text)
    return text

def normalize_text(text: str) -> str:
    # Remove BMP PUA
    text = _BMP_PUA_RE.sub('', text)
    # Remove Supplementary PUA
    text = _SUPP_PUA_RE.sub('', text)
    text = _REPLACEMENT_CHAR_RE.sub('', text)
    return text

def clean_pdf_page(text: str) -> str:
    # We start with fixing the weird whitespace == \t\r \xa0[^\S\n]*
    text = normalize_weird_spaces(text)
    # Then fix the punctuation lines
    text = fix_ocr_punctuation_lines(text)
    # Then fix the split lines
    text = fix_split_any_at_line_start(text)
    # Then normalize the whole text, remove PUA
    text = normalize_text(text)

    # Normalize double \n\n+ to \n\n
    text = _TWO_PLUS_NEWLINES_RE.sub("\n\n", text)

    return text.strip()

def remove_tags(text: str, is_from_docling: bool = False) -> str:
    if not is_from_docling:
        text = FAILED_PAGE_RE.sub("", text)
        text = CORRUPTED_PAGE_RE.sub("", text)
    else:
        text = DOCLING_TAGS.sub("", text)
    return text

def pages_to_text(pages: list[str]) -> tuple[str, list[int], list[int]]:
    non_empty_pages_with_indices = [(i, page.strip()) for i, page in enumerate(pages) if page.strip()]
    if not non_empty_pages_with_indices:
        return "", [], []
    
    pages = [page for _, page in non_empty_pages_with_indices]
    page_indices = [i for i, _ in non_empty_pages_with_indices]
    
    result_text = ftfy.fix_text(pages[0], config=ftfy_config)
    page_offsets = [len(result_text)]
    for i in range(1, len(pages)):
        current_page = ftfy.fix_text(pages[i], config=ftfy_config)
        
        # Check if previous page ends with a dash (hyphenation) and the character before the dash is alphabetic
        if result_text.endswith(tuple(DASHES)) and len(result_text) >= 2 and result_text[-2].isalpha():
            # Remove the trailing dash and join directly
            result_text = result_text[:-1] + current_page
        else:
            # Add double newline separator
            result_text += "\n\n" + current_page
        page_offsets.append(len(result_text))

    return result_text, page_offsets, page_indices

class Normalize(PipelineStep):
    def __init__(self, is_from_docling: bool = False):
        self.is_from_docling = is_from_docling
        super().__init__()

    def run(self, data, rank: int = 0, world_size: int = 1):
        from postprocessing.normalize import clean_pdf_page, pages_to_text, remove_tags
        from datatrove.utils.media import iter_pages
        for document in data:
            if not self.is_from_docling:
                pages = [remove_tags(page, is_from_docling=self.is_from_docling) for page in iter_pages(document)]
            else:
                pages = [clean_pdf_page(remove_tags(page, is_from_docling=self.is_from_docling)) for page in iter_pages(document)]
            text, page_offsets, page_indices = pages_to_text(pages)
            document.text = text
            document.media[0].metadata["page_offsets"] = page_offsets
            # We add indicies because normalization can results in pages removal
            document.media[0].metadata["page_indices"] = page_indices
            yield document


if __name__ == "__main__":

    # Run test for fix_split_any_at_line_start
    example = fix_split_any_at_line_start("C O N S E I L  M U N I C I P A L Séance du 22 février 2016")
    assert example == "CONSEIL MUNICIPAL Séance du 22 février 2016", example

    example = fix_split_any_at_line_start("P ř i vyšší")
    assert example == "Při vyšší", example

    # Test for minimal word chars
    example = fix_split_any_at_line_start("1 x 9 x 8 * 9")
    assert example == "1 x 9 x 8 * 9", example