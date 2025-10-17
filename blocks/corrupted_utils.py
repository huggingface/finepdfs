import re
import pymupdf
import logging
from blocks.loggers import LoggerStream
from collections import deque

regexes = [
    r"format error: object out of range",
    r"syntax error: no XObject subtype specified",
    r"syntax error: syntax error in content stream",
    r"object is not a stream",
    r"syntax error: syntax error in array",
    r"format error: cannot load page tree",
    r"syntax error: cannot parse indirect object",
]

corrupted_logs_regex = re.compile("|".join(regexes))


def check_is_corrupted_or_encrypted(pymupdf_doc: pymupdf.Document) -> bool:
    if pymupdf_doc.is_encrypted or pymupdf_doc.needs_pass:
        return True
    
    # Capture logs 
    logger_stream = LoggerStream(logging.getLogger("pymupdf"))
    with logger_stream as log_output:
        # Read all pages to trigger errors
        error_logs = ""
        try:
            deque((page.get_text() for page in pymupdf_doc.pages()), maxlen=0)
        except Exception as e:
            error_logs += str(e)
        logs = log_output.value().encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        if error_logs:
            logs += f"\n\nError logs: {error_logs}"
    
    if corrupted_logs_regex.search(logs):
        return True

    return False