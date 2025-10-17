import re
from datatrove.data import Document


def get_dump(file_path: str):
    crawl = re.search(r"crawl=(.*?)/", file_path)
    if crawl:
        return crawl.group(1)
    crawl = re.search(r"crawl-data/(.*?)/", file_path)
    if crawl:
        return crawl.group(1)
    raise ValueError(f"Could not get dump from file path: {file_path}")


def push_adapter(self, doc: Document):
    text = doc.text
    id_ = doc.id
    url = doc.media[0].url
    file_path = doc.metadata.get("file_path")
    token_count = doc.metadata.get("token_count")
    dump = get_dump(file_path) if file_path else None
    date = doc.media[0].metadata.get("fetch_time", None)
    offset = doc.media[0].offset

    language_bucket = doc.metadata.get("language_bucket")
    average_page_language = doc.metadata.get("best_page_average_language")
    average_page_language_score = doc.metadata.get("best_page_average_score")
    full_doc_lid = doc.metadata.get("language")
    full_doc_lid_score = doc.metadata.get("language_score")
    best_language_per_page = doc.metadata.get("best_page_languages")

    is_truncated = doc.metadata.get("is_truncated")
    extractor = "docling" if doc.metadata.get("is_docling") else "rolmOCR"
    page_ends = doc.media[0].metadata.get("page_offsets")

    return {
        "text": text,
        "id": id_,
        "dump": dump,
        "url": url,
        "date": date,
        "file_path": file_path,
        "offset": offset,
        "token_count": token_count,
        "language": language_bucket,
        "page_average_lid": average_page_language,
        "page_average_lid_score": average_page_language_score,
        "full_doc_lid": full_doc_lid,
        "full_doc_lid_score": full_doc_lid_score,
        "per_page_languages": best_language_per_page,
        "is_truncated": is_truncated,
        "extractor": extractor,
        "page_ends": page_ends,
    }








