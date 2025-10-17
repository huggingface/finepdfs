from datatrove.data import MediaType

MIME_TYPES = {
    "pdf": [
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "applications/vnd.pdf",
        "text/pdf",
    ]
}

def filter_non_pdf(data):
    from loguru import logger
    from urllib.parse import urlparse
    pdf_mime_types = MIME_TYPES["pdf"]
    def is_mime_pdf(data):
        if data.media[0].metadata["content_mime_detected"] in pdf_mime_types or data.media[0].metadata["content_mime_type"] in pdf_mime_types:
            return True
        return False

    if is_mime_pdf(data):
        return True

    # check url too
    parsed_url = urlparse(data.media[0].url)
    path = parsed_url.path.lower()
    if path.endswith(".pdf"):
        logger.info(f"Keeping {data.media[0].url} because it ends with .pdf, it has mime-type {data.media[0].metadata['content_mime_detected']}")
        return True
    return False

def index_adapter(self, data, file_path, _):
    from uuid import uuid4
    from pandas import Timestamp
    dato_uuid = str(uuid4())
    dato = {
        "text": "<no content>",
        "id": dato_uuid,
        "metadata": {
            "index_file": file_path,
        },
        "media": [{
            "id": dato_uuid,
            "url": data["url"],
            "type": MediaType.DOCUMENT,
            "path": data['warc_filename'],
            "offset": data["warc_record_offset"],
            "length": data["warc_record_length"],
            "metadata": {
                "fetch_time": data["fetch_time"],
                "fetch_status": data["fetch_status"],
                "content_mime_type": data["content_mime_type"],
                "content_mime_detected": data["content_mime_detected"],
                "content_languages": data.get("content_languages", None),
                "content_truncated": data["content_truncated"],
            }
        }],
    }
    if isinstance(dato["media"][0]["metadata"]["fetch_time"], Timestamp):
        # For some reason some crawls e.g 2019-47 have fetch_time as pandas Timestamp
        dato["media"][0]["metadata"]["fetch_time"] = dato["media"][0]["metadata"]["fetch_time"].to_pydatetime()
    return dato

def filter_non_truncated(data):
    pdf_mime_types = MIME_TYPES["pdf"]
    is_pdf_but_not_mime_type_pdf = (data.media[0].metadata["content_mime_detected"] not in pdf_mime_types and data.media[0].metadata["content_mime_type"] not in pdf_mime_types)
    if data.media[0].metadata["content_truncated"] != None and not is_pdf_but_not_mime_type_pdf:
        return True

    return False

def filter_non_pdf_ending_with_pdf(data):
    def is_mime_pdf(data):
        pdf_mime_types = MIME_TYPES["pdf"]
        if data.media[0].metadata["content_mime_detected"] in pdf_mime_types or data.media[0].metadata["content_mime_type"] in pdf_mime_types:
            return True
        return False
    
    return is_mime_pdf(data)