from datatrove.data import MediaType
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.io import DataFolderLike, DataFileLike
from typing import Callable, Literal
from pandas import Timestamp
from uuid import uuid4

class WarcIndexReprocess(BaseDiskReader):
    """Read WARC files and extract metadata for indexing purposes.
        
        This reader processes WARC files to extract metadata without reading the full content,
        making it suitable for creating indexes of WARC archives. Each WARC record becomes
        a separate document with metadata about the record.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        compression: the compression to use (default: "infer")
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
    """

    name = "🕷 Warc"
    _requires_dependencies = ["warcio", ("cchardet", "faust-cchardet"), ("magic", "python-magic")]

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        self.compression = compression
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )

    def read_file(self, filepath: str):
        from warcio.archiveiterator import ArchiveIterator

        with self.data_folder.open(filepath, "rb") as f:
            archive_iterator = ArchiveIterator(f)
            for ri, record in enumerate(archive_iterator):
                offset = archive_iterator.offset
                with self.track_time():
                    name = f.path[len("commoncrawl/"):]
                    extracted_data = process_record(record, offset, name)
                    if not extracted_data:
                        continue
                    document = self.get_document_from_dict(extracted_data, filepath, ri)
                    if not document:
                        continue
                
                yield document


def process_record(record: "ArcWarcRecord", offset: int, name: str) -> dict | None:
    """Process a WARC record to extract metadata for indexing.
    
    Args:
        record: The WARC record to process
        offset: The byte offset of the record in the WARC file
        name: The name/path of the WARC file
        
    Returns:
        A dictionary containing extracted metadata, or None if the record should be skipped.
        The returned dict includes:
        - text: Empty string (no content extracted for indexing)
        - id: WARC record ID
        - url: Target URI
        - fetch_status: HTTP status line
        - fetch_time: Date of the record
        - content_mime_type: Original MIME type from headers
        - content_mime_detected: Detected MIME type
        - content_truncated: Whether content was truncated (if >= 1MB)
        - warc_record_offset: Byte offset in the WARC file
        - warc_filename: Name of the WARC file
        - content_digest: Content digest hash if available
    """
    import magic

    # record type
    if record.rec_type not in ["response", "conversion", "resource"]:  # wet files have "conversion" type
        return

    # content type filtering
    content_bytes = record.content_stream().read()

    original_mime_type = record.content_type.split(";")[0]
    detected_mime_type = record.rec_headers.get("WARC-Identified-Payload-Type", None)
    if detected_mime_type is None:
        detected_mime_type = magic.from_buffer(content_bytes, mime=True)


    truncated = None
    # 1MB truncation
    if len(content_bytes) >= 1024 * 1024:
        truncated = "length"

    id_ = record.rec_headers["WARC-Record-ID"]
    url = record.rec_headers.get("WARC-Target-URI", None)
    date = record.rec_headers.get("WARC-Date", None)
    if not url:
        url = dict(record.rec_headers.headers)["uri"]
    if not date:
        date = dict(record.rec_headers.headers)["archive-date"]

    date = date.to_pydatetime() if isinstance(date, Timestamp) else date



    ret = {
        "text": "<no content>",
        "id": id_,
        "metadata": {
            "index_file": name,
        },
        "media": [{
            "id": id_,
            "url": url,
            "type": MediaType.DOCUMENT,
            "path": name,
            "offset": offset,
            "length": record.length,
            "metadata": {
                "fetch_time": date,
                "fetch_status": record.http_headers.statusline,
                "content_mime_type": original_mime_type,
                "content_mime_detected": detected_mime_type,
                "content_languages": None,
                "content_truncated": truncated,
                "content_digest": record.rec_headers.get("WARC-Payload-Digest", None),
            }
        }],
    }
    return ret
