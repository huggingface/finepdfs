import glob
from pathlib import Path
from typing import Dict

from docling_core.types.doc.document import (
    DoclingDocument,
    DocTagsDocument,
    DocTagsPage,
)


class TextFilesAdapter:
    def __init__(self, extension: str):
        r"""
        The file extension to load (e.g. "md")
        """
        self._extension = extension

    def __call__(self, input_path: Path) -> Dict[str, str]:
        r"""
        Read text files from a directory where each filename is <doc_id>.<extension>
        Produce a Dict[str, str] where the key is the doc_id and key is the file content
        """
        predictions: Dict[str, str] = {}

        fns = glob.glob(str(input_path / f"*.{self._extension}"))
        for fn in fns:
            doc_id = Path(fn).stem
            with open(fn, "r") as fd:
                predictions[doc_id] = fd.read()

        return predictions


class DocTagsFilesToDoclingDocumentAdapter:
    def __init__(self):
        r""" """
        self._doctag_extension = "dt"

    def __call__(self, input_path: Path) -> Dict[str, DoclingDocument]:
        r"""
        Read doctag files from the given directory and build DoclingDocument objects
        The doctag files must follow the naming convention <doc_id>.<extension>
        """
        predictions: Dict[str, DoclingDocument] = {}

        fns = glob.glob(str(input_path / f"*.{self._doctag_extension}"))
        for fn in fns:
            doc_id = Path(fn).stem
            with open(fn, "r") as fd:
                doctags = fd.read()
            doc = self._doctags_to_doc(doc_id, doctags)
            predictions[doc_id] = doc

        return predictions

    def _doctags_to_doc(self, doc_id, doctags: str) -> DoclingDocument:
        r"""
        Convert string with doctags to DoclingDocument
        """
        doctags_page = DocTagsPage(tokens=doctags)
        doctags_doc = DocTagsDocument(pages=[doctags_page])
        doc = DoclingDocument(name=doc_id)
        doc.load_from_doctags(doctags_doc)
        return doc
