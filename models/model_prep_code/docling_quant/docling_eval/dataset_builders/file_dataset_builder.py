import logging
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

from docling_core.types import DoclingDocument
from docling_core.types.doc import ImageRef, PageItem, Size
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns
from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
from docling_eval.utils.utils import (
    add_pages_to_true_doc,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
)

# Get logger
_log = logging.getLogger(__name__)


class FileDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    File dataset builder implementing the base dataset builder interface.

    This builder processes a folder of PDFs or image files and creates a plain
    ground-truth dataset without annotations.
    """

    def __init__(
        self,
        dataset_source: Path,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
        file_extensions: List[str] = [
            "pdf",
            "tif",
            "tiff",
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "gif",
        ],
    ):
        """
        Initialize the File dataset builder.

        Args:
            dataset_source: Folder where data files reside
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="FileDataset",
            dataset_source=dataset_source,  # Local Path to dataset
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )
        self.file_extensions = file_extensions
        self.must_retrieve = False

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Iterate through the dataset and yield DatasetRecord objects.

        Yields:
            DatasetRecord objects
        """

        assert isinstance(self.dataset_source, Path)

        files: List[Path] = []

        for ext in self.file_extensions:
            files.extend(self.dataset_source.glob(f"*.{ext}"))
            files.extend(self.dataset_source.glob(f"*.{ext.upper()}"))
        files.sort()

        # Apply index range
        begin, end = self.get_effective_indices(len(files))
        selected_filenames = files[begin:end]

        # Log stats
        self.log_dataset_stats(len(files), len(selected_filenames))
        _log.info(f"Processing File dataset with {len(selected_filenames)} files")

        for filename in tqdm(
            selected_filenames,
            desc="Processing files for DP-Bench",
            ncols=128,
        ):
            mime_type, _ = mimetypes.guess_type(filename)

            # Create the ground truth Document
            true_doc = DoclingDocument(name=f"{filename}")
            if mime_type == "application/pdf":
                true_doc, _ = add_pages_to_true_doc(
                    pdf_path=filename, true_doc=true_doc, image_scale=2.0
                )
            elif mime_type is not None and mime_type.startswith("image/"):
                image: Image.Image = Image.open(filename)
                image = image.convert("RGB")
                image_ref = ImageRef(
                    mimetype="image/png",
                    dpi=72,
                    size=Size(width=image.width, height=image.height),
                    uri=from_pil_to_base64uri(image),
                )
                page_item = PageItem(
                    page_no=1,
                    size=Size(width=float(image.width), height=float(image.height)),
                    image=image_ref,
                )

                true_doc.pages[1] = page_item
            else:
                raise ValueError(
                    f"{filename} was not recognized as a supported type, aborting."
                )

            # Extract images from the ground truth document
            true_doc, true_pictures, true_page_images = extract_images(
                document=true_doc,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
            )

            # Get PDF as binary data
            pdf_bytes = get_binary(filename)
            pdf_stream = DocumentStream(name=filename.name, stream=BytesIO(pdf_bytes))

            # Create dataset record
            record = DatasetRecord(
                doc_id=str(filename.name),
                doc_hash=get_binhash(pdf_bytes),
                ground_truth_doc=true_doc,
                ground_truth_pictures=true_pictures,
                ground_truth_page_images=true_page_images,
                original=pdf_stream,
                mime_type=mime_type,
            )

            yield record
