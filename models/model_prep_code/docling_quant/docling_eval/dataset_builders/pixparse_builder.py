import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Optional

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    ImageRef,
    PageItem,
    ProvenanceItem,
    Size,
)
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns, EvaluationModality
from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval.utils.utils import (
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
)


class PixparseDatasetBuilder(BaseEvaluationDatasetBuilder):
    def __init__(
        self,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
        dataset_source: Optional[Path] = None,
    ):
        super().__init__(
            name="pixparse-idl",
            dataset_source=dataset_source or HFSource(repo_id="samiuc/pixparse-idl"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )
        self.split = split
        self.must_retrieve = True

    def _create_ground_truth_doc(
        self, doc_id: str, gt_data: Dict, image: Image.Image
    ) -> DoclingDocument:
        """Create a DoclingDocument from ground truth data and image file."""
        true_doc = DoclingDocument(name=doc_id)

        # Add page with image
        image_ref = ImageRef(
            mimetype=f"image/png",
            dpi=72,
            size=Size(width=float(image.width), height=float(image.height)),
            uri=from_pil_to_base64uri(image),
        )
        page_item = PageItem(
            page_no=1,
            size=Size(width=float(image.width), height=float(image.height)),
            image=image_ref,
        )
        true_doc.pages[1] = page_item

        for page_idx, page in enumerate(gt_data["pages"], 1):
            for text, bbox, _ in zip(page["text"], page["bbox"], page["score"]):
                bbox_obj = BoundingBox.from_tuple(
                    (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[0] + bbox[2]),
                        float(bbox[1] + bbox[3]),
                    ),
                    CoordOrigin.TOPLEFT,
                )
                prov = ProvenanceItem(
                    page_no=page_idx, bbox=bbox_obj, charspan=(0, len(text))
                )
                true_doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)

        return true_doc

    def iterate(self) -> Iterable[DatasetRecord]:
        if not self.retrieved and self.must_retrieve:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None

        output_dir = self.target
        output_dir.mkdir(parents=True, exist_ok=True)

        ground_truth_files = sorted(
            list(self.dataset_local_path.rglob("ground_truth.json"))
        )

        # Apply index range
        begin, end = self.get_effective_indices(len(ground_truth_files))
        ground_truth_files = ground_truth_files[begin:end]

        for gt_file in tqdm(
            ground_truth_files,
            desc="Processing files for PixParse IDL dataset",
            total=len(ground_truth_files),
            ncols=128,
        ):
            try:
                image_file = gt_file.parent / "original.tif"
                if not image_file.exists():
                    logging.info(f"Warning: No image file found for {gt_file}")
                    continue

                doc_id = gt_file.parent.name

                with open(gt_file, "r") as f:
                    gt_data = json.load(f)

                image_bytes = get_binary(image_file)

                image: Image.Image = Image.open(BytesIO(image_bytes))
                if image.mode not in (
                    "RGB",
                    "RGBA",
                ):
                    image = image.convert("RGB")

                true_doc = self._create_ground_truth_doc(doc_id, gt_data, image)

                # Extract images from the ground truth document
                true_doc, true_pictures, true_page_images = extract_images(
                    document=true_doc,
                    pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                    page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                )

                # Convert image to bytes for storage
                with BytesIO() as img_byte_stream:
                    image.save(img_byte_stream, format="PNG")
                    img_byte_stream.seek(0)
                    img_bytes = img_byte_stream.getvalue()

                image_stream = DocumentStream(
                    name=image_file.name, stream=BytesIO(image_bytes)
                )

                record = DatasetRecord(
                    doc_id=doc_id,
                    doc_hash=get_binhash(img_bytes),
                    ground_truth_doc=true_doc,
                    original=image_stream,
                    mime_type="image/png",
                    modalities=[
                        EvaluationModality.OCR,
                    ],
                    ground_truth_pictures=true_pictures,
                    ground_truth_page_images=true_page_images,
                )

                yield record

            except Exception as e:
                logging.error(f"Error processing {gt_file}: {str(e)}")
                raise
