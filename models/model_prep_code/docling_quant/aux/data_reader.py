import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from pathlib import Path
from docling.models.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.datamodel.document import ConversionResult
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from PIL import Image
from transformers.models.rt_detr import RTDetrImageProcessor

class ObjectDetectionDataReader(CalibrationDataReader):
    def __init__(self, data_folder: str):
        self.batch_id = 0
        self.input_folder = Path(data_folder)

        if not self.input_folder.is_dir():
            raise RuntimeError(
                f"Can't find input data directory: {str(self.input_folder)}"
            )
        self.files = list(self.input_folder.glob("*.png"))
        self._image_processor = RTDetrImageProcessor.from_json_file("model_artifacts/layout/model_artifacts/layout/preprocessor_config.json")

    def get_next(self):
        if self.batch_id >= len(self.files):
            return None
        file = self.files[self.batch_id]
        # Load image with PIL
        orig_img = Image.open(file)
        resize_size = {"height": 640, "width": 640}
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        pixel_values = self._image_processor(page_img, return_tensors="np", size=resize_size)["pixel_values"]
        self.batch_id += 1
        target_sizes = np.array([orig_img.size[::-1]], dtype=np.int64)
        return {"pixel_values": pixel_values, "target_sizes": target_sizes}

    def rewind(self):
        self.batch_id = 0

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        """Makes the class iterable. Resets the batch counter."""
        self.rewind()
        return self

    def __next__(self):
        """Returns the next item for iteration."""
        next_item = self.get_next()
        if next_item is None:
            raise StopIteration
        return next_item
