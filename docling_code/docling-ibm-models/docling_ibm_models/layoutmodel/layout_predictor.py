# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
import time
import os
from collections.abc import Iterable
from typing import Set, Union

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)

def post_process_object_detection_onnx(scores, labels, boxes, threshold):
    results = []
    for score, label, box in zip(scores, labels, boxes):
        results.append(
            {
                "scores": score[score > threshold],
                "labels": label[score > threshold],
                "boxes": box[score > threshold],
            }
        )
    return results

class LayoutPredictor:
    """
    Document layout prediction using safe tensors
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Set[str] = set(),
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        device: (Optional) device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'

        Raises
        ------
        FileNotFoundError when the model's torch file is missing
        """
        self._classes_map = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
            11: "Document Index",
            12: "Code",
            13: "Checkbox-Selected",
            14: "Checkbox-Unselected",
            15: "Form",
            16: "Key-Value Region",
        }

        # Blacklisted classes
        self._black_classes = blacklist_classes  # set(["Form", "Key-Value Region"])

        # Set basic params
        self._threshold = base_threshold  # Score threshold
        self._image_size = 640
        self._size = np.asarray([[self._image_size, self._image_size]], dtype=np.int64)

        # Set number of threads for CPU
        self._num_threads = num_threads

        path = os.environ.get("LAYOUT_VINO_PATH", "models/heron/heron_int8_quant.xml")
        self.init_vino_model(path, True)

        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "device": "cpu",
            "num_threads": self._num_threads,
            "image_size": self._image_size,
            "threshold": self._threshold,
        }
        return info

    def init_vino_model(self, model_path, preprocess_in_vino=False):
        from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
        from openvino import Type, Layout
        import openvino as ov
        # If the model_path is already ov.Model, use it
        if isinstance(model_path, ov.Model):
            _vino_model = model_path
        else:
            _vino_model = ov.Core().read_model(model_path)
        if preprocess_in_vino:
            vino_with_preproc = PrePostProcessor(_vino_model)
            vino_with_preproc.input("pixel_values").tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_shape([1, -1, -1, 3])
            vino_with_preproc.input("target_sizes").tensor().set_shape([1, 2])
            # resize then rescale then normalize then pad
            vino_with_preproc.input("pixel_values").preprocess().resize(ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, 640, 640) \
                .convert_element_type(Type.f32) \
                .scale(255) \
                .convert_layout(Layout("NCHW"))

            _vino_model = vino_with_preproc.build()

        self._preprocess_in_vino = preprocess_in_vino
        threads = os.environ.get("OMP_NUM_THREADS", -1)
        add_config = {}
        if threads != -1:
            add_config["INFERENCE_NUM_THREADS"] = threads

        cpu_runtime_cache_capacity = os.environ.get("CPU_RUNTIME_CACHE_CAPACITY", 3)
        # This setting seems to work well (doesn't introduce slowdowns) while keeping memory under 2GB
        os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"
        self._model = ov.compile_model(_vino_model, device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY", "CPU_RUNTIME_CACHE_CAPACITY": cpu_runtime_cache_capacity, **add_config})

        
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence", "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        resize = {"height": self._image_size, "width": self._image_size}
        if not self._preprocess_in_vino:
            raise NotImplementedError("Preprocessing without VINO is not implemented")
        else:
            inputs = {
                "pixel_values": np.array(page_img)[np.newaxis, ...],
            }
        target_sizes = np.array([page_img.size[::-1]])
        outputs = self._model((inputs["pixel_values"], target_sizes), share_outputs=True)
        results = post_process_object_detection_onnx(outputs[0], outputs[1], outputs[2], self._threshold)

        w, h = page_img.size

        result = results[0]
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score = float(score)

            label_id = int(label_id)
            label_str = self._classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self._black_classes:
                continue

            bbox_float = [float(b) for b in box]
            l = min(w, max(0, bbox_float[0]))
            t = min(h, max(0, bbox_float[1]))
            r = min(w, max(0, bbox_float[2]))
            b = min(h, max(0, bbox_float[3]))
            yield {
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "label": label_str,
                "confidence": score,
            }
