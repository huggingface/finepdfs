import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import PIL
from datasets import Features
from datasets import Image as Features_Image
from datasets import Sequence, Value
from docling.datamodel.base_models import ConversionStatus
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream
from pydantic import BaseModel, ConfigDict, Field, model_validator

from docling_eval.datamodels.types import EvaluationModality, PredictionFormats


class DatasetRecord(
    BaseModel
):  # TODO make predictionrecord class, factor prediction-related fields there.

    doc_id: str = Field(alias="document_id")
    doc_path: Optional[Path] = Field(alias="document_filepath", default=None)
    doc_hash: Optional[str] = Field(alias="document_filehash", default=None)

    ground_truth_doc: DoclingDocument = Field(alias="GroundTruthDocument")
    original: Optional[Union[DocumentStream, Path]] = Field(
        alias="BinaryDocument", default=None
    )
    # TODO add optional columns to store the SegmentedPage, both for GT and prediction

    ground_truth_page_images: List[PIL.Image.Image] = Field(
        alias="GroundTruthPageImages", default=[]
    )
    ground_truth_pictures: List[PIL.Image.Image] = Field(
        alias="GroundTruthPictures", default=[]
    )

    mime_type: str = Field(default="application/pdf")
    modalities: List[EvaluationModality] = Field(default=[])

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @classmethod
    def get_field_alias(cls, field_name: str) -> str:
        return cls.model_fields[field_name].alias or field_name

    @classmethod
    def features(cls):
        return Features(
            {
                cls.get_field_alias("doc_id"): Value("string"),
                cls.get_field_alias("doc_path"): Value("string"),
                cls.get_field_alias("doc_hash"): Value("string"),
                cls.get_field_alias("ground_truth_doc"): Value("string"),
                cls.get_field_alias("ground_truth_pictures"): Sequence(
                    Features_Image()
                ),
                cls.get_field_alias("ground_truth_page_images"): Sequence(
                    Features_Image()
                ),
                cls.get_field_alias("original"): Value("string"),
                cls.get_field_alias("mime_type"): Value("string"),
                cls.get_field_alias("modalities"): Sequence(Value("string")),
            }
        )

    def _extract_images(
        self,
        document: DoclingDocument,
        pictures_field_prefix: str,
        pages_field_prefix: str,
    ):

        pictures = []
        page_images = []

        # Save page images
        for img_no, picture in enumerate(document.pictures):
            if picture.image is not None:
                # img = picture.image.pil_image
                # pictures.append(to_pil(picture.image.uri))
                pictures.append(picture.image.pil_image)
                picture.image.uri = Path(f"{pictures_field_prefix}/{img_no}")

        # Save page images
        for page_no, page in document.pages.items():
            if page.image is not None:
                # img = page.image.pil_image
                # img.show()
                page_images.append(page.image.pil_image)
                page.image.uri = Path(f"{pages_field_prefix}/{page_no}")

        return pictures, page_images

    def as_record_dict(self):
        record = {
            self.get_field_alias("doc_id"): self.doc_id,
            self.get_field_alias("doc_path"): str(self.doc_path),
            self.get_field_alias("doc_hash"): self.doc_hash,
            self.get_field_alias("ground_truth_doc"): json.dumps(
                self.ground_truth_doc.export_to_dict()
            ),
            self.get_field_alias("ground_truth_pictures"): self.ground_truth_pictures,
            self.get_field_alias(
                "ground_truth_page_images"
            ): self.ground_truth_page_images,
            self.get_field_alias("mime_type"): self.mime_type,
            self.get_field_alias("modalities"): list(self.modalities),
        }
        if isinstance(self.original, Path):
            with self.original.open("rb") as f:
                record.update({self.get_field_alias("original"): f.read()})
        elif isinstance(self.original, DocumentStream):
            record.update(
                {self.get_field_alias("original"): self.original.stream.read()}
            )
        else:
            record.update({self.get_field_alias("original"): None})

        return record

    @model_validator(mode="after")
    def validate_images(self) -> "DatasetRecord":
        if not len(self.ground_truth_pictures) and not len(
            self.ground_truth_page_images
        ):
            pictures, page_images = self._extract_images(
                self.ground_truth_doc,
                pictures_field_prefix=self.get_field_alias("ground_truth_pictures"),
                pages_field_prefix=self.get_field_alias("ground_truth_page_images"),
            )

            self.ground_truth_page_images = page_images
            self.ground_truth_pictures = pictures

        return self

    @model_validator(mode="before")
    @classmethod
    def validate_record_dict(cls, data: dict):
        gt_doc_alias = cls.get_field_alias("ground_truth_doc")
        if gt_doc_alias in data and isinstance(data[gt_doc_alias], str):
            data[gt_doc_alias] = json.loads(data[gt_doc_alias])

        gt_page_img_alias = cls.get_field_alias("ground_truth_page_images")
        if gt_page_img_alias in data:
            for ix, item in enumerate(data[gt_page_img_alias]):
                if isinstance(item, dict):
                    data[gt_page_img_alias][ix] = Features_Image().decode_example(item)

        gt_pic_img_alias = cls.get_field_alias("ground_truth_pictures")
        if gt_pic_img_alias in data:
            for ix, item in enumerate(data[gt_pic_img_alias]):
                if isinstance(item, dict):
                    data[gt_pic_img_alias][ix] = Features_Image().decode_example(item)

        gt_binary = cls.get_field_alias("original")
        if gt_binary in data and isinstance(data[gt_binary], bytes):
            data[gt_binary] = DocumentStream(
                name="file", stream=BytesIO(data[gt_binary])
            )

        return data


class DatasetRecordWithPrediction(DatasetRecord):
    predictor_info: Dict = Field(alias="predictor_info", default={})
    status: ConversionStatus = Field(alias="status", default=ConversionStatus.PENDING)

    predicted_doc: Optional[DoclingDocument] = Field(
        alias="PredictedDocument", default=None
    )
    original_prediction: Optional[str] = None
    prediction_format: PredictionFormats  # some enum type

    predicted_page_images: List[PIL.Image.Image] = Field(
        alias="PredictionPageImages", default=[]
    )
    predicted_pictures: List[PIL.Image.Image] = Field(
        alias="PredictionPictures", default=[]
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @classmethod
    def features(cls):
        return {
            cls.get_field_alias("predictor_info"): Value("string"),
            cls.get_field_alias("status"): Value("string"),
            cls.get_field_alias("doc_id"): Value("string"),
            cls.get_field_alias("doc_path"): Value("string"),
            cls.get_field_alias("doc_hash"): Value("string"),
            cls.get_field_alias("ground_truth_doc"): Value("string"),
            cls.get_field_alias("ground_truth_pictures"): Sequence(Features_Image()),
            cls.get_field_alias("ground_truth_page_images"): Sequence(Features_Image()),
            cls.get_field_alias("predicted_doc"): Value("string"),
            cls.get_field_alias("predicted_pictures"): Sequence(Features_Image()),
            cls.get_field_alias("predicted_page_images"): Sequence(Features_Image()),
            cls.get_field_alias("original"): Value("string"),
            cls.get_field_alias("mime_type"): Value("string"),
            cls.get_field_alias("modalities"): Sequence(Value("string")),
            cls.get_field_alias("prediction_format"): Value("string"),
        }

    def as_record_dict(self):
        record = super().as_record_dict()
        record.update(
            {
                self.get_field_alias("prediction_format"): self.prediction_format.value,
            }
        )

        if self.predicted_doc is not None:
            record.update(
                {
                    self.get_field_alias("predicted_doc"): json.dumps(
                        self.predicted_doc.export_to_dict()
                    ),
                    self.get_field_alias("predicted_pictures"): self.predicted_pictures,
                    self.get_field_alias(
                        "predicted_page_images"
                    ): self.predicted_page_images,
                    self.get_field_alias("original_prediction"): (
                        self.original_prediction
                    ),
                    self.get_field_alias("status"): (self.status),
                }
            )

        return record

    @model_validator(mode="after")
    def validate_images(self) -> "DatasetRecordWithPrediction":
        # super().validate_images()

        if self.predicted_doc is not None:
            if not len(self.predicted_pictures) and not len(self.predicted_page_images):
                pictures, page_images = self._extract_images(
                    self.predicted_doc,
                    pictures_field_prefix=self.get_field_alias("predicted_pictures"),
                    pages_field_prefix=self.get_field_alias("predicted_page_images"),
                )

                self.predicted_page_images = page_images
                self.predicted_pictures = pictures

        return self

    @model_validator(mode="before")
    @classmethod
    def validate_prediction_record_dict(cls, data: dict):
        pred_doc_alias = cls.get_field_alias("predicted_doc")
        if pred_doc_alias in data and isinstance(data[pred_doc_alias], str):
            data[pred_doc_alias] = json.loads(data[pred_doc_alias])

        pred_page_img_alias = cls.get_field_alias("predicted_page_images")
        if pred_page_img_alias in data:
            for ix, item in enumerate(data[pred_page_img_alias]):
                if isinstance(item, dict):
                    data[pred_page_img_alias][ix] = Features_Image().decode_example(
                        item
                    )

        pred_pic_img_alias = cls.get_field_alias("predicted_pictures")
        if pred_pic_img_alias in data:
            for ix, item in enumerate(data[pred_pic_img_alias]):
                if isinstance(item, dict):
                    data[pred_pic_img_alias][ix] = Features_Image().decode_example(item)

        return data
