"""Models for the labels types."""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel


class DocLinkLabel(str, Enum):
    """DocLinkLabel."""

    READING_ORDER = "reading_order"

    TO_CAPTION = "to_caption"
    TO_FOOTNOTE = "to_footnote"
    TO_VALUE = "to_value"

    MERGE = "merge"
    GROUP = "group"

    def __str__(self):
        """Get string value."""
        return str(self.value)

    @staticmethod
    def get_color(label: "DocLinkLabel") -> Tuple[int, int, int]:
        """Return the RGB color associated with a given label."""
        color_map = {
            DocLinkLabel.READING_ORDER: (255, 0, 0),
            DocLinkLabel.TO_CAPTION: (0, 255, 0),
            DocLinkLabel.TO_FOOTNOTE: (0, 255, 0),
            DocLinkLabel.TO_VALUE: (0, 255, 0),
            DocLinkLabel.MERGE: (255, 0, 255),
            DocLinkLabel.GROUP: (255, 255, 0),
        }
        return color_map[label]


class TableComponentLabel(str, Enum):
    """TableComponentLabel."""

    TABLE_ROW = "table_row"  # the most atomic row
    TABLE_COL = "table_column"  # the most atomic col
    TABLE_GROUP = (
        "table_group"  # table-cell group with at least 1 row- or col-span above 1
    )

    def __str__(self):
        """Get string value."""
        return str(self.value)

    @staticmethod
    def get_color(label: "TableComponentLabel") -> Tuple[int, int, int]:
        """Return the RGB color associated with a given label."""
        color_map = {
            TableComponentLabel.TABLE_ROW: (255, 0, 0),
            TableComponentLabel.TABLE_COL: (0, 255, 0),
            TableComponentLabel.TABLE_GROUP: (0, 0, 255),
        }
        return color_map[label]


class BenchMarkDirs(BaseModel):

    source_dir: Path = Path("")
    target_dir: Path = Path("")

    tasks_dir: Path = Path("")
    bins_dir: Path = Path("")

    annotations_dir: Path = Path("")
    annotations_zip_dir: Path = Path("")
    annotations_xml_dir: Path = Path("")

    dataset_dir: Path = Path("")
    dataset_train_dir: Path = Path("")
    dataset_test_dir: Path = Path("")

    page_imgs_dir: Path = Path("")

    json_true_dir: Path = Path("")
    json_pred_dir: Path = Path("")
    json_anno_dir: Path = Path("")

    html_anno_dir: Path = Path("")
    html_comp_dir: Path = Path("")

    project_desc_file: Path = Path("")
    overview_file: Path = Path("")

    def set_up_directory_structure(self, source: Path, target: Path) -> "BenchMarkDirs":

        assert os.path.exists(str(source)), f"os.path.exists({source})"

        self.source_dir = source
        self.target_dir = target

        self.tasks_dir = self.target_dir / "cvat_tasks"

        self.annotations_dir = self.target_dir / "cvat_annotations"
        self.annotations_zip_dir = self.annotations_dir / "zips"
        self.annotations_xml_dir = self.annotations_dir / "xmls"

        self.project_desc_file = self.target_dir / "cvat_description.json"
        self.overview_file = self.target_dir / "cvat_overview.json"

        self.bins_dir = self.target_dir / "cvat_bins"

        self.dataset_dir = self.target_dir / "datasets"
        self.dataset_train_dir = self.dataset_dir / "test"
        self.dataset_test_dir = self.dataset_dir / "train"

        self.page_imgs_dir = self.target_dir / "page_imgs"

        self.json_true_dir = self.target_dir / "json_groundtruth"
        self.json_pred_dir = self.target_dir / "json_predictions"
        self.json_anno_dir = self.target_dir / "json_annotations"

        self.html_anno_dir = self.target_dir / "html_annotations"
        self.html_comp_dir = self.target_dir / "html_comparisons"

        for _ in [
            self.target_dir,
            self.tasks_dir,
            self.bins_dir,
            self.annotations_dir,
            self.annotations_zip_dir,
            self.annotations_xml_dir,
            self.dataset_dir,
            self.dataset_train_dir,
            self.dataset_test_dir,
            self.page_imgs_dir,
            self.json_true_dir,
            self.json_pred_dir,
            self.json_anno_dir,
            self.html_anno_dir,
            self.html_comp_dir,
        ]:
            os.makedirs(_, exist_ok=True)

        return self


class AnnotationBBox(BaseModel):

    bbox_id: int
    bbox: BoundingBox
    label: DocItemLabel

    def to_cvat(self) -> str:
        return f'<box label="{self.label.value}" source="docling" occluded="0" xtl="{self.bbox.l}" ytl="{self.bbox.t}" xbr="{self.bbox.r}" ybr="{self.bbox.b}" z_order="{self.bbox_id}"></box>'


class AnnotationLine(BaseModel):

    line: List[AnnotationBBox]
    label: DocLinkLabel


class AnnotatedDoc(BaseModel):
    """Document representation for CVAT annotation."""

    mime_type: str = ""
    document_file: Path = Path("")  # Single document file path
    bin_file: Path = Path("")
    doc_hash: str = ""
    doc_name: str = ""


class AnnotatedImage(BaseModel):
    """Image representation for CVAT annotation."""

    mime_type: str = ""
    document_file: Path = Path("")  # Single document file path
    bin_file: Path = Path("")
    bucket_dir: Path = Path("")
    doc_hash: str = ""
    doc_name: str = ""
    img_id: int = -1
    img_w: int = -1
    img_h: int = -1
    img_file: Path = Path("")
    page_nos: List[int] = []
    page_img_files: List[Path] = []
    bbox_annotations: List[AnnotationBBox] = []  # Renamed from pred_boxes
    line_annotations: List[AnnotationLine] = []  # Renamed from pred_lines

    def to_cvat(self, lines: bool = False) -> str:
        """
        Generate CVAT XML annotation for this image.

        Args:
            lines: Whether to include line annotations (default: False)

        Returns:
            String containing CVAT XML for this image
        """
        tmp = [
            f'<image id="{self.img_id}" name="{os.path.basename(self.img_file)}" width="{self.img_w}" height="{self.img_h}">'
        ]

        # Add bounding box annotations
        for item_id, item in enumerate(self.bbox_annotations):
            tmp.append(item.to_cvat())

        tmp.append("</image>")

        return "\n".join(tmp)


class AnnotationOverview(BaseModel):

    doc_annotations: List[AnnotatedDoc] = []
    img_annotations: Dict[str, AnnotatedImage] = {}

    def export_to_dict(
        self,
        mode: str = "json",
        by_alias: bool = True,
        exclude_none: bool = True,
    ) -> Dict:
        """Export to dict."""
        return self.model_dump(mode=mode, by_alias=by_alias, exclude_none=exclude_none)

    def save_as_json(self, filename: Path, indent: int = 2):
        """Save as json."""
        out = self.export_to_dict()
        with open(filename, "w", encoding="utf-8") as fw:
            json.dump(out, fw, indent=indent)

    @classmethod
    def load_from_json(cls, filename: Path) -> "AnnotationOverview":
        """load_from_json."""
        with open(filename, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
