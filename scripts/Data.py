from uuid import UUID
from enum import Enum
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, List, Tuple

class OpStatus(Enum):
    SUCCESS = 0
    FAILED = 1


class Encoding(Enum):
    Unicode = 0
    Wylie = 1

class CharsetEncoder(Enum):
    Wylie = 0
    Stack = 1


class LineMode(Enum):
    Line = 0
    Layout = 1


class LineMerge(Enum):
    Merge = 0
    Stack = 1


class LineSorting(Enum):
    Threshold = 0
    Peaks = 1

class OCRArchitecture(Enum):
    Easter2 = 0
    CRNN = 1


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class Line:
    contour: npt.NDArray
    bbox: BBox
    center: Tuple[int, int]


@dataclass
class OCRLine:
    text: str
    encoding: Encoding

@dataclass
class LayoutData:
    image: npt.NDArray
    rotation: float
    images: List[BBox]
    text_bboxes: List[BBox]
    lines: List[Line]
    captions: List[BBox]
    margins: List[BBox]
    predictions: Dict[str, npt.NDArray]



@dataclass
class OCRData:
    guid: UUID
    image_path: str
    image_name: str
    ocr_lines: List[OCRLine] | None
    lines: List[Line] | None
    preview: npt.NDArray | None
    angle: float


@dataclass
class LineDetectionConfig:
    model_file: str
    patch_size: int



@dataclass
class LayoutDetectionConfig:
    model_file: str
    patch_size: int
    classes: List[str]


@dataclass
class OCRModelConfig:
    model_file: str
    architecture: OCRArchitecture
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    encoder: CharsetEncoder
    charset: List[str]
    add_blank: bool
    version: str


@dataclass
class LineDataResult:
    guid: UUID
    lines: List[Line]


@dataclass
class OCRLineUpdate:
    ocr_line: OCRLine

@dataclass
class OCRLineEncodingUpdate:
    ocr_lines: List[OCRLine]

@dataclass
class OCResult:
    mask: npt.NDArray
    lines: List[Line]
    text: List[OCRLine]
    angle: float

@dataclass
class OCRSample:
    cnt: int
    name: str
    result: OCResult


@dataclass
class OCRModel:
    name: str
    path: str
    config: OCRModelConfig

@dataclass
class OCRSettings:
    line_mode: LineMode
    line_merge: LineMerge
    line_sorting: LineSorting
    k_factor: float
    bbox_tolerance: float
    dewarping: bool
    merge_lines: bool
    output_encoding: Encoding