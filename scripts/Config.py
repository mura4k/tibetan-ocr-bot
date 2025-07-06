from Data import (
    Encoding,
    LineMode,
    LineMerge,
    LineSorting,
    CharsetEncoder,
    OCRArchitecture
)

"""
Mappings for each data type
"""

COLOR_DICT = {
    "background": "0, 0, 0",
    "image": "45, 255, 0",
    "text": "255, 243, 0",
    "margin": "0, 0, 255",
    "caption": "255, 100, 243",
    "table": "0, 255, 0",
    "pagenr": "0, 100, 15",
    "header": "255, 0, 0",
    "footer": "255, 255, 100",
    "line": "0, 100, 255"
}


ENCODINGS = {
    "unicode": Encoding.Unicode,
    "wylie": Encoding.Wylie
}

CHARSETENCODER = {
    "wylie": CharsetEncoder.Wylie,
    "stack": CharsetEncoder.Stack
}

OCRARCHITECTURE = {
    "Easter2": OCRArchitecture.Easter2,
    "CRNN": OCRArchitecture.CRNN
}


LINE_MODES = {
    "line": LineMode.Line,
    "layout": LineMode.Layout
}

LINE_MERGE = {
    "merge": LineMerge.Merge,
    "stack": LineMerge.Stack
}

LINE_SORTING = {
    "threshold": LineSorting.Threshold,
    "peaks": LineSorting.Peaks
}
