#!/usr/bin/env python3
"""
main.py — batch OCR transliteration tool without argparse
"""
import json
from pathlib import Path
import cv2
import os

from Data import OCRModelConfig, Encoding
from Utils import read_ocr_model_config, read_line_model, read_layout_model
from Inference import OCRPipeline

# === CONFIGURE PATHS BELOW ===
OCR_CONFIG_PATH = Path("../models/woodblock_stacks/model_config.json")
LINE_MODEL_CONFIG = Path("../models/lines/config.json")
LAYOUT_MODEL_CONFIG = Path("../models/layout/config.json")
RAW_IMAGES_DIR = Path("../raw_images")
OUTPUT_DIR = Path("../translit")
# =============================


def main():
    # 1) load OCR model config
    ocr_cfg: OCRModelConfig = read_ocr_model_config(str(OCR_CONFIG_PATH))

    # 2) choose detection model: set use_layout = True to use layout detection
    use_layout = False
    if use_layout:
        line_cfg = read_layout_model(str(LAYOUT_MODEL_CONFIG))
    else:
        line_cfg = read_line_model(str(LINE_MODEL_CONFIG))

    # 3) build pipeline
    pipeline = OCRPipeline(ocr_cfg, line_cfg)

    # 4) ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5) process each image file in RAW_IMAGES_DIR
    for img_path in RAW_IMAGES_DIR.iterdir():
        if not img_path.is_file():
            continue
        # read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] cannot read {img_path.name}")
            continue

        # run OCR and transliteration
        status, result = pipeline.run_ocr(
            img,
            use_tps=False,
            merge_lines=True,
            target_encoding=Encoding.Wylie
        )
        if status.name != "SUCCESS":
            print(f"[ERROR] {img_path.name}: {result}")
            continue

        # extract lines and write output text
        _, _, ocr_lines, _ = result
        text_out = "\n".join([line.text for line in ocr_lines])
        out_file = OUTPUT_DIR / img_path.with_suffix('.txt').name
        out_file.write_text(text_out, encoding="utf-8")
        print(f"[OK] {img_path.name} → {out_file.name}")


if __name__ == "__main__":
    main()
