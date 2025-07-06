import logging
import os
import time
import zipfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
)

import pyewts
import cv2

# === IMPORT CUSTOM OCR MODULES ===
from Data import OCRModelConfig, Encoding
from Utils import read_ocr_model_config, read_line_model, read_layout_model
from Inference import OCRPipeline

# === CONFIG ===
OCR_CONFIG_PATH = Path("../models/ocr/Woodblock/model_config.json")
LINE_MODEL_CONFIG = Path("../models/lines/config.json")
TMP_DIR = Path("/tmp/ocr_bot")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === STATE ===
SESSION_TIMEOUT = 3600  # seconds
SESSION_KEY = "session"
TRANSLIT = pyewts.pyewts()

# === HELPERS ===
def get_pipeline():
    ocr_cfg = read_ocr_model_config(str(OCR_CONFIG_PATH))
    line_cfg = read_line_model(str(LINE_MODEL_CONFIG))
    return OCRPipeline(ocr_cfg, line_cfg)

def reset_session(context):
    context.user_data[SESSION_KEY] = {
        "images": [],
        "active": True,
        "start_time": time.time(),
        "log": []
    }

def cleanup_old_files():
    now = time.time()
    for f in TMP_DIR.glob("*"):
        if f.is_file() and now - f.stat().st_mtime > SESSION_TIMEOUT:
            try:
                f.unlink()
            except Exception:
                pass

def get_reply_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("📤 Done"), KeyboardButton("♻️ Clear")]
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        is_persistent=True
    )

def get_inline_output_options():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔤 Unicode", callback_data="ocr"),
            InlineKeyboardButton("🔁 EWTS", callback_data="translit"),
            InlineKeyboardButton("📝 Both", callback_data="both"),
        ]
    ])

def get_inline_postprocess_options(zip_path: Path):
    keyboard = [
        [InlineKeyboardButton("⬇️ Download All (.zip)", callback_data="download_zip")],
        [InlineKeyboardButton("🆕 New Session", callback_data="new_session")]
    ]
    return InlineKeyboardMarkup(keyboard)

# === COMMANDS ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_session(context)
    await update.message.reply_text(
        "📸 Send uncompressed Tibetan image documents. When ready, press Done.",
        reply_markup=get_reply_keyboard()
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop(SESSION_KEY, None)
    await update.message.reply_text("❌ Session cancelled.", reply_markup=get_reply_keyboard())

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_session(context)
    await update.message.reply_text("♻️ Session reset.", reply_markup=get_reply_keyboard())

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = context.user_data.get(SESSION_KEY, {})
    count = len(session.get("images", []))
    await update.message.reply_text(f"📦 You have {count} images in session.", reply_markup=get_reply_keyboard())

async def log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = context.user_data.get(SESSION_KEY)
    if not session:
        await update.message.reply_text("📭 No session log available.", reply_markup=get_reply_keyboard())
        return
    log_lines = [
        f"{img['orig_name']} - {img.get('status', 'N/A')}"
        for img in session["images"]
    ]
    await update.message.reply_text("\n".join(log_lines), reply_markup=get_reply_keyboard())

# === FILE HANDLING ===
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cleanup_old_files()
    session = context.user_data.get(SESSION_KEY)
    if not session or not session.get("active"):
        reset_session(context)
        session = context.user_data[SESSION_KEY]

    photo = update.message.photo[-1] if update.message.photo else update.message.document
    if update.message.document and not update.message.document.mime_type.startswith("image"):
        await update.message.reply_text("⚠️ Image files only.", reply_markup=get_reply_keyboard())
        return

    img_id = str(uuid4())
    img_path = TMP_DIR / f"{img_id}.jpg"
    file = await photo.get_file()
    await file.download_to_drive(str(img_path))
    orig_name = update.message.document.file_name if update.message.document else f"{img_id}.jpg"

    session["images"].append({"id_path": img_path, "orig_name": orig_name})
    await update.message.reply_text(f"📥 Image received: {orig_name}", reply_markup=get_reply_keyboard())

# === DONE HANDLER ===
async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = context.user_data.get(SESSION_KEY)
    if not session or not session.get("images"):
        await update.message.reply_text("⚠️ No images in session.", reply_markup=get_reply_keyboard())
        return

    filenames = [f"\u2022 {img['orig_name']}" for img in session["images"]]
    await update.message.reply_text(
        f"{len(filenames)} files received:\n" + "\n".join(filenames),
        reply_markup=get_inline_output_options()
    )

# === CALLBACK HANDLER for OCR/Translit buttons ===
async def output_choice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    session = context.user_data.get(SESSION_KEY)
    if not session or not session.get("images"):
        await query.edit_message_text("⚠️ Session expired or no images. Please start again.")
        return

    mode = query.data  # 'ocr', 'translit', 'both', 'download_zip', 'new_session'
    if mode in ("ocr", "translit", "both"):
        pipeline = get_pipeline()
        results = []
        errors = []
        for img_info in session["images"]:
            img_path = img_info["id_path"]
            orig_name = img_info["orig_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"{orig_name} — could not read image")
                continue

            status, result = pipeline.run_ocr(
                img, use_tps=False, merge_lines=True, target_encoding=Encoding.Unicode
            )
            if status.name != "SUCCESS":
                errors.append(f"{orig_name} — OCR error: {status.name}")
                continue

            _, _, ocr_lines, _ = result
            unicode_text = "\n".join(line.text for line in ocr_lines)

            if mode == "ocr":
                text_out = f"# OCR Result for {orig_name}\n\n" + unicode_text
            elif mode == "translit":
                text_out = f"# Transliteration (EWTS) for {orig_name}\n\n" + TRANSLIT.toWylie(unicode_text)
            else:  # both
                text_out = (
                    f"# OCR Result for {orig_name}\n\n{unicode_text}\n\n"
                    f"# Transliteration (EWTS) for {orig_name}\n\n{TRANSLIT.toWylie(unicode_text)}"
                )

            out_path = TMP_DIR / Path(orig_name).with_suffix(".txt")
            out_path.write_text(text_out, encoding="utf-8")
            results.append(out_path)

        # Store results paths in session for ZIP download
        session["results"] = results

        # Send errors summary if any
        if errors:
            await query.message.reply_text(
                "❌ Could not process:\n" + "\n".join(errors),
                reply_markup=get_reply_keyboard()
            )

        # Send all result files individually
        for file_path in results:
            await context.bot.send_document(chat_id=query.message.chat_id, document=str(file_path))

        # Offer ZIP download or new session buttons
        await query.edit_message_text(
            "✅ Processing done. Choose an option:",
            reply_markup=get_inline_postprocess_options(None)
        )

    elif mode == "download_zip":
        # Create ZIP archive
        results = session.get("results", [])
        if not results:
            await query.edit_message_text("❌ No files to zip. Run OCR first.")
            return

        zip_filename = TMP_DIR / f"ocr_results_{uuid4()}.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file_path in results:
                zipf.write(file_path, arcname=file_path.name)

        # Send ZIP file
        await context.bot.send_document(chat_id=query.message.chat_id, document=str(zip_filename))

        await query.edit_message_text(
            "✅ ZIP sent. Start a new session or /clear to reset.",
            reply_markup=get_reply_keyboard()
        )

    elif mode == "new_session":
        reset_session(context)
        await query.edit_message_text(
            "♻️ Session reset. Send images to start.",
            reply_markup=get_reply_keyboard()
        )
    else:
        await query.edit_message_text("⚠️ Unknown action.")


# === MAIN ===
def main():
    token = ""  # insert your bot token here
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("log", log))

    app.add_handler(MessageHandler(filters.Regex("^📤 Done$"), done))
    app.add_handler(MessageHandler(filters.Regex("^♻️ Clear$"), clear))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))

    app.add_handler(CallbackQueryHandler(output_choice_handler))

    app.run_polling()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
