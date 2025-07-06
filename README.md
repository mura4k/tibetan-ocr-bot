# Tibetan OCR Telegram Bot

A Telegram bot for recognizing and transliterating classical Tibetan texts from archival images. Built for archive and research users with clarity, reliability, and batch support.

---

## ✨ Features

- 📥 Accepts batches of image files (JPEG, PNG, etc.)
- 🔤 OCR in Unicode or EWTS (transliteration) or both
- 🪪 Preserves original filenames in output
- 🧾 Bundles results into `.zip` for easy download
- ❌ Error summary for failed images
- 🧘 Commands: `/start`, `/done`, `/clear`, `/log`, `/status`
- 🧹 Auto-cleans sessions after 1 hour
- 📄 Configs read from:
  - `../models/ocr/Woodblock/model_config.json`
  - `../models/lines/config.json`
  - `../models/layout/photiv2/config.json`

---

## 🛠 Installation

```bash
git clone https://github.com/your-org/tibetan-ocr-bot.git
cd tibetan-ocr-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ⚙️ Configuration
Set your bot token via environment variable:

```bash
export TELEGRAM_TOKEN=your-bot-token
```
Ensure you have your model configs and .onnx files at these paths (or adjust them in code).