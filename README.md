# PDF Question Cropper

A lightweight desktop tool for turning a multi-question PDF into a set of single-question workspace pages.

## Features

- Import a PDF and view pages
- Navigate pages and control zoom/scroll
- Draw and edit rectangles around questions
- Reorder questions independently of page position
- Export one workspace page per selected question
- Keep a blank area for handwriting on each page

## Requirements

- Python 3.12+
- PyQt6
- PyMuPDF (`fitz`)
- reportlab
- pytest

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Notes

- The initial prototype focuses on import, manual question selection, question ordering, and export.
- OCR, automatic detection, and cloud features are intentionally not included yet.
- The exported PDF is designed to be opened in Okular or Charta for handwritten annotation.
