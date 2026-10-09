from __future__ import annotations

from .main_window import PdfCropperMainWindow
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication([])
    window = PdfCropperMainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
