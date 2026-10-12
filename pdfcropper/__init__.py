"""PDF question cropper prototype."""

from .analysis import (
    DocumentAnalysis,
    ExtractedWord,
    PageAnalysis,
    TextAvailability,
    analyze_pdf_document,
    page_indices_for_range,
)
from .layout import LayoutRegion, PageLayout, analyze_page_layout
from .ocr import ocr_page_analysis, ocr_page_words
from .pipeline import (
    PageLayoutMode,
    PageLayoutOptions,
    PageQuestionAnalysisResult,
    QuestionAnalysisRun,
    analyze_selected_pages,
)
from .questions import (
    AmbiguousQuestionLine,
    QuestionAnalysis,
    QuestionCandidate,
    QuestionDetectionConfig,
    QuestionLine,
    analyze_questions,
)
from .rectangles import (
    DEFAULT_RECTANGLE_PADDING,
    ProposedQuestionRectangle,
    RectangleGenerationResult,
    propose_document_rectangles,
    propose_question_rectangles,
)

__all__ = [
    "__version__",
    "DocumentAnalysis",
    "ExtractedWord",
    "PageAnalysis",
    "TextAvailability",
    "analyze_pdf_document",
    "page_indices_for_range",
    "LayoutRegion",
    "PageLayout",
    "analyze_page_layout",
    "AmbiguousQuestionLine",
    "QuestionAnalysis",
    "QuestionCandidate",
    "QuestionDetectionConfig",
    "QuestionLine",
    "analyze_questions",
    "DEFAULT_RECTANGLE_PADDING",
    "ProposedQuestionRectangle",
    "RectangleGenerationResult",
    "propose_document_rectangles",
    "propose_question_rectangles",
    "ocr_page_analysis",
    "ocr_page_words",
    "PageQuestionAnalysisResult",
    "QuestionAnalysisRun",
    "analyze_selected_pages",
    "PageLayoutMode",
    "PageLayoutOptions",
]
__version__ = "0.1.0"
