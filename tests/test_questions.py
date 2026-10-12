from __future__ import annotations

import fitz
import pytest

from pdfcropper.analysis import ExtractedWord, analyze_pdf_document
from pdfcropper.layout import LayoutRegion, PageLayout
from pdfcropper.questions import QuestionDetectionConfig, analyze_questions


def _word(text, x, y, width=None, height=12, *, block=0, line=0, word=0, page=3):
    width = width if width is not None else max(8, len(text) * 6)
    return ExtractedWord(text, fitz.Rect(x, y, x + width, y + height), page, block, line, word)


def _layout(lines, *, page_index=3, region_name="single", width=500, height=700, metadata=False, kind="content"):
    words = []
    for line_index, (text, x, y) in enumerate(lines):
        for word_index, token in enumerate(text.split()):
            words.append(
                _word(
                    token,
                    x + word_index * 9,
                    y,
                    block=1 if metadata else 0,
                    line=line_index + 1 if metadata else 0,
                    word=word_index,
                    page=page_index,
                )
            )
    region = LayoutRegion(region_name, fitz.Rect(0, 0, width, height), words, kind=kind)
    return PageLayout(page_index, fitz.Rect(0, 0, width, height), [region])


@pytest.mark.parametrize(
    ("lines", "expected_numbers"),
    [
        ([("1. Find the value of x", 30, 40), ("2. Solve the equation", 30, 80)], [1, 2]),
        ([("1) Calculate the area", 30, 40), ("2: Explain the result", 30, 80)], [1, 2]),
        ([("Q.1 Find the determinant", 30, 40), ("Q1 Solve for x", 30, 80)], [1, 1]),
        ([("Question 1: Describe the graph", 30, 40), ("12. Calculate the total", 30, 80)], [1, 12]),
    ],
)
def test_recognizes_common_numbering_formats(lines, expected_numbers):
    result = analyze_questions(_layout(lines))

    assert len(result.candidates) == 2
    assert [candidate.number for candidate in result.candidates] == expected_numbers


def test_multi_digit_number_and_page_index_are_preserved():
    result = analyze_questions(_layout([("12. Calculate the total", 30, 40)]))

    assert result.candidates[0].number == 12
    assert result.candidates[0].page_index == 3
    assert result.candidates[0].order == 1


def test_candidate_label_preserves_the_detected_marker_style():
    results = [
        analyze_questions(_layout([(marker, 30, 40)])).candidates[0].label
        for marker in (
            "1. Find x", "1) Find x", "1: Find x", "Q.1 Find x", "Q1 Find x", "Question 1: Find x"
        )
    ]

    assert results == ["1.", "1)", "1:", "Q.1", "Q1", "Question 1:"]


def test_two_columns_keep_independent_numbering_and_region_order():
    left = LayoutRegion("left", fitz.Rect(0, 0, 240, 600), [
        _word("1.", 20, 40), _word("Find", 40, 40), _word("x", 70, 40),
        _word("2.", 20, 100), _word("Solve", 40, 100), _word("y", 75, 100),
    ])
    right = LayoutRegion("right", fitz.Rect(260, 0, 500, 600), [
        _word("1.", 280, 40), _word("Find", 300, 40), _word("z", 330, 40),
        _word("2.", 280, 100), _word("Show", 300, 100), _word("work", 335, 100),
    ])
    result = analyze_questions(PageLayout(3, fitz.Rect(0, 0, 500, 700), [left, right]))

    assert [(item.region_name, item.number) for item in result.candidates] == [
        ("left", 1), ("left", 2), ("right", 1), ("right", 2)
    ]
    assert [item.order for item in result.candidates] == [1, 2, 3, 4]


def test_left_question_never_absorbs_same_height_right_column_text():
    left = _layout([
        ("Q1 Find the value of x", 30, 100),
        ("continue working here", 30, 130),
        ("Q2 Solve the equation", 30, 200),
    ], region_name="left", kind="left", metadata=True).regions[0]
    right = _layout([
        ("Q6 Calculate the area", 320, 100),
        ("right column continuation", 320, 130),
        ("Q7 Explain the result", 320, 200),
    ], region_name="right", kind="right", metadata=True).regions[0]
    layout = PageLayout(3, fitz.Rect(0, 0, 500, 700), [left, right], split_x=300, split_detected=True)

    result = analyze_questions(layout)

    assert [(item.label, item.region_name) for item in result.candidates] == [
        ("Q1", "left"), ("Q2", "left"), ("Q6", "right"), ("Q7", "right")
    ]
    assert all(word.rect.x1 < 300 for word in result.candidates[0].words)
    assert all(word.rect.x0 >= 300 for word in result.candidates[2].words)


def test_bottom_of_one_column_does_not_merge_with_other_columns_first_question():
    left = _layout([
        ("Q4 Find the value", 30, 640),
        ("last line in the left column", 30, 675),
    ], region_name="left", kind="left", metadata=True).regions[0]
    right = _layout([
        ("Q1 Calculate the next value", 320, 655),
        ("right column continuation", 320, 690),
    ], region_name="right", kind="right", metadata=True).regions[0]
    layout = PageLayout(3, fitz.Rect(0, 0, 500, 720), [left, right], split_x=300, split_detected=True)

    result = analyze_questions(layout)

    left_candidate, right_candidate = result.candidates
    assert left_candidate.label == "Q4"
    assert right_candidate.label == "Q1"
    assert [line.text for line in left_candidate.lines] == ["Q4 Find the value", "last line in the left column"]
    assert [line.text for line in right_candidate.lines] == ["Q1 Calculate the next value", "right column continuation"]
    assert all(word.rect.x1 < 300 for word in left_candidate.words)
    assert all(word.rect.x0 >= 300 for word in right_candidate.words)


def test_subparts_and_options_stay_with_parent_in_their_column():
    left = _layout([
        ("Q1 Solve the equation", 30, 100),
        ("(a) find the first root", 48, 125),
        ("(b) find the second root", 48, 150),
        ("A) first option", 48, 175),
        ("B) second option", 48, 200),
    ], region_name="left", kind="left", metadata=True).regions[0]
    right = _layout([("Q2 Explain your reasoning", 320, 100)], region_name="right", kind="right", metadata=True).regions[0]
    layout = PageLayout(3, fitz.Rect(0, 0, 500, 700), [left, right], split_x=300, split_detected=True)

    result = analyze_questions(layout)

    assert [candidate.label for candidate in result.candidates] == ["Q1", "Q2"]
    assert [line.text for line in result.candidates[0].lines] == [
        "Q1 Solve the equation", "(a) find the first root", "(b) find the second root",
        "A) first option", "B) second option",
    ]
    assert all(word.rect.x1 < 300 for word in result.candidates[0].words)


def test_later_column_preamble_is_not_reported_as_before_first_question():
    left = _layout([("1. Find x", 30, 40)]).regions[0]
    right = _layout([("Section notes", 280, 20), ("1. Find y", 280, 60)], region_name="right").regions[0]
    result = analyze_questions(PageLayout(3, fitz.Rect(0, 0, 500, 700), [left, right]))

    assert [line.text for line in result.preamble_lines] == ["Section notes"]
    assert not result.content_preceding_first_question_retained


def test_question_continuation_subparts_and_options_stay_grouped():
    result = analyze_questions(_layout([
        ("1. Solve the following equation", 30, 40),
        ("show all of your working", 30, 60),
        ("(a) Find the first value", 50, 80),
        ("(b) Find the second value", 50, 100),
        ("(i) use substitution", 70, 120),
        ("(ii) check your result", 70, 140),
        ("A) first option", 50, 160),
        ("B) second option", 50, 180),
    ]))

    assert len(result.candidates) == 1
    assert len(result.candidates[0].lines) == 8
    assert "(ii)" in [word.text for word in result.candidates[0].words]
    assert not result.ambiguous_lines


def test_indented_numbered_answer_choices_are_ambiguous_not_new_questions():
    result = analyze_questions(_layout([
        ("1. Which answer is correct?", 30, 40),
        ("1) first answer", 55, 65),
        ("2) second answer", 55, 85),
    ]))

    assert len(result.candidates) == 1
    assert len(result.candidates[0].lines) == 3
    assert len(result.ambiguous_lines) == 2


def test_equations_marks_and_standalone_page_numbers_are_not_questions():
    result = analyze_questions(_layout([
        ("2024", 250, 20),
        ("5 marks", 30, 40),
        ("1. 2 + 2 = 4", 30, 80),
        ("The answer is 12", 30, 120),
    ]))

    assert result.candidates == []
    assert result.no_reliable_starts
    assert len(result.ambiguous_lines) == 3
    assert len(result.preamble_lines) == 4


def test_bare_numbers_require_sequence_context():
    result = analyze_questions(_layout([
        ("1 Find the value of x", 30, 40),
        ("2 Solve the equation", 30, 80),
        ("5 Calculate the total", 30, 120),
    ]))

    assert [candidate.number for candidate in result.candidates] == [1, 2]
    assert any("sequence context" in item.reason for item in result.ambiguous_lines)
    assert result.no_reliable_starts


def test_explicit_question_prefix_accepts_split_math_prompt_lead_ins():
    result = analyze_questions(_layout([
        ("Q22 If", 30, 40),
        ("f(x) = x + 1", 30, 60),
        ("Q23 Let", 30, 100),
        ("g(x) = f(x) + 2", 30, 120),
    ]))

    assert [candidate.label for candidate in result.candidates] == ["Q22", "Q23"]
    assert [len(candidate.lines) for candidate in result.candidates] == [2, 2]


def test_split_column_margin_uses_text_inset_without_absorbing_other_column():
    left = _layout([
        ("Q24 Video Solution:", 29, 60),
        ("continued solution text", 29, 80),
    ], region_name="left", kind="left", width=152, metadata=True).regions[0]
    right = _layout([
        ("stray continuation", 226, 40),
        ("Q25 Video Solution:", 320, 60),
        ("right solution text", 320, 80),
    ], region_name="right", kind="right", width=350, metadata=True).regions[0]
    layout = PageLayout(3, fitz.Rect(0, 0, 500, 700), [left, right], split_x=181, split_detected=True)

    result = analyze_questions(layout)

    assert [(candidate.label, candidate.region_name) for candidate in result.candidates] == [
        ("Q24", "left"), ("Q25", "right")
    ]
    assert all(word.rect.x1 < 181 for word in result.candidates[0].words)
    assert all(word.rect.x0 >= 181 for word in result.candidates[1].words)


def test_content_before_first_question_is_retained_as_preamble():
    result = analyze_questions(_layout([
        ("Instructions for this section", 30, 20),
        ("Use exact values", 30, 40),
        ("1. Find the value of x", 30, 80),
    ]))

    assert [line.text for line in result.preamble_lines] == ["Instructions for this section", "Use exact values"]
    assert result.content_preceding_first_question_retained
    assert result.candidates[0].lines[0].text == "1. Find the value of x"


def test_numbered_question_can_restart_at_new_region():
    layout = PageLayout(3, fitz.Rect(0, 0, 500, 700), [
        _layout([("3. Find x", 30, 40)]).regions[0],
        _layout([("1. Find y", 30, 40)], region_name="section-2").regions[0],
    ])

    result = analyze_questions(layout)

    assert [(candidate.number, candidate.region_name) for candidate in result.candidates] == [
        (3, "single"), (1, "section-2")
    ]


def test_geometric_fallback_groups_ocr_words_without_line_metadata():
    result = analyze_questions(_layout([
        ("1. Find the value", 30, 40),
        ("of x", 30, 60),
        ("2. Solve the equation", 30, 100),
    ]))

    assert len(result.candidates) == 2
    assert [line.text for line in result.candidates[0].lines] == ["1. Find the value", "of x"]


def test_metadata_lines_are_preferred_when_consistent():
    result = analyze_questions(_layout([
        ("1. Find the value", 30, 40),
        ("of x", 30, 60),
        ("2. Solve the equation", 30, 100),
    ], metadata=True))

    assert len(result.candidates) == 2
    assert [line.text for line in result.candidates[0].lines] == ["1. Find the value", "of x"]


def test_bounding_box_is_exact_union_of_assigned_words():
    layout = _layout([
        ("1. Find x", 30, 40),
        ("on this line", 35, 60),
        ("2. Solve y", 30, 100),
    ])
    result = analyze_questions(layout)
    candidate = result.candidates[0]
    all_rects = [word.rect for word in candidate.words]

    assert candidate.rect == fitz.Rect(
        min(rect.x0 for rect in all_rects),
        min(rect.y0 for rect in all_rects),
        max(rect.x1 for rect in all_rects),
        max(rect.y1 for rect in all_rects),
    )
    assert candidate.rect.width >= 0 and candidate.rect.height >= 0


def test_custom_start_patterns_are_configurable():
    result = analyze_questions(
        _layout([("Problem 7 - Calculate the total", 30, 40)]),
        config=QuestionDetectionConfig(custom_marker_patterns=(r"^Problem\s+(?P<number>\d+)\s*[-:]",)),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].number == 7


def test_repeated_calls_have_deterministic_candidate_and_line_order():
    layout = _layout([("1. Find x", 30, 40), ("continued", 30, 60), ("2. Solve y", 30, 100)])

    first = analyze_questions(layout)
    second = analyze_questions(layout)

    assert [(candidate.number, [line.text for line in candidate.lines]) for candidate in first.candidates] == [
        (candidate.number, [line.text for line in candidate.lines]) for candidate in second.candidates
    ]


def test_real_pdf_text_analysis_flows_into_question_grouping(tmp_path):
    source = tmp_path / "questions.pdf"
    document = fitz.open()
    page = document.new_page(width=500, height=700)
    page.insert_text((40, 60), "1. Find the value of x")
    page.insert_text((40, 90), "show your working")
    page.insert_text((40, 130), "2) Solve the equation")
    document.save(source)
    document.close()

    analysis = analyze_pdf_document(source)
    region = LayoutRegion("single", analysis.pages[0].bounds, analysis.pages[0].words)
    result = analyze_questions(PageLayout(0, analysis.pages[0].bounds, [region]))

    assert [(candidate.number, len(candidate.lines)) for candidate in result.candidates] == [(1, 2), (2, 1)]


def test_invalid_custom_pattern_is_reported():
    with pytest.raises(ValueError, match="Invalid custom"):
        analyze_questions(_layout([("1. Find x", 30, 40)]), config=QuestionDetectionConfig(custom_marker_patterns=("[",)))