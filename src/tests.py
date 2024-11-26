import pytest
from symai import Symbol
from hierarchical import HierarchicalSummary

CONTENT_TYPES = [
    "Interview",
    "Keynote",
    "Scientific Paper",
    "Report",
    "Book",
    "Presentation Slides"
]

@pytest.mark.interview
@pytest.mark.parametrize("file_path", ["../testfiles/interview_transcript.pdf"])
def test_interview_summary(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path,
        content_types=CONTENT_TYPES
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == "Interview"
    
    # Verify speakers are identified
    sym = Symbol(f"Does this summary identify different speakers and their key discussion points? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.keynote
@pytest.mark.parametrize("file_path", ["../testfiles/keynote_presentation.pdf"])
def test_keynote_summary(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path,
        content_types=CONTENT_TYPES
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == "Keynote"
    
    # Verify speaker details and key messages
    sym = Symbol(f"Does this summary include speaker details, their expertise, and key messages? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.paper
@pytest.mark.parametrize("file_path", ["../testfiles/symbolicai_no_refs.pdf"])
def test_scientific_paper_summary(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path,
        content_types=CONTENT_TYPES
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == "Scientific Paper"
    
    # Verify methodology and findings
    sym = Symbol(f"Does this summary include methodology details and research findings? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.report
@pytest.mark.parametrize("file_path", ["../testfiles/google_report.pdf"])
def test_report_summary(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path,
        content_types=CONTENT_TYPES
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == "Report"
    
    # Verify numerical results and statistics
    sym = Symbol(f"Does this summary include specific numerical results and statistics? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.book
@pytest.mark.parametrize("file_path", ["../testfiles/book.pdf"])
def test_book_summary(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path,
        content_types=CONTENT_TYPES
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == "Book"
    
    # Verify character descriptions and relationships
    sym = Symbol(f"Does this summary include character descriptions and their relationships? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.presentation
@pytest.mark.parametrize("file_path", ["../testfiles/pitch_deck.pdf"])
def test_pitch_presentation_slides_summary(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path,
        content_types=CONTENT_TYPES
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == "Presentation Slides"
    
    # Verify core idea and value proposition
    sym = Symbol(f"Does this summary include the core idea and value proposition? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

# @pytest.mark.parametrize("file_path", ["path/to/presentation/files"])
# def test_motivational_presentation_slides_summary(file_path):
#     summarizer = HierarchicalSummary(
#         file_link=file_path,
#         content_types=CONTENT_TYPES
#     )
#     summary, _ = summarizer()

#     # Verify content type
#     assert summary.type == "Presentation Slides"
    
#     # Verify key messages and call-to-action
#     sym = Symbol("Does this summary include key messages and a call-to-action? {summary.summary}")
#     assert "yes" in sym().lower() or "true" in sym().lower()

# @pytest.mark.parametrize("file_path", ["path/to/presentation/files"])
# def test_results_presentation_slides_summary(file_path):
#     summarizer = HierarchicalSummary(
#         file_link=file_path,
#         content_types=CONTENT_TYPES
#     )
#     summary, _ = summarizer()

#     # Verify content type
#     assert summary.type == "Presentation Slides"
    
#     # Verify numerical results and achievements
#     sym = Symbol("Does this summary include numerical results and achievements? {summary.summary}")
#     assert "yes" in sym().lower() or "true" in sym().lower()
