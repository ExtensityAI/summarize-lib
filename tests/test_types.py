import logging
import time

import pytest
from symai import Symbol

from src.hierarchical import HierarchicalSummary
from src.hierarchical_OLD import HierarchicalSummary as HierarchicalSummaryOld
from src.hierarchical_OLD_BASIC import \
    HierarchicalSummary as HierarchicalSummaryOldBasic
from src.types import DocumentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONTENT_TYPES = [
    "Interview",
    "Keynote",
    "Scientific Paper",
    "Report",
    "Book",
    "Presentation Slides"
]

@pytest.mark.interview
@pytest.mark.parametrize("file_path", ["testfiles/interview_transcript.pdf"])
def test_type_interview(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == DocumentType.INTERVIEW
    
    # Verify speakers are identified
    sym = Symbol(f"Does this summary identify different speakers and their key discussion points? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.keynote
@pytest.mark.parametrize("file_path", ["testfiles/keynote_presentation.pdf"])
def test_type_keynote(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == DocumentType.KEYNOTE
    
    # Verify speaker details and key messages
    sym = Symbol(f"Does this summary include speaker details, their expertise, and key messages? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.paper
@pytest.mark.parametrize("file_path", ["testfiles/symbolicai_no_refs.pdf"])
def test_type_scientific_paper(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == DocumentType.SCIENTIFIC_PAPER
    
    # Verify methodology and findings
    sym = Symbol(f"Does this summary include methodology details and research findings? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.report
@pytest.mark.parametrize("file_path", ["testfiles/google_report.pdf"])
def test_type_report(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == DocumentType.REPORT
    
    # Verify numerical results and statistics
    sym = Symbol(f"Does this summary include specific numerical results and statistics? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.book
@pytest.mark.parametrize("file_path", ["testfiles/book.pdf"])
def test_type_book(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == DocumentType.BOOK
    
    # Verify character descriptions and relationships
    sym = Symbol(f"Does this summary include character descriptions and their relationships? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.presentation
@pytest.mark.parametrize("file_path", ["testfiles/pitch_deck.pdf"])
def test_type_presentation_slides(file_path):
    summarizer = HierarchicalSummary(
        file_link=file_path
    )
    summary, _ = summarizer()
    
    # Verify content type
    assert summary.type == DocumentType.PRESENTATION_SLIDES
    
    # Verify core idea and value proposition
    sym = Symbol(f"Does this summary include the core idea and value proposition? Return yes or no.\n{summary.summary}").interpret()
    assert "yes" in sym.lower() or "true" in sym.lower()

@pytest.mark.performance
@pytest.mark.parametrize("file_path", ["testfiles/symbolicai_no_refs.pdf"])
def test_summary_performance_comparison(file_path):
    # Test new summarizer
    start_time = time.time()
    summarizer_new = HierarchicalSummary(
        file_link=file_path
    )
    summary_new, _ = summarizer_new()
    elapsed_time_new = time.time() - start_time
    
    # Performance assertions for new
    assert elapsed_time_new < 300
    print(f"New paper summarization took {elapsed_time_new:.2f} seconds")
    print(f"New paper summary: \n {summary_new.summary} \n {summary_new.facts}")

    # Test old summarizer 
    start_time = time.time()
    summarizer_old = HierarchicalSummaryOld(
        file_link=file_path
    )
    summary_old, _ = summarizer_old()
    elapsed_time_old = time.time() - start_time

    # Performance assertions for old
    assert elapsed_time_old < 300
    print(f"Old paper summarization took {elapsed_time_old:.2f} seconds")
    print(f"Old paper summary: \n {summary_old.summary} \n {summary_old.facts}")

    comparison = Symbol(f"Does the new summary include as detailed information as the old summary?"
                         + f"\nIs the new summary of the same or better calibre than the old summary? Return yes or no and explain why."
                         + f"\nNew summary: {summary_new}"
                         + f"\nOld summary: {summary_old}").interpret()
    
    assert "yes" in comparison.lower() or "true" in comparison.lower()
    print(comparison)

@pytest.mark.compare
@pytest.mark.parametrize("file_path", ["testfiles/symbolicai_no_refs.pdf"])
def test_summary_comparison(file_path):
    # Test new summarizer
    summarizer_new = HierarchicalSummary(
        file_link=file_path
    )
    summary_new, _ = summarizer_new()
    print(f"New paper summary: \n {summary_new.summary} \n {summary_new.facts}")

    # Test old summarizer
    summarizer_old = HierarchicalSummaryOldBasic(file_link=file_path)
    summary_old, _ = summarizer_old()
    print(f"Old paper summary: \n {summary_old.summary} \n {summary_old.facts}")

    comparison = Symbol(f"Does the new summary include as detailed information as the old summary?"
                         + f"\nIs the new summary of the same or better calibre than the old summary?"
                         + f"\nDoes the new summary include the title, authors, and publication details?"
                         + f"\nDoes the new summary include the main topic and scope?"
                         + f"\nDoes the new summary include key statements, contributions, main results, and important references?"
                         + f"\nDoes the new summary focus on methodology and findings?"
                         + f"\nReturn yes or no as an overall answer and then explain the answer to each question."
                         + f"\nNew summary: {summary_new}"
                         + f"\nOld summary: {summary_old}").interpret()
    
    assert "yes" in comparison.lower() or "true" in comparison.lower()
    print(comparison)