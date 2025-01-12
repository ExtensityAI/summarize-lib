import pytest
import time
from symai import Symbol
from hierarchical import HierarchicalSummary, Summary
from hierarchical_OLD import HierarchicalSummary as HierarchicalSummaryOld
from hierarchical_OLD import Summary as SummaryOld
from hierarchical_OLD_BASIC import HierarchicalSummary as HierarchicalSummaryOldBasic
from hierarchical_OLD_BASIC import Summary as SummaryOldBasic
import logging
from collections import defaultdict

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

# Define the context and expected keywords for validation
CONTEXT = {
    "book_content": """
        This book is about solving the Minecraft Challenge, a difficult test for AIs. 
        In this challenge, an AI system must solve difficult tasks in Minecraft. 
        We developed a method, Align-RUDDER, that is able to solve the challenge without human intervention. 
        Align-RUDDER was the first pure learning method that was able to mine a diamond, the most difficult task in this challenge.""",
    "audience_essential_message": "",
    "audience_target": """
        People interested in artificial intelligence but have no understanding of machine learning. 
        Explain basic concepts if it can't be expected for the average person to know them.""", 
    "audience_age_group": "18+",
    "book_interests": "Machine Learning, Artificial Intelligence, Minecraft",
    "book_narrative_style": "Narrative Nonfiction"
}

KEYWORDS = {
    "summary": [
        "Align-RUDDER", "Minecraft Challenge", "AI", "learning method", "diamond", "mine",
        "autonomous", "artificial intelligence", "performance", "achievement",
        "solved", "AI concepts", "pure learning method", "human intervention", 
        "limitations", "future work", "methodology", "results", "implementation", 
        "experiments", "training process", "success rate", "comparison", "baseline",
        "Minecraft", "challenge", "method", "system"
    ],
    "facts": [
        "Align-RUDDER", "Minecraft Challenge", "AI", "learning method", "diamond", "mine",
        "autonomous", "artificial intelligence", "performance", "achievement",
        "solved", "AI concepts", "pure learning method", "human intervention", 
        "limitations", "future work", "methodology", "results", "implementation", 
        "experiments", "training process", "success rate", "comparison", "baseline",
        "Minecraft", "challenge", "method", "system"
    ],
    "quotes": [
        "Align-RUDDER", "Minecraft Challenge", "AI", "learning method", "diamond", "mine",
        "autonomous", "artificial intelligence", "performance", "achievement",
        "solved", "AI concepts", "pure learning method", "human intervention", 
        "limitations", "future work", "methodology", "results", "implementation", 
        "experiments", "training process", "success rate", "comparison", "baseline",
        "Minecraft", "challenge", "method", "system"
    ]
}

# Initialize hit counters
hits = defaultdict(lambda: {"old": 0, "new": 0})

@pytest.mark.compare_facts
@pytest.mark.parametrize("file_path", ["/Users/ryang/Work/ExtensityAI/summarize-lib/testfiles/Align-RUDDER.pdf"])
def test_summary_comparison(file_path):
    num_runs = 10
    total_hits = defaultdict(lambda: {"old": 0, "new": 0})
    
    for run in range(num_runs):
        logger.info(f"Running comparison {run + 1}/{num_runs}")
        
        # Test new summarizer
        summarizer_new = HierarchicalSummary(
            file_link=file_path,
            content_types=True,
            user_prompt=str(CONTEXT)
        )
        summary_new, _ = summarizer_new()

        # Test old summarizer
        summarizer_old = HierarchicalSummaryOldBasic(file_link=file_path)
        summary_old, _ = summarizer_old()
        
        # Validate keywords and track hits for this run
        for field in KEYWORDS:
            if field == "quotes":
                continue
            total_hits[field]["old"] += validate_keywords(summary_old, field)
            total_hits[field]["new"] += validate_keywords(summary_new, field)

    # Calculate and print average hits
    print("\nAverage Keyword Validation Results:")
    total_improvement = 0
    total_fields = len(KEYWORDS)
    
    for field in KEYWORDS:
        if field == "quotes":
                continue
        avg_old = total_hits[field]["old"] / num_runs
        avg_new = total_hits[field]["new"] / num_runs
        improvement_pct = ((avg_new - avg_old) / avg_old * 100) if avg_old > 0 else 0
        total_improvement += improvement_pct
        
        print(f"{field.capitalize()}:")
        print(f"  Old: {avg_old:.2f} avg hits")
        print(f"  New: {avg_new:.2f} avg hits")
        print(f"  Improvement: {improvement_pct:.1f}%")

    overall_improvement = total_improvement / total_fields
    print(f"\nOverall Average Improvement: {overall_improvement:.1f}%")

    # Replace individual assertions with single overall improvement check
    assert overall_improvement > 0, \
        f"New summarizer should show positive improvement (got {overall_improvement:.1f}%)"

@pytest.mark.compare_quotes
@pytest.mark.parametrize("file_path", ["/Users/ryang/Work/ExtensityAI/summarize-lib/testfiles/Align-RUDDER.pdf"])
def test_quotes_comparison(file_path):
    num_runs = 10
    total_hits = defaultdict(lambda: {"old": 0, "new": 0})
    
    for run in range(num_runs):
        logger.info(f"Running quotes comparison {run + 1}/{num_runs}")
        
        # Test new summarizer with quotes enabled
        summarizer_new = HierarchicalSummary(
            file_link=file_path,
            content_types=True,
            user_prompt=str(CONTEXT),
            include_quotes=True
        )
        summary_new, _ = summarizer_new()

        # Test old summarizer with quotes enabled
        summarizer_old = HierarchicalSummaryOldBasic(
            file_link=file_path,
            include_quotes=True
        )
        summary_old, _ = summarizer_old()
        
        # Validate keywords and track hits for this run
        for field in KEYWORDS:
            total_hits[field]["old"] += validate_keywords(summary_old, field)
            total_hits[field]["new"] += validate_keywords(summary_new, field)

    # Calculate and print average hits
    print("\nAverage Keyword Validation Results:")
    total_improvement = 0
    total_fields = len(KEYWORDS)
    
    for field in KEYWORDS:
        avg_old = total_hits[field]["old"] / num_runs
        avg_new = total_hits[field]["new"] / num_runs
        improvement_pct = ((avg_new - avg_old) / avg_old * 100) if avg_old > 0 else 0
        total_improvement += improvement_pct
        
        print(f"{field.capitalize()}:")
        print(f"  Old: {avg_old:.2f} avg hits")
        print(f"  New: {avg_new:.2f} avg hits")
        print(f"  Improvement: {improvement_pct:.1f}%")

    overall_improvement = total_improvement / total_fields
    print(f"\nOverall Average Improvement: {overall_improvement:.1f}%")

    # Assert based on overall improvement
    assert overall_improvement > 0, \
        f"New summarizer should show positive improvement (got {overall_improvement:.1f}%)"

def validate_keywords(output, field):
    """
    Validates if the output contains the expected keywords.
    :param output: Summary object from the summarizer
    :param field: Field to validate (e.g., 'summary' or 'facts')
    :return: Number of matches
    """
    matches = 0

    # Check if output is a Summary instance from either class
    if not isinstance(output, (Summary, SummaryOldBasic, SummaryOld)):
        logger.warning(f"Output is not a Summary instance: {type(output)}")
        return 0
    # Check if the field exists in the output
    if not hasattr(output, field):
        logger.warning(f"Output missing expected field: {field}")
        return 0
    
    # Get the field content using getattr
    field_content = getattr(output, field)
    # Handle facts list separately
    if field == 'facts':
        field_content = ' '.join(field_content)  # Join facts list into single string
    else:
        field_content = str(field_content)
        
    for keyword in KEYWORDS[field]:
        if keyword.lower() in field_content.lower():
            matches += 1
    return matches


@pytest.mark.compare_sentiment
@pytest.mark.parametrize("file_path", ["/Users/ryang/Work/ExtensityAI/summarize-lib/testfiles/Align-RUDDER.pdf"])
def test_sentiment_comparison(file_path):
    num_runs = 10
    summary_results = []
    facts_results = []
    quotes_results = []
    
    for run in range(num_runs):
        logger.info(f"Running sentiment comparison {run + 1}/{num_runs}")
        
        # Test new summarizer with quotes enabled
        summarizer_new = HierarchicalSummary(
            file_link=file_path,
            content_types=True,
            user_prompt=str(CONTEXT),
            include_quotes=True
        )
        summary_new, _ = summarizer_new()

        # Test old summarizer with quotes enabled
        summarizer_old = HierarchicalSummaryOldBasic(
            file_link=file_path,
            include_quotes=True
        )
        summary_old, _ = summarizer_old()

        comparison_summary = Symbol(
            "# Summary Comparison Analysis\n\n"
            + "## Contextual Accuracy\n"
            + "Does the new summary more (or equally) accurately reflect the given context?\n\n"
            + "## Content Requirements Analysis\n" 
            + "Does the new summary more (or equally) accurately reflect the required content?\n\n"
            + "## Required Content Specifications\n"
            + "- Extract key statements, contributions, main results, and important references\n"
            + "- Focus on research questions, methodology, data analysis, findings, and conclusions\n"
            + "- Include limitations, future work, and important references\n\n"
            + "## Context and Summaries\n"
            + f"### Context\n{CONTEXT}\n\n"
            + f"### New Summary\n{summary_new.summary}\n\n"
            + f"### Old Summary\n{summary_old.summary}"
            + f"**Important:** Answer the questions with a single commulative 'yes' or 'no'."
        ).interpret()

        comparison_facts = Symbol(
            "# Summary Comparison Analysis\n\n"
            + "## Contextual Accuracy\n"
            + "Do the new facts more accurately reflect the given context?\n\n"
            + "## Content Requirements Analysis\n"
            + "Do the new facts more accurately reflect the required content?\n\n"
            + "## Required Content Specifications\n"
            + "- Extract key statements, contributions, main results, and important references\n"
            + "- Focus on research questions, methodology, data analysis, findings, and conclusions\n"
            + "- Include limitations, future work, and important references\n\n"
            + "## Context and Summaries\n"
            + f"### Context\n{CONTEXT}\n\n"
            + f"### New Facts\n{summary_new.facts}\n\n"
            + f"### Old Facts\n{summary_old.facts}"
            + f"**Important:** Answer the questions with a single commulative 'yes' or 'no'."
        ).interpret()

        comparison_quotes = Symbol(
            "# Summary Comparison Analysis\n\n"
            + "## Contextual Accuracy\n"
            + "Do the new quotes more accurately reflect the given context?\n\n"
            + "## Content Requirements Analysis\n"
            + "Do the new quotes more accurately reflect the required content?\n\n"
            + "## Required Content Specifications\n"
            + "- Extract key statements, contributions, main results, and important references\n"
            + "- Focus on research questions, methodology, data analysis, findings, and conclusions\n"
            + "- Include limitations, future work, and important references\n\n"
            + "## Context and Summaries\n"
            + f"### Context\n{CONTEXT}\n\n"
            + f"### New Quotes\n{summary_new.quotes}\n\n"
            + f"### Old Quotes\n{summary_old.quotes}"
            + f"**Important:** Answer the questions with a single commulative 'yes' or 'no'."
        ).interpret()

        summary_results.append("yes" in comparison_summary.lower() or "true" in comparison_summary.lower())
        facts_results.append("yes" in comparison_facts.lower() or "true" in comparison_facts.lower())
        quotes_results.append("yes" in comparison_quotes.lower() or "true" in comparison_quotes.lower())

    # Calculate final scores
    summary_score = sum(summary_results)
    facts_score = sum(facts_results)
    quotes_score = sum(quotes_results)

    print(f"\nFinal Scores after {num_runs} runs:")
    print(f"Summary score: {summary_score}/{num_runs}")
    print(f"Facts score: {facts_score}/{num_runs}")
    print(f"Quotes score: {quotes_score}/{num_runs}")

    # Assert final scores are above 50%
    assert summary_score > num_runs/2, f"Summary score {summary_score}/{num_runs} is not above 50%"
    assert facts_score > num_runs/2, f"Facts score {facts_score}/{num_runs} is not above 50%"
    assert quotes_score > num_runs/2, f"Quotes score {quotes_score}/{num_runs} is not above 50%"