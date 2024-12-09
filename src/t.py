from hierarchical import HierarchicalSummary

CONTENT_TYPES = [
    "Interview",
    "Keynote",
    "Scientific Paper",
    "Report",
    "Book",
    "Presentation Slides"
]

def test():
    # Test with local file
    summarizer = HierarchicalSummary(
        file_link="/Users/ryang/Work/ExtensityAI/summarize-lib/testfiles/symbolicai_no_refs.pdf",
        content_types=CONTENT_TYPES
    )
    summary, _ = summarizer()
    
    summary, usage = summarizer.forward()
    print("\nFile Summary:")
    print(f"Type: {summary.type}")
    print(f"Summary: {summary.summary}")
    print(f"Facts: {summary.facts}")

if __name__ == "__main__":
    print("Running HierarchicalSummary test...")
    test()
