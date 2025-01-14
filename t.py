from src.hierarchical import HierarchicalSummary

def test():
    # Test with local file
    summarizer = HierarchicalSummary(
        file_link="testfiles/symbolicai_no_refs.pdf"
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
