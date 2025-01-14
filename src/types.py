from enum import Enum


class DocumentType(str, Enum):
    BOOK = "book"
    ARTICLE = "article"
    WIKI = "wiki"
    INTERVIEW = "interview"
    PRESENTATION_SLIDES = "presentation_slides"
    TALK = "talk"
    KEYNOTE = "keynote"
    SCIENTIFIC_PAPER = "scientific_paper"
    REVIEW_PAPER = "review_paper"
    REPORT = "report"
    PODCAST = "podcast"
    UNKNOWN = "unknown"


TYPE_SPECIFIC_PROMPTS = {
    DocumentType.SCIENTIFIC_PAPER: """
            - Extract the title, authors, and publication details. Identify the main topic and scope.
            - Extract key statements, contributions, main results, and important references. Focus on research questions, methodology, data analysis, findings, and conclusions. Include limitations, future work, and important references.
            """,
    DocumentType.REVIEW_PAPER: """
            - Extract the title, authors, and publication details. Identify the main topic and scope.
            - Highlight the reviewed topics, key findings from literature, and synthesis of current knowledge.
            """,
    DocumentType.KEYNOTE: """
            - Identify the presenter, target audience, and overall structure.
            - Include speaker details and their expertise. Highlight key messages and main takeaways.""",
    DocumentType.PRESENTATION_SLIDES: """
            - Identify the presenter, target audience, and overall structure.
            - Determine if this is a motivational talk, results presentation, or idea/pitch. For motivational talks, focus on key messages and call-to-action. For result presentations, emphasize numerical results and achievements. For idea/pitch presentations, highlight the core idea and value proposition.""",
    DocumentType.INTERVIEW: "Identify and distinguish between different speakers. Include key quotes and main discussion points.",
    DocumentType.REPORT: "Highlight numerical results, key statistics, and main takeaways. Include significant findings and conclusions.",
    DocumentType.BOOK: "Include author information, main plot points, and key character descriptions. Highlight character development and relationships.",
}
