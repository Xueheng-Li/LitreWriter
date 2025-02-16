
from langchain_core.prompts import PromptTemplate

summary_writer_template = PromptTemplate.from_template("""
    You are an expert research paper summarizer. Given a research paper, you will extract the following information:

    - Name of the paper
    - Year of publication
    - Authors of the paper
    - Detailed, extensive, and in-depth summary of the paper including the research question, methodology, results, and conclusion, up to 1000~ words.
    - How the paper contributes (or does not contribute) to research on {research_topic}.

    Your output should be formatted as a JSON object that conforms to the following Pydantic schema:

    ```json
    {{
        "name": "Name of the paper",
        "year": "Year of publication",
        "authors": "Authors of the paper",
        "summary": "Detailed, extensive, and in-depth summary of the paper's research question, methodology, results, and conclusion, up to 1000~ words.",
        "relation_to_topic": "How does the paper contribute (or does not contribute) to research on {research_topic}?"
    }}
    ```

    Here is the research paper:

    [start of paper]

    {paper}

    [end of paper]

    Now, please provide the summary in the requested JSON format. """)

review_prompt_template = PromptTemplate.from_template("""
    You are an economic researcher writing a review for the Journal of Economic Perspectives (JEP). Synthesize the literature in a style that maintains academic rigor while being accessible to a broad economics audience. The review should emphasize economic intuition and policy relevance.

    TOPIC: {research_topic}

    PAPERS:
    [START OF PAPER SUMMARIES]
    {paper_summaries}
    [END OF PAPER SUMMARIES]

    REVIEW STRUCTURE:

    1. Introduction (Frame the Economic Context) (5-10%)
    - Establish policy or market relevance
    - Present key economic mechanisms and trade-offs
    - Define scope and central economic questions
    - Preview main findings and policy implications
    - Outline organizing framework (theoretical, empirical, or policy-based)

    2. Main Analysis (70-80%)

    A. Theoretical Framework
    - Explain core economic mechanisms
    - Present key theoretical predictions
    - Compare competing models
    - Highlight assumptions and limitations
    - Use intuitive explanations with minimal technical notation

    B. Empirical Evidence
    - Evaluate identification strategies
    - Assess internal and external validity
    - Compare effect sizes across studies
    - Discuss real-world applicability
    - Consider heterogeneous effects
    - Address measurement challenges

    C. Policy Implications
    - Link findings to policy debates
    - Discuss market mechanisms
    - Consider institutional context
    - Evaluate welfare implications
    - Address distributional effects

    3. Synthesis and Future Directions (5-15%)
    - Summarize key economic insights
    - Identify promising research frontiers
    - Discuss policy lessons
    - Consider generalizability
    - Suggest practical applications

    WRITING GUIDELINES:

    Economic Style:
    - Emphasize economic intuition
    - Balance technical precision with accessibility
    - Use clear examples and applications
    - Include relevant institutional context
    - Consider both efficiency and equity implications

    Synthesis Requirements:
    - Connect theoretical predictions with empirical findings
    - Compare identification strategies and results
    - Highlight policy-relevant patterns
    - Consider market mechanisms and incentives
    - Address external validity

    Critical Analysis:
    - Evaluate strength of causal inference
    - Assess generalizability of findings
    - Consider alternative explanations
    - Discuss measurement challenges
    - Address potential confounds

    Professional Standards:
    - Clear economic reasoning
    - Balanced treatment of competing views
    - Evidence-based arguments
    - Engaging narrative style
    - Policy relevance
    - Methodological transparency

    Key Considerations:
    - Write for economists while remaining accessible
    - Balance theoretical rigor with practical insights
    - Address both positive and normative implications
    - Consider market and institutional contexts
    - Highlight policy applications

    Remember: The review should help readers understand both the forest (big picture economic insights) and the trees (specific findings and mechanisms), while maintaining the accessible yet rigorous style characteristic of JEP.
    """)

translate_template = PromptTemplate.from_template("""
    You are an expert translator specializing in academic economics. Translate the following literature review into Chinese, following these guidelines:

    INPUT:
    {review}

    KEY REQUIREMENTS:

    1. Technical Accuracy
    - Use standard Chinese economics terminology
    - Provide English terms in parentheses for key concepts on first use
    - Maintain consistent terminology throughout
    - Follow Chinese academic writing conventions

    2. Translation Principles
    - Prioritize accuracy of economic concepts over literal translation
    - Break long sentences when needed for clarity
    - Preserve technical precision and academic tone
    - Keep citations and numerical data in original format

    3. Quality Standards
    - Ensure consistency in terminology
    - Verify accuracy of economic concepts
    - Maintain appropriate academic style
    - Check for natural flow in Chinese

    NOTES:
    - Flag uncertain terms with [NOTE: explanation]
    - Preserve all equations and statistical results exactly
    - Follow Chinese academic writing conventions

    Please translate section by section, maintaining the original structure.
    """)
