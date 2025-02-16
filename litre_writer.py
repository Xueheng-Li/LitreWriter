# %%
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import pymupdf
import argparse
from review_prompts import summary_writer_template, review_prompt_template, translate_template

from dotenv import load_dotenv
load_dotenv()
# Get environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME')

# %%


# load a paper pdf file
file_path = "papers/Goyal et al. - 2017 - Information Acquisition and Exchange in Social Networks.pdf"
def load_pdf(file_path: str, max_chars: int = 0):
    """Load PDF with length limit to avoid context window issues"""
    doc = pymupdf.open(file_path)
    combined_pages = "\n\n".join([page.get_text() for page in doc])
    # If max_chars is 0, return the entire content
    if max_chars > 0 and len(combined_pages) > max_chars:
        truncated = combined_pages[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > 0:
            truncated = truncated[:last_period + 1]
        return truncated
    return combined_pages

paper_str = load_pdf(file_path)
import json

def extract_json(text: str) -> dict:
    # Extract the JSON part from the content
    json_start = text.find('```json') + len('```json')
    json_end = text.rfind('```')
    json_str = text[json_start:json_end].strip()
    # Parse the JSON string
    return json.loads(json_str)


#%%
# list all pdf paths in the papers directory
import os
def list_pdf_paths(folder: str) -> List[str]:
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    return [os.path.join(folder, f) for f in pdf_files]

def save_review(review_text: str, topic: str, path: str, language: str = ""):
    # write review to a markdown file
    # replace spaces with underscores
    topic = topic.replace(" ", "_")
    if language:
        language = f"_{language}"
    md_path = os.path.join(path, f"review_{topic}{language}.md")
    # check if the file exists; if so, create a new file like literature_review_1.md
    if os.path.exists(md_path):
        base, ext = os.path.splitext(md_path)
        i = 1
        while os.path.exists(md_path):
            md_path = f"{base}_{i}{ext}"
            i += 1
    with open(md_path, "w") as f:
        f.write(review_text)
        print(f"Review written to {md_path}")


llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    model=MODEL_NAME, 
    temperature=0.5,
    streaming=False,
)


# %%
# schema for the summary call
def create_summary_model(research_topic: str) -> type:
    class Summary(BaseModel):
        name: str = Field(
            description="Name of the paper."
        )
        year: str = Field(
            description="Year of publication"
        )
        authors: str = Field(
            description="Authors of the paper."
        )
        summary: str = Field(
            description="Detailed, extensive, and in-depth summary of the paper resesarch question, methodology, results and conclusion, up to 1000~ words."
        )
        relation_to_topic: str = Field(
            description=f"How does the paper contribute (or does not contribute) to research on {research_topic}?"
        )
    
    return Summary


def format_summaries(summaries) -> str:
    """
    Converts a list of Summary objects into a formatted string.
    
    Args:
        summaries: List of Summary objects containing paper information
        
    Returns:
        str: Formatted string containing all paper summaries
        
    Raises:
        ValueError: If summaries list is empty
    """
    if not summaries:
        raise ValueError("No summaries provided to format")

    formatted_papers = []
    for summary in summaries:
        paper_summary = (
            f"# {summary.name}\n"
            f"**Year:** {summary.year}\n"
            f"**Authors:** {summary.authors}\n\n"
            f"## Summary\n{summary.summary}\n\n"
            f"## Relation to topic\n{summary.relation_to_topic}"
        )
        formatted_papers.append(paper_summary)

    return "\n\n---\n\n".join(formatted_papers)


class Paper(BaseModel):
    path: str = Field(
        description="Path to the paper PDF file.",
    )
    research_topic: str = Field(
        description="Research topic specified by the user.",
    )

class Papers(BaseModel):
    papers: List[Paper] = Field(
        description="List of paper paths.",
    )


# Graph state
class State(TypedDict):
    research_topic: str  # research topic specified by the user
    papers: list[Paper]  # List of paper paths
    completed_summaries: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    review: str  # English review
    chinese_review: Optional[str]
    paper_folder: str  # Path to the folder containing papers
    formatted_summaries: str  # Formatted summaries for review


# Worker state
class WorkerState(TypedDict):
    paper: Paper # Paper to summarize
    completed_summaries: Annotated[list, operator.add]


# %%
# Nodes
def orchestrator(state: State):
    paper_paths = list_pdf_paths(state["paper_folder"])
    papers = [Paper(path=path, research_topic=state["research_topic"]) for path in paper_paths]
    return {"papers": papers}


import time
def summary_call(state: WorkerState):
    """Worker writes a summary of the paper with error handling"""
    try:
        # Generate summary
        paper_str = load_pdf(state["paper"].path)
        paper_prompt = summary_writer_template.invoke({
            "paper": paper_str, 
            "research_topic": state["paper"].research_topic
        })
        Summary = create_summary_model(state["paper"].research_topic)
        summary_writer = llm.with_structured_output(Summary)
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                summary_dict = summary_writer.invoke(paper_prompt)
                return {"completed_summaries": [summary_dict]}
            except ValueError as e:
                retry_count += 1
                if retry_count == max_retries:
                    # If all retries failed, return a basic summary
                    return {"completed_summaries": [{
                        "name": os.path.basename(state["paper"].path),
                        "year": "N/A",
                        "authors": "N/A",
                        "summary": f"Error processing paper: {str(e)}",
                        "relation_to_topic": "Unable to determine due to processing error"
                    }]}
                time.sleep(2)  # Wait before retrying
                
    except Exception as e:
        # Return a basic summary if there's any error
        return {"completed_summaries": [{
            "name": os.path.basename(state["paper"].path),
            "year": "N/A",
            "authors": "N/A",
            "summary": f"Error processing paper: {str(e)}",
            "relation_to_topic": "Unable to determine due to processing error"
        }]}


def review_writer(state: State):
    """Write a literature review based on the summaries"""

    # List of completed sections
    formatted_summaries = format_summaries(state["completed_summaries"])
    review_prompt = review_prompt_template.invoke({
        "research_topic": state["research_topic"],
        "paper_summaries": formatted_summaries})
    review = llm.invoke(review_prompt).content
    # save the review to a local file
    save_review(review, state['research_topic'], state['paper_folder'], language="Eng")
    print("Review completed.")
    return {"review": review, "formatted_summaries": formatted_summaries}

def translator(state: State):
    """Translate the literature review to Chinese"""
    translate_prompt = translate_template.invoke({"review": state["review"]})
    chinese_review = llm.invoke(translate_prompt).content
    save_review(chinese_review, state['research_topic'], state['paper_folder'], language="zh")
    print("Translation completed.")
    return {"chinese_review": chinese_review}


# Conditional edge function to create summary_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [Send("summary_call", {"paper": paper}) for paper in state["papers"]]


# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("summary_call", summary_call)
orchestrator_worker_builder.add_node("review_writer", review_writer)
orchestrator_worker_builder.add_node("translator", translator)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["summary_call"]
)
orchestrator_worker_builder.add_edge("summary_call", "review_writer")
orchestrator_worker_builder.add_edge("review_writer", "translator")
orchestrator_worker_builder.add_edge("translator", END)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Literature Review Generator")
    parser.add_argument("--paper_folder", type=str, required=True, help="Path to the folder containing papers")
    parser.add_argument("--topic", type=str, required=True, help="Research topic")


    # Parse arguments
    args = parser.parse_args()

    # Compile the workflow
    orchestrator_worker = orchestrator_worker_builder.compile()

    # # Show the workflow
    # display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))

    # Invoke with correct parameter name
    state = orchestrator_worker.invoke({"research_topic": args.topic, "paper_folder": args.paper_folder})

    # write state["review"] and formatted summaries to a markdown file
    review_text = f"{state['review']}\n\n---\n\n{state['chinese_review']}\n\n---\n\nFormatted Summaries:\n{state['formatted_summaries']}"
    save_review(review_text, state['research_topic'], state['paper_folder'])

    return state

if __name__ == "__main__":
    main()