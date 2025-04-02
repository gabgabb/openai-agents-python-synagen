import os
import anthropic
import json
import pymupdf
import requests
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

# Initialize the Anthropics client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Prompt for the Claude model
SYSTEM_PROMPT = (
    "You are an expert medical research assistant. You will be provided with one or more medical documents and a question.\n"
    "Your task is to answer the question clearly and concisely using evidence from the documents.\n"
    "\n"
    "Do not write anything before the 'Answer:' section. Do not explain, apologize, or clarify before the answer, even if the documents are empty, partial, or incomplete.\n"
    "\n"
    "If the documents are insufficient to answer, say so **only** under 'Answer:' and proceed with the 'Quotes:' section.\n"
    "\n"
    "Write in complete, professional sentences suitable for a clinical or academic audience. Avoid listing items mechanically. Use transitions to create a natural flow.\n"
    "\n"
    "Use numbered references like [1], [2], etc., to cite information directly in the answer.\n"
    "\n"
    "After your answer, include a 'Quotes:' section. Only include meaningful quotes from the documents that contain actual content. Do **not** cite document titles, fragments, or incomplete words (e.g., 'Fatig'). If no relevant quotes exist, write:\n"
    "Quotes:\nNo relevant quotes.\n"
    "\n"
    "Start your output exactly like this:\n"
    "\n"
    "Answer:\n"
    "... your answer with [1], [2], etc.\n"
    "\n"
    "Quotes:\n"
    "[1] \"...\" – Document title\n"
    "[2] \"...\" – Document title\n"
)

# Function to extract text from a PDF file
def extract_text_from_pdf_path(path):
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)

# Function to extract text from a PDF file at a URL
def extract_text_from_pdf_url(url):
    response = requests.get(url)
    response.raise_for_status()
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    return extract_text_from_pdf_path("temp.pdf")

def ask_with_citations(question, documents, model="claude-3-5-sonnet-latest", with_raw_response=False):
    """
    Send a question to the Claude model with provided documents and return the response.
    """
    content_blocks = []
    documents = [doc for doc in documents if doc.get("data", "").strip()]

    if not documents:
        raise ValueError("No valid document data provided. Please include at least one non-empty document.")

    for doc in documents:
        raw_data = doc.get("data", "")
        title = doc.get("title", "Untitled")

        # Détection et extraction du texte
        try:
            if isinstance(raw_data, str):
                # PDF local
                if raw_data.lower().endswith(".pdf") and os.path.isfile(raw_data):
                    text_data = extract_text_from_pdf_path(raw_data)

                # PDF en ligne
                elif raw_data.lower().endswith(".pdf") and urlparse(raw_data).scheme in ["http", "https"]:
                    text_data = extract_text_from_pdf_url(raw_data)

                # Texte brut
                else:
                    text_data = raw_data.strip()
            else:
                continue  # Skip non-str data

        except Exception as e:
            print(f"[Warning] Failed to extract '{title}': {e}")
            continue

        if not text_data.strip():
            continue

        content_blocks.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": text_data
            },
            "title": doc["title"],
            "context": doc.get("context", "Context: medical reference document."),
            "citations": {"enabled": True},
            "cache_control": {"type": "ephemeral"}
        })

    content_blocks.append({
        "type": "text",
        "text": question
    })

    response = client.messages.create(
        model=model,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": content_blocks
        }],
        max_tokens=1024
    )

    if with_raw_response:
        print("\n================================================================================")
        print("Raw response:")
        print("================================================================================")
        print(raw_response(response))

        print("================================================================================")
        print("Formatted Response:")
        print("================================================================================")

    return print_annotated_response(response)

def raw_response(response):
    raw_blocks = []
    for block in response.content:
        if block.type == "text":
            block_data = {
                "type": "text",
                "text": block.text.strip(),
            }
            if hasattr(block, "citations") and block.citations:
                block_data["citations"] = []
                for citation in block.citations:
                    c = citation.dict() if hasattr(citation, "dict") else citation
                    block_data["citations"].append({
                        "cited_text": c.get("cited_text", "—"),
                        "document_title": c.get("document", {}).get("title", "Unknown")
                    })
            raw_blocks.append(block_data)

    return json.dumps({"content": raw_blocks}, indent=2)

def print_annotated_response(response):
    """
    Display the response from Claude with citations and quotes.
    Format:
    Answer:
    ...

    Quotes:
    [1] "..."
    """
    output = ""
    for block in response.content:
        if block.type == "text":
            output += block.text.strip() + " "

    print(output.strip())
