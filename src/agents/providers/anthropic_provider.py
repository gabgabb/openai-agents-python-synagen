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
    "You are an expert research assistant. You will be provided with one or more documents and a question.\n"
    "Your task is to answer the question clearly and concisely using evidence from the documents.\n"
    "\n"
    "Use numbered citations like [1], [2], etc., only when the document contains meaningful, relevant information that helps answer the question.\n"
    "⚠️ Do not include citations for generic or trivial statements like 'Fatigue is normal' or 'Sleep is important.'\n"
    "If the document contains only vague or insufficient content, do not cite it.\n"
    "\n"
    "Do not include a 'Quotes:' section. Do not explain your citations.\n"
    "If no relevant quotes exist, say exactly:\n"
    "Quotes:\nNo relevant quotes.\n"
    "\n"
    "If the question cannot be answered from the document, say so clearly.\n"
    "Write in complete, professional sentences suitable for a clinical or academic audience."
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

    index_to_title = {}
    for i, doc in enumerate(documents):
        raw_data = doc.get("data", "")
        title = doc.get("title", f"Document {i+1}")
        index_to_title[i] = title

        try:
            if isinstance(raw_data, str):
                # Local PDF
                if raw_data.lower().endswith(".pdf") and os.path.isfile(raw_data):
                    text_data = extract_text_from_pdf_path(raw_data)

                # Online PDF
                elif raw_data.lower().endswith(".pdf") and urlparse(raw_data).scheme in ["http", "https"]:
                    text_data = extract_text_from_pdf_url(raw_data)

                # Brut text
                else:
                    text_data = raw_data.strip()
            else:
                continue  # Skip non-str data

        except Exception as e:
            print(f"[Warning] Failed to extract '{title}': {e}")
            continue

        if not text_data.strip():
            continue

        is_pdf = (
            raw_data.lower().endswith(".pdf")
            or (doc.get("media_type") == "application/pdf")
        )

        parsed = urlparse(raw_data)
        is_url = parsed.scheme in ["http", "https"]
        is_pdf = raw_data.lower().endswith(".pdf")

        # Gestion des différents formats
        if is_url and is_pdf:
            source = {
                "type": "url",
                "url": raw_data
            }
        elif is_pdf and os.path.isfile(raw_data):
            text_data = extract_text_from_pdf_path(raw_data)
            source = {
                "type": "base64",
                "media_type": "application/pdf",
                "data": text_data
            }
        else:
            text_data = raw_data.strip()
            source = {
                "type": "text",
                "media_type": "text/plain",
                "data": text_data
            }

        document_block = {
            "type": "document",
            "source": source,
            "title": doc["title"],
            "context": doc.get("context", "Context: medical reference document."),
            "citations": {"enabled": True},
        }

        if is_pdf or is_url:
            document_block["cache_control"] = {"type": "ephemeral"}

        content_blocks.append(document_block)

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

    return print_annotated_response(response, index_to_title)

# Function to format the raw response
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

# Function to print the annotated response
def print_annotated_response(response, index_to_title=None):
    """
    Reconstruct the answer and citations dynamically from Claude's response.
    """
    answer = ""
    citation_bank = []
    citation_map = {}

    for block in response.content:
        if block.type == "text":
            text = block.text.strip()
            refs = getattr(block, "citations", [])

            if refs:
                # Create a new citation number
                ref_number = len(citation_bank) + 1
                citation_bank.append(refs)
                citation_map[id(refs[0])] = ref_number
                answer += f"{text} [{ref_number}] "
            else:
                answer += f"{text} "

    print("Answer :\n" + answer.strip())

    if citation_bank:
        print("\nQuotes :")
        for i, ref_group in enumerate(citation_bank, 1):
            for ref in ref_group:
                ref_dict = ref.dict() if hasattr(ref, "dict") else ref
                cited_text = ref_dict.get("cited_text", "—").strip()
                title = ref_dict.get("document", {}).get("title")

                if not title and index_to_title and "document_index" in ref_dict:
                    title = index_to_title.get(ref_dict["document_index"], "Unknown")

                print(f"[{i}] \"{cited_text}\" – {title}")

    print(f"\n🔢 Tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}")


