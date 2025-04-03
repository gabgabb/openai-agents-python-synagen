import base64
import json
import os
from typing import List, Dict, Optional
from urllib.parse import urlparse
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Initialize the Anthropics client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# System prompt for the Claude model
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

def to_dict(obj):
    """
    Convert an object to a dictionary if it has a dict() method, otherwise return the object itself.
    :param obj:
    :return: dict or object
    """
    return obj.dict() if hasattr(obj, "dict") else obj

def resolve_source(raw_data: str) -> Dict:
    """
    Resolve the source of the document data, determining if it's a URL, a file path, or base64 encoded data.
    :param raw_data:
    :return: dict
    """
    parsed = urlparse(raw_data)
    is_url = parsed.scheme in ["http", "https"] and raw_data.lower().endswith(".pdf")
    is_pdf_path = raw_data.lower().endswith(".pdf") and os.path.isfile(raw_data)

    if is_url:
        return {"type": "url", "url": raw_data}
    elif is_pdf_path:
        with open(raw_data, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type": "base64",
            "media_type": "application/pdf",
            "data": encoded
        }
    elif len(raw_data) > 1000 and not raw_data.startswith(" "):  # crude base64
        return {
            "type": "base64",
            "media_type": "application/pdf",
            "data": raw_data
        }
    else:
        return {
            "type": "text",
            "media_type": "text/plain",
            "data": raw_data.strip()
        }

def ask_with_citations(
    question: str,
    documents: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-latest",
    with_raw_response: bool = False
) -> str:
    """
    Send a question to the Claude model with provided documents and return the response.
    :param question: The question to ask.
    :param documents: List of documents to provide as context.
    :param model: The model to use (default is "claude-3-5-sonnet-latest").
    :param with_raw_response: Whether to include the raw response for debugging.
    :return: The formatted response from the model.
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
            source = resolve_source(raw_data)

            document_block = {
                "type": "document",
                "source": source,
                "title": doc["title"],
                "context": doc.get("context") or "No specific context provided.",
                "citations": {"enabled": True},
            }

            if source["type"] in ["base64", "url"]:
                document_block["cache_control"] = {"type": "ephemeral"}

            content_blocks.append(document_block)

        except Exception as e:
            print(f"[Warning] Failed to process document '{title}': {e}")
            continue

    if not content_blocks:
        raise ValueError("No valid documents to send.")

    content_blocks.append({
        "type": "text",
        "text": question
    })

    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": content_blocks
        }],
        max_tokens=1024
    )

    if with_raw_response:
        print("=" * 80)
        print("Raw Response:\n" + "=" * 80)
        print(raw_response(response))
        print("\n" + "=" * 80)
        print("Formatted Response:\n" + "=" * 80)

    return format_annotated_response(response, index_to_title)

def raw_response(
    response: anthropic.types.Message
) -> str:
    """
    Format the raw response from the Claude model for debugging purposes.
    :param response:
    :return:
    """
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

def format_annotated_response(
    response: anthropic.types.Message,
    index_to_title: Optional[Dict[int, str]] = None
) -> str:
    """
    Format the response from the Claude model, including citations and token usage.
    :param response:
    :param index_to_title:
    :return: str
    """
    answer = ""
    citation_bank = []

    for block in response.content:
        if block.type == "text":
            text = block.text.strip()
            refs = getattr(block, "citations", [])
            if refs:
                ref_number = len(citation_bank) + 1
                citation_bank.append(refs)
                answer += f"{text} [{ref_number}] "
            else:
                answer += f"{text} "

    output = "Answer:\n" + answer.strip()

    if citation_bank:
        output += "\n\nQuotes:"
        for i, ref_group in enumerate(citation_bank, 1):
            for ref in ref_group:
                ref_dict = to_dict(ref)
                cited_text = ref_dict.get("cited_text", "—").strip()
                title = (
                    ref_dict.get("document_title")
                    or ref_dict.get("document", {}).get("title")
                    or (
                        index_to_title.get(ref_dict.get("document_index"))
                        if index_to_title and "document_index" in ref_dict
                        else "Unknown"
                    )
                )
                output += f"\n[{i}] \"{cited_text}\" – {title}"

    output += (
        f"\n\n🔢 Tokens used: input={response.usage.input_tokens}, "
        f"output={response.usage.output_tokens}"
    )

    return output
