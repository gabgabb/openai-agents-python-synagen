# Anthropic citations agent

This module extends the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python) with support for **Anthropic-style sentence-level citations**, enabling source-grounded answers for clinical, scientific, and academic use cases.

## Design & implementation choices

This tool was built to bridge the gap between the OpenAI Agents SDK and Anthropic’s native support for structured citations.

### Document flexibility

We support three input types:

| Type | Format                    | Example                |
|------|---------------------------|------------------------|
| Text | Plain string              | `This is a text.`      |
| PDF  | Locale file url           | `/data/file.pdf`       |
| PDF  | PDF URL Direct public URL | `https://.../file.pdf` |

### Modular citation parsing
The output is reconstructed dynamically from Claude's response, where:

- Each citation block is parsed and numbered

- Citations are printed inline [1], [2] after sentences

- Quotes are extracted only if meaningful (as per the system prompt)

This design ensures:

- Minimal token overhead (citations are handled post-response)

- Consistent formatting

- Easy reuse across notebooks, APIs, or other agents

## Installation

This project requires **Python 3.10+, recommended: Python 3.12.x**  
It is tested and developed with **Python 3.12.3**.

```bash
  cp .env.example .env
````
```
  Edit .env to set your Anthropic API key
  ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

```bash
  pip install -r requirements.txt
```

## Usage

You can use `ClaudeCitationAgent` like any standard agent in the OpenAI Agents SDK.  
It accepts a `task` (the user question) and a `context` dictionary with the documents.

### 📘 Document format

Each document must be a dictionary with the following structure:

```python
{
    "title": str,        # Required: human-readable title of the document
    "data": str,         # Required: text content, local PDF path, or PDF URL
    "context": str       # Optional: custom context to guide Claude (e.g., "Journal article", "Hospital record")
}
```

### Example
```python
from src.agents.implementations.claude_citation_agent import ClaudeCitationAgent

agent = ClaudeCitationAgent()

question = "What are the causes of cancer?"
documents = [
    {
        "title": "Cancer Overview",
        "data": "https://example.com/cancer.pdf"  # can be text, file path, or URL
    }
]

response = agent.run(task=question, context={"documents": documents})
print(response)
```

Use the optional `with_raw_response=True` flag for debugging or to get the raw response from Claude :
```python
response = agent.run(
    task="What are causes of fatigue?",
    context={
        "documents": [
            {
                "title": "Medical Notes",
                "data": "Fatigue is normal and can have many causes such as lack of sleep or illness."
            }
        ]
    },
    with_raw_response=True
)
```

## Features
- Dynamic document preprocessing
- PDF support from local files or URLs
- Annotated answer with inline references [1], [2] etc.
- Fallback to “No relevant quotes” when appropriate

## Limitations

This module was developed using a personal Anthropic API key, without access to enterprise or elevated-rate limits.

As a result:

- Token usage was limited, especially when working with large PDFs or multiple documents.

- Some design choices (e.g., truncating content, limiting document count per call) were made to stay within personal usage quotas.

- Performance may vary depending on your claude model.

## References

- [Github anthropic](https://github.com/anthropics/anthropic-cookbook/tree/main/misc)
- [Anthropic doc](https://docs.anthropic.com/en/docs)
- [OpenAI agents](https://openai.github.io/openai-agents-python)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Genes_1](https://www.sciencedirect.com/science/article/pii/S1063458424000682)
- [Genes_2](https://www.sciencedirect.com/science/article/pii/S1063458424001122?ref=pdf_download&fr=RR-2&rr=92a9c9e558748cf9)