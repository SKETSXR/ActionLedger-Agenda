import argparse
import asyncio
import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage
from rich import print as rprint
from rich.console import Console

from src.model_handling import llm_cv

SYSTEM_PROMPT = """
You are a resume parsing agent.
The input is a Markdown-formatted resume.
```{markdown_text}```\n

Now extract structured data and return only in a JSON format in this given format:

{{
  "skills": ["..."],
  "experience": [
    {{
      "id": "...", <Assign a unique id to each experience like E1, E2, etc>
      "title": "...",
      "company": "...",
      "description": "..."
    }}
  ],
  "projects": [
    {{
      "id": "...", <Assign a unique id to each project like P1, P2, etc>
      "title": "...",
      "description": "..." <If a proper description is not there but only skills or any related and relevant text is there for the respective project then write that as it should not be blank>
    }}
  ]
}}

Guidelines:
- Extract each skill individually, even if listed together in a paragraph or bullet.
- If fields are missing, use null or an empty array — never guess or add fields that don't match the format.
- Do NOT include any notes, markdown, or explanation — return only the JSON object."""

# Precompiled regexes to strip optional fenced code blocks around model JSON
_CODE_FENCE_START = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_CODE_FENCE_END = re.compile(r"\s*```$")


async def extract_cv_text_from_pdf(pdf_path: str) -> str:
    """
    Extract plain text from a PDF resume and return a single concatenated string.
    """
    path = Path(pdf_path)
    with fitz.open(path, filetype="pdf") as doc:
        parts = []
        for page in doc:
            parts.append((page.get_text() or "").strip())
        return "\n".join(filter(None, parts)).strip()


async def parse_pdf_to_json(pdf_path: str) -> str:
    """
    Parse a resume PDF into structured JSON via the LLM.
    On error, returns: "CV does not contain proper text"
    """
    markdown_text = await extract_cv_text_from_pdf(pdf_path)
    try:
        response = await llm_cv.ainvoke(
            [HumanMessage(content=SYSTEM_PROMPT.format(markdown_text=markdown_text))]
        )
        json_text = (response.content or "").strip()

        # Remove leading/trailing code fences if present
        if json_text.startswith("```"):
            json_text = _CODE_FENCE_START.sub("", json_text)
            json_text = _CODE_FENCE_END.sub("", json_text)

        return json_text.strip()
    except Exception:
        return "CV does not contain proper text"


async def main() -> None:
    """
    CLI entry:
      --pdf_path : resume PDF path
      --save     : path to save the JSON string output
    """
    parser = argparse.ArgumentParser(description="Resume Parser CLI")
    parser.add_argument(
        "--pdf_path",
        help="Path to the resume PDF file",
        default=r"C:\Users\akshivk\Desktop\sample resumes\Sourav_s_CV.pdf",
    )
    parser.add_argument(
        "--save",
        help="Path to save parsed JSON output",
        default=r"C:\Users\akshivk\Desktop\agenda_fsd_data\sourav_fsd_cv.json",
    )
    args = parser.parse_args()

    console = Console()
    console.print(f"\n[bold blue] Parsing resume:[/bold blue] {args.pdf_path}\n")

    parsed_json = await parse_pdf_to_json(args.pdf_path)

    console.print("[bold green] Parsed Resume JSON:[/bold green]\n")
    rprint(parsed_json)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve behavior: write the JSON string (not parsed) to file
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=4)
        console.print(f"\n[bold yellow] Saved output to:[/bold yellow] {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
