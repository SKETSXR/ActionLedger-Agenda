import argparse
import json
import re
import fitz
import asyncio
from langchain_core.messages import HumanMessage
from rich import print as rprint
from rich.console import Console
from ..model_handling import llm_cv

SYSTEM_PROMPT = '''
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
      "description": "..."
    }}
  ]
}}
 
Guidelines:
- Extract each skill individually, even if listed together in a paragraph or bullet.
- If fields are missing, use null or an empty array — never guess or add fields that don't match the format.
- Do NOT include any notes, markdown, or explanation — return only the JSON object.'''


# ----------------------------
# Convert Resume to Markdown
# ----------------------------
async def extract_cv_text_from_pdf(pdf_path):
    doc=fitz.open(pdf_path, filetype="pdf") 
    all_text = ""
    for page in doc:
        all_text += page.get_text().strip() + "\n"
    doc.close()
    cv_text = all_text.strip()

    return cv_text


# ----------------------------
# Parse pdf text and then send Markdown to Model
# ----------------------------
async def parse_pdf_to_json(pdf_path):
    
    markdown_text = await extract_cv_text_from_pdf(pdf_path)
    try:
        response = await llm_cv.ainvoke([HumanMessage(content=SYSTEM_PROMPT.format(markdown_text=markdown_text))])
        # cleaning and converting into pure json format.
        json_text = response.content.strip()
        if json_text.startswith("```"):
            json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
            json_text = re.sub(r"\s*```$", "", json_text)
        return json_text

    except:
        return "CV does not contain proper text"


# ----------------------------
# CLI Entry Point
# ----------------------------
async def main():
    parser = argparse.ArgumentParser(description="Resume Parser CLI")
    parser.add_argument("--pdf_path", help="Path to the resume PDF file", default=r"C:\Users\akshivk\Desktop\sample resumes\Sourav_s_CV.pdf")
    parser.add_argument("--save", help="Path to save parsed JSON output", default=r"C:\Users\akshivk\Desktop\agenda_fsd_data\sourav_fsd_cv.json")
    args = parser.parse_args()

    console = Console()
    console.print(f"\n[bold blue] Parsing resume:[/bold blue] {args.pdf_path}\n")

    parsed_json = await parse_pdf_to_json(args.pdf_path)

    console.print("[bold green] Parsed Resume JSON:[/bold green]\n")
    rprint(parsed_json)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=4)
        console.print(f"\n[bold yellow] Saved output to:[/bold yellow] {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
