import re
import logging
from langchain_core.messages import HumanMessage
from ..model_handling import llm_jd

# Module-level logger (diagnostics only)
LOGGER = logging.getLogger(__name__)

# Precompiled regexes to strip fenced code blocks around JSON
_CODE_FENCE_START = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_CODE_FENCE_END = re.compile(r"\s*```$")


async def parse_jd_text_to_json(jd_text: str) -> str:
    """
    Ask the JD model to extract a structured JSON from raw job description text.
    Returns the model's content (string) with any surrounding markdown code
    fences removed. On any exception, returns: "JD not contain any text"
    """
    llm = llm_jd

    prompt = (
        "Give me a JSON from the provided job description, that should have the job_role as its key "
        "with a value having the actual job role name then a - followed by the given details of technical knowledge and "
        "fundamental_knowledge as its other key whose value mentions any educational requirements if provided otherwise it should be kept null and "
        "company_background as its other key having its value as the actual company name then a - followed by the provided background details, "
        "if no company name is provided but background details are then just writing those details without the company name and also if the background "
        "details are missing but the company name is present then again only write the company name there without any background details, also if none of "
        "background details or company name are present then just write company background is not provided in this key.\n"
        "Now extract structured data and return only a JSON object in this exact format:\n"
        "{\n"
        '    "job_role":"actual job role name - ...",\n'
        '    "fundamental_knowledge": "...",  <If any educational requirements are not mentioned then it should be null always>\n'
        '    "company_background":"actual company name - ..."\n'
        "}\n"
        "Job Description Text:\n"
        f'\"\"\"{jd_text}\"\"\"'
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = (response.content or "").strip()

        # Strip code fences if the model wrapped the JSON
        if content.startswith("```"):
            content = _CODE_FENCE_START.sub("", content)
            content = _CODE_FENCE_END.sub("", content)

        return content.strip()
    except Exception as e:
        LOGGER.warning("Failed to parse JD to JSON: %s", e)
        return "JD not contain any text"
