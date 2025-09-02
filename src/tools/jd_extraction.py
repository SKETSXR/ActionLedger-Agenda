import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


async def parse_jd_text_to_json(jd_text):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_retries=10)
    
    prompt = f'''Give me a JSON from the provided job description, that should have the job_role as its key with a value having the actual job role name then a - followed by the given details of technical knowledge and fundamental_knowledge as its other key whose value mentions any educational requirements and company_background as its other key having its value as the actual company name then a - followed by the provided background details, if no company name is provided but background details are then just writing those details without the company name and also if the background details are missing but the company name is present then again only write the company name there without any background details, also if none of background details or company name are present then just write company background is not provided in this key.
    Now extract structured data and return only a JSON object in this exact format:
    {{
        "job_role":"actual job role name - ...",
        "fundamental_knowledge": "...", // If it is not mentioned then it can be null
        "company_background":"actual company name - ..."
    }}
    Job Description Text:
    \"\"\"{jd_text}\"\"\"'''

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        # cleaning and converting into pure json format.
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return content.strip()
    except:
        return "JD not contain any text"

