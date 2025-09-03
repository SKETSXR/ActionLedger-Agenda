import json
import pprint
import yaml
import time
import asyncio
from src.agent.AgendaGenerationAgent import AgendaGenerationAgent
from src.schema.input_schema import JobDescriptionSchema, CandidateProfileSchema, SkillTreeSchema, InputSchema
# from src.schema.output_schema import OutputSchema
from src.tools.jd_extraction import parse_jd_text_to_json
from dotenv import load_dotenv
# from src.db_fetch import read_start

load_dotenv()

# with open(r"new_jd.json", "r", encoding='utf-8') as j:
#     jd = json.load(j)
# jd_inp_text = """We are looking for a skilled Data Scientist to join our team at Tech Innovators Inc. The ideal candidate will have a strong background in statistical analysis, machine learning, and data visualization. Responsibilities include analyzing large datasets to extract insights, building predictive models, and collaborating with cross-functional teams to implement data-driven solutions. A bachelor's degree in Computer Science, Statistics, or a related field is required, with a preference for candidates holding a master's degree or higher. Experience with Python, R, SQL, and cloud platforms such as AWS or Azure is essential. Familiarity with big data technologies like Hadoop and Spark is a plus. Join us at Tech Innovators Inc., a leading company in the tech industry known for its innovative solutions and dynamic work environment."""
jd_inp_text = """Edwisely - Intelligent Learning Infrastructure

Were seeking an AI Solutions Engineer to lead the integration of GenAI tools such as ChatGPT or Claudeinto our core platform. Youll design, prototype, and deploy features that elevate student outcomes, make teaching exciting, and enrich dashboards with intelligent insights, all while upholding privacy and governance standards. Key Responsibilities Design and build GenAI-powered features like guided study assistants, automated remediation engine, and intelligent feedback tools. Develop end-to-end pipelines (prompt design model integration CI/CD deployment) that align with Edwiselys Intelligent Learning Infrastructure. Collaborate with product teams, faculty, and UX designers to deploy AI features in classrooms and dashboards. Ensure all AI implementations meet ISO 27001 security and CERT-In data protection guidelines. Measure adoption and impact via analytics dashboardssupport evidence-based learning outcomes. What You’ll Bring Strong in prompt engineering, RAG, embeddings, or QA systems using frameworks like OpenAI API, LangChain, Hugging Face Transformers. Familiarity with ML deployment—FastAPI, Docker, AWS/GCP—within production-grade applications. Experience with education data, knowledge graphs, student modeling, or assessment systems is a big plus. Results-driven: ability to turn prototypes into scalable features. Passionate about improving higher education with impactful, AI-driven solutions."""
jd_json_string = asyncio.run(parse_jd_text_to_json(jd_inp_text))
jd = json.loads(jd_json_string)
jdes = JobDescriptionSchema(job_role=jd["job_role"], company_background=jd["company_background"], fundamental_knowledge=jd.get("fundamental_knowledge"))
# print(jdes.model_dump_json(indent=2))

with open(r"parsed_cv7.json", "r", encoding='utf-8') as c:
    candidate_profile = json.load(c)
cp = CandidateProfileSchema(skills=candidate_profile["skills"],
                            projects=candidate_profile["projects"],
                            experience=candidate_profile["experience"])
# print(cp.model_dump_json(indent=2))

# Skill tree loading using json
def load_skill_tree(tree_json: dict) -> SkillTreeSchema:
    return SkillTreeSchema(**tree_json)

with open("skilltree.json", "r") as f:
    tree_data = json.load(f)

root = load_skill_tree(tree_data)
inp = InputSchema(
    job_description=jdes,
    skill_tree=root,
    candidate_profile=cp
)

# with open("config.yaml", "r") as yamlfile:
#     config = yaml.load(yamlfile, Loader=yaml.FullLoader)

# start = time.time_ns()
# # print(get_graph().invoke(inp, config))
# graph = AgendaGenerationAgent.get_graph()

# # otpt = graph.invoke(inp, config)
# otpt = asyncio.run(graph.ainvoke(inp, config))
# # print(otpt)
# end = time.time_ns()

# for i, v in otpt.items():
#     if hasattr(v, "model_dump_json"):
#         print(json.dumps(json.loads(v.model_dump_json()), indent=2))
#     elif hasattr(v, "model_dump"):
#         print(json.dumps(v.model_dump(), indent=2))
#     else:
#         pprint.pprint(v, indent=2)
# print(f"\nTime taken: {(end - start) / 60000000000} mins")

inp_cv = inp.candidate_profile.model_dump_json(indent=2)
inp_jd = inp.job_description.model_dump_json(indent=2)
inp_skill_tree = inp.skill_tree.model_dump_json(indent=2)
print(inp_cv + "\n")
print(inp_jd + "\n")
# pprint.pprint(inp.skill_tree.model_dump_json())

print(inp_skill_tree)

with open("config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
graph = AgendaGenerationAgent.get_graph()

# otpt = graph.invoke(inp, config)
otpt = asyncio.run(graph.ainvoke(inp, config))

# Test database fetch (having jd, cv, skill tree, summary)
# read_start()

# Test o/p as a schema object (for entire agenda output)
# print(otpt)

# for i, v in otpt.items():
#     if hasattr(v, "model_dump_json"):
#         print(json.dumps(json.loads(v.model_dump_json()), indent=2))
#     elif hasattr(v, "model_dump"):
#         print(json.dumps(v.model_dump(), indent=2))
#     else:
#         pprint.pprint(v, indent=2)
