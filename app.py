import json
import yaml
import time
import asyncio
from src.agent.AgendaGenerationAgent import AgendaGenerationAgent
from src.schema.input_schema import JobDescriptionSchema, CandidateProfileSchema, SkillTreeSchema, InputSchema
from src.tools.jd_extraction import parse_jd_text_to_json
from src.tools.cv_extraction import parse_pdf_to_json
from dotenv import load_dotenv

start = time.time_ns()
load_dotenv()

with open(r"testing\Ayam\jd.txt", "r", encoding="utf-8") as f:
    jd_inp_text = f.read()
jd_json_string = asyncio.run(parse_jd_text_to_json(jd_inp_text))
if jd_json_string == "JD not contain any text":
    raise("Open AI API not running")
jd = json.loads(jd_json_string)
jdes = JobDescriptionSchema(job_role=jd["job_role"], company_background=jd["company_background"], fundamental_knowledge=jd.get("fundamental_knowledge"))

candidate_profile = asyncio.run(parse_pdf_to_json(r"testing\Ayam\AyamHeniberMeitei_2025L - ayam heniber.pdf"))
if candidate_profile == "CV does not contain proper text":
    raise("Open AI API not running")
candidate_profile = json.loads(candidate_profile)

cp = CandidateProfileSchema(skills=candidate_profile["skills"],
                            projects=candidate_profile["projects"],
                            experience=candidate_profile["experience"])

# Skill tree loading using json
def load_skill_tree(tree_json: dict) -> SkillTreeSchema:
    return SkillTreeSchema(**tree_json)

with open(r"testing\custom_testing_inputs\skilltree3_priority.json", "r", encoding="utf-8") as f:
    tree_data = json.load(f)

root = load_skill_tree(tree_data)

with open(r"testing\custom_testing_inputs\question_guidelines.json", "r", encoding="utf-8") as f:
  question_guidelines = json.load(f)

inp = InputSchema(
    job_description=jdes,
    skill_tree=root,
    candidate_profile=cp,
    question_guidelines=question_guidelines
)

inp_cv = inp.candidate_profile.model_dump_json(indent=2)
inp_jd = inp.job_description.model_dump_json(indent=2)
inp_skill_tree = inp.skill_tree.model_dump_json(indent=2)
print(inp_cv + "\n")
print(inp_jd + "\n")

print(inp_skill_tree)

with open("config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

graph = AgendaGenerationAgent.get_graph()

otpt = asyncio.run(graph.ainvoke(inp, config))

x = ""
for k, v in otpt.items():
    k = str(k).capitalize()
    print(f"\n{k} --->\n\n {v.model_dump_json(indent=2)}\n")
    x += f"\n{k} --->\n\n {v.model_dump_json(indent=2)}\n"

with open(r"testing\op51.txt", "w", encoding="utf-8") as f:
    f.write(x)

end = time.time_ns()
print(f"\nTime taken: {(end - start) / 60000000000} mins")