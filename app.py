import json
import pprint
import yaml
import time
import asyncio
from src.agent.AgendaGenerationAgent import AgendaGenerationAgent
from src.schema.input_schema import JobDescriptionSchema, CandidateProfileSchema, SkillTreeSchema, InputSchema, QuestionGuidelinesSchema
# from src.schema.output_schema import OutputSchema
from src.tools.jd_extraction import parse_jd_text_to_json
from src.tools.cv_extraction import parse_pdf_to_json
from dotenv import load_dotenv
# from src.db_fetch import read_start

start = time.time_ns()
load_dotenv()

# with open(r"new_jd.json", "r", encoding='utf-8') as j:
#     jd = json.load(j)
# jd_inp_text = """We are looking for a skilled Data Scientist to join our team at Tech Innovators Inc. The ideal candidate will have a strong background in statistical analysis, machine learning, and data visualization. Responsibilities include analyzing large datasets to extract insights, building predictive models, and collaborating with cross-functional teams to implement data-driven solutions. A bachelor's degree in Computer Science, Statistics, or a related field is required, with a preference for candidates holding a master's degree or higher. Experience with Python, R, SQL, and cloud platforms such as AWS or Azure is essential. Familiarity with big data technologies like Hadoop and Spark is a plus. Join us at Tech Innovators Inc., a leading company in the tech industry known for its innovative solutions and dynamic work environment."""
jd_inp_text = '''
AI/ML Engineer - L2 (Engineer I) 

Location & Type: Delhi, Full-time 

Role Overview 

Own ML features and  end-to-end  data prep, training, evaluation, and production inference with clear SLOs for latency, throughput, and model quality. . 

What You’ll Do 

Design and ship E2E ML features. 

Build and operate model-serving APIs (REST/GraphQL/gRPC), implement batching, retries, and circuit breakers. 

Optimize inference via model/graph optimizations (e.g., quantization, distillation, ONNX/TensorRT or equivalents). 

Implement drift detection (data/label/embedding) and schedule versioned retraining. 

Ensure reproducibility with experiment and data/model versioning, maintain clean runbooks. 

Collaborate on data contracts and reliable feature pipelines with product/data/platform teams. 

Contribute to on-call for ML services you own,write postmortems and harden guardrails. 

Technical Qualifications 

2–3 years in ML engineering - strong Python and one major DL framework (PyTorch or TensorFlow). 

Solid data manipulation (NumPy/Pandas or equivalent) and evaluation design (precision/recall, ROC-AUC, calibration). 

Production serving experience , API design understanding. 

Comfortable with relational and NoSQL data stores, can design efficient schemas/queries for features and logs. 

Containers + CI/CD basics (Docker/Kubernetes or managed equivalents), can ship, roll back, and instrument services. 

Monitoring/observability (logs, metrics, traces) for both system and model quality signals. 

Working knowledge of caching and when not to cache. 

Nice to Have 

Experience with LLM adapters , embeddings + vector stores , and RAG patterns. 

Orchestration/ETL tools , experiment tracking, and data versioning. 

Practical cost/performance tuning (GPU utilization, mixed precision). 

 
    
'''
jd_json_string = asyncio.run(parse_jd_text_to_json(jd_inp_text))
if jd_json_string == "JD not contain any text":
    raise("Open AI API not running")
jd = json.loads(jd_json_string)
jdes = JobDescriptionSchema(job_role=jd["job_role"], company_background=jd["company_background"], fundamental_knowledge=jd.get("fundamental_knowledge"))
# print(jdes.model_dump_json(indent=2))

# with open(r"parsed_cv7.json", "r", encoding='utf-8') as c:
#     candidate_profile = json.load(c)

candidate_profile = asyncio.run(parse_pdf_to_json(r"testing\Sreelal\Sreelal_H_Resume (5) - Sreelal H.pdf"))
# print(candidate_profile)
if candidate_profile == "CV does not contain proper text":
    raise("Open AI API not running")
candidate_profile = json.loads(candidate_profile)
cp = CandidateProfileSchema(skills=candidate_profile["skills"],
                            projects=candidate_profile["projects"],
                            experience=candidate_profile["experience"])
# print(cp.model_dump_json(indent=2))

# Skill tree loading using json
def load_skill_tree(tree_json: dict) -> SkillTreeSchema:
    return SkillTreeSchema(**tree_json)

with open(r"testing\Sreelal\skill_tree.json", "r") as f:
    tree_data = json.load(f)

root = load_skill_tree(tree_data)
# question_guidelines = {"question_guidelines":[{"question_guidelines": '''
# 	Minimize questions like what difficulties did you face, how you achieved them etc. Focus on technical aspects of why and how things were done.
# 	In projects, try starting with why and then go to how
# 	The direct questions or QAs for projects should feel like questions are based on their project. They should be "As per you project details, you used A, why did you do so?"''',
#     "question_type_name": "Case study type questions"},
#     {"question_guidelines": '''Minimize questions like what difficulties did you face, how you achieved them etc. Focus on technical aspects of why and how things were done.
# 	In projects, try starting with why and then go to how
# 	The direct questions or QAs for projects should feel like questions are based on their project. They should be "As per you project details, you used A, why did you do so?"''',
#     "question_type_name": "Project based questions"},
#     {"question_guidelines": '''The counter questions should be of the types: 
#     1. Twist- What would happen if you do A instead of B
#     2. Interrogatory- Why did you use A?''',
#     "question_type_name": "Counter questions"}
#     ]}
question_guidelines = {"question_guidelines":[{"question_guidelines": '''
	Guidelines:

Design the case as a realistic situation that requires decision-making, not abstract hypotheticals.

Questions should encourage the candidate to analyze trade-offs (performance vs cost, scalability vs maintainability, etc.).

Probe on approach and reasoning, not only on the final outcome.

Ask about metrics, evaluation criteria, and constraints the candidate would consider.

Avoid generic "tell me about a challenge" prompts — instead, anchor the case in technical detail relevant to the candidate's skill set or the job role.

Example frames:

"If you were designing a [system/problem] under [constraints], how would you decide between option A and option B?"

"What bottlenecks could appear in [situation], and how would you measure and mitigate them?"

"How would you adapt your solution if the scale increased by 10x?"''',
    "question_type_name": "Case study type questions"},
    {"question_guidelines": '''Guidelines:

Start with "why" (motivation, design choice, architecture decision), then follow with "how" (implementation, tools, processes).

Frame questions so they feel grounded in the candidate's project details (P1, P2, etc.).

Avoid repetitive or generic prompts like "What challenges did you face?" or "How did you solve them?" unless expanded into technical reasoning.

Direct QAs should resemble peer-to-peer technical reviews, e.g., "In your project where you implemented X, why did you choose Y approach?"

Encourage the candidate to explain trade-offs, alternatives considered, and design impact.

Keep the phrasing specific and contextualized to avoid vague storytelling answers.

Example frames:

"In [Project], you mentioned using [technology/tool]. Why was it chosen over alternatives?"

"During [Project], how did you ensure performance/scalability/security in the design?"

"If you had to extend [Project] today to handle [new requirement], what architectural changes would you propose?"''',
    "question_type_name": "Project based questions"},
    {"question_guidelines": '''The counter questions should be of the types: 
    1. Twist- What would happen if you do A instead of B
    2. Interrogatory- Why did you use A?''',
    "question_type_name": "Counter questions"}
]}
# question_guidelines["question_type_name"] = ["Case study type questions", "Project based questions"]

inp = InputSchema(
    job_description=jdes,
    skill_tree=root,
    candidate_profile=cp,
    question_guidelines=question_guidelines
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
# config["configurable"]["thread_id"]= "thread_12"

# with open('config.yaml', 'w') as file:
#     yaml.safe_dump(config, file)

# otpt = graph.invoke(inp, config)
otpt = asyncio.run(graph.ainvoke(inp, config))

# Test database fetch (having jd, cv, skill tree, summary)
# read_start()

# Test o/p as a schema object (for entire agenda output)
# print(otpt)

x = ""
for k, v in otpt.items():
    k = str(k).capitalize()
    print(f"\n{k} --->\n\n {v.model_dump_json(indent=2)}\n")
    x += f"\n{k} --->\n\n {v.model_dump_json(indent=2)}\n"

with open(r"testing\op11.txt", "a") as f:
    f.write(x)
# for i, v in otpt.items():
#     if hasattr(v, "model_dump_json"):
#         print(json.dumps(json.loads(v.model_dump_json()), indent=2))
#     elif hasattr(v, "model_dump"):
#         print(json.dumps(v.model_dump(), indent=2))
#     else:
#         pprint.pprint(v, indent=2)
end = time.time_ns()
print(f"\nTime taken: {(end - start) / 60000000000} mins")