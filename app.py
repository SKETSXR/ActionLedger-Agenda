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
# jd_inp_text = '''
# Software Engineer - L3 (Software Engineer II) 

# Location & Type: Delhi , Full – Time 

# Role Overview 

# As a Software Engineer II, you will own and deliver larger cross-cutting modules and systems end-to-end across backend and frontend. You’ll write design docs, mentor junior engineers, lead technical discussions, and ensure reliability of critical features. This role expects strong skills in backend services, APIs, databases, and modern frontend frameworks (React/Next.js). 

# What You’ll Do 

# Own and deliver larger modules/systems that span backend services and frontend applications. 

# Author and present detailed design docs, drive technical discussions and trade-off decisions. 

# Build production-grade services in Node.js and integrate AI systems in Python. 

# Architect high-performance REST/GraphQL APIs, ensure versioning, security, and backward compatibility. 

# Design and optimize schemas in Postgres and MongoDB for scale and availability. 

# Lead development of frontend features in React/Next.js with focus on performance, accessibility, and maintainability. 

# Enforce CI/CD best practices: test automation, deployment readiness, and rollback strategies. 

# Define and monitor observability standards (metrics, logs, alerts) and lead incidents. 

# Mentor and coach junior engineers through reviews, pair programming, and knowledge sharing. 

# Design and roll out multi-layer caching for high-traffic paths, define hit-rate/latency SLOs. 

# Establish cache governance: keys/namespaces, TTL policies, invalidation playbooks, and observability (hit/miss dashboards). 

# Technical Qualifications 

# 3–4 years of professional software engineering experience. 

# Advanced proficiency in Node.js services and Python integrations. 

# Strong experience in REST/GraphQL API design and scaling. 

# Deep knowledge of Postgres schema design, indexing, and query optimization. 

# Hands-on with MongoDB aggregation pipelines and sharding strategies. 

# Proficiency with React/Next.js (or equivalent) for building production UIs. 

# Experience with AWS ECS/ECR and scaling containerized workloads. 

# Strong CI/CD practices and release automation experience. 

# Skilled in diagnosing and fixing production issues using logs, metrics, and traces. 

# Solid system design skills: concurrency, fault tolerance, latency vs. throughput trade-offs. 

# Hands-on with Redis at scale (pipelines, Lua scripts, locks), CDN edge caching, and GraphQL/REST response caching. 

# Deep understanding of consistency vs. freshness trade-offs, idempotency, and rate limiting around cached flows. 

# Nice to Have 

# TypeScript proficiency in both frontend and backend. 

# Kubernetes (EKS) and service mesh (Istio/Linkerd). 

# Infrastructure-as-Code (Terraform/CDK/CloudFormation). 

# Distributed systems patterns (event-driven, CQRS, async messaging). 

# Advanced monitoring/alerting (Prometheus, Grafana, ELK, OpenTelemetry). 

# Experience leading technical spikes, POCs, or cross-team integrations. 

 
    
# '''

with open(r"testing\Ayam\jd.txt", "r", encoding="utf-8") as f:
    jd_inp_text = f.read()
jd_json_string = asyncio.run(parse_jd_text_to_json(jd_inp_text))
if jd_json_string == "JD not contain any text":
    raise("Open AI API not running")
jd = json.loads(jd_json_string)
jdes = JobDescriptionSchema(job_role=jd["job_role"], company_background=jd["company_background"], fundamental_knowledge=jd.get("fundamental_knowledge"))
# print(jdes.model_dump_json(indent=2))

# with open(r"parsed_cv7.json", "r", encoding='utf-8') as c:
#     candidate_profile = json.load(c)

# candidate_profile = asyncio.run(parse_pdf_to_json(r"testing\Ayam\AyamHeniberMeitei_2025L - ayam heniber.pdf"))
# # print(candidate_profile)
# if candidate_profile == "CV does not contain proper text":
#     raise("Open AI API not running")
# candidate_profile = json.loads(candidate_profile)
with open(r"test_cv.json", "r", encoding="utf-8") as f:
    candidate_profile = json.load(f)

cp = CandidateProfileSchema(skills=candidate_profile["skills"],
                            projects=candidate_profile["projects"],
                            experience=candidate_profile["experience"])
# print(cp.model_dump_json(indent=2))

# Skill tree loading using json
def load_skill_tree(tree_json: dict) -> SkillTreeSchema:
    return SkillTreeSchema(**tree_json)

with open(r"skilltree3_priority.json", "r", encoding="utf-8") as f:
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

# Running guidelines
# question_guidelines = {"question_guidelines":[{"question_guidelines": '''
# 	Guidelines:

# Design the case as a realistic situation that requires decision-making, not abstract hypotheticals.

# Questions should encourage the candidate to analyze trade-offs (performance vs cost, scalability vs maintainability, etc.).

# Probe on approach and reasoning, not only on the final outcome.

# Ask about metrics, evaluation criteria, and constraints the candidate would consider.

# Avoid generic "tell me about a challenge" prompts — instead, anchor the case in technical detail relevant to the candidate's skill set or the job role.

# Example frames:

# "If you were designing a [system/problem] under [constraints], how would you decide between option A and option B?"

# "What bottlenecks could appear in [situation], and how would you measure and mitigate them?"

# "How would you adapt your solution if the scale increased by 10x?"''',
#     "question_type_name": "Case study type questions"},
#     {"question_guidelines": '''Guidelines:

# Start with "why" (motivation, design choice, architecture decision), then follow with "how" (implementation, tools, processes).

# Frame questions so they feel grounded in the candidate's project details (P1, P2, etc.).

# Avoid repetitive or generic prompts like "What challenges did you face?" or "How did you solve them?" unless expanded into technical reasoning.

# Direct QAs should resemble peer-to-peer technical reviews, e.g., "In your project where you implemented X, why did you choose Y approach?"

# Encourage the candidate to explain trade-offs, alternatives considered, and design impact.

# Keep the phrasing specific and contextualized to avoid vague storytelling answers.

# Example frames:

# "In [Project], you mentioned using [technology/tool]. Why was it chosen over alternatives?"

# "During [Project], how did you ensure performance/scalability/security in the design?"

# "If you had to extend [Project] today to handle [new requirement], what architectural changes would you propose?"''',
#     "question_type_name": "Project based questions"},
#     {"question_guidelines": '''The counter questions should be of the types: 
#     1. Twist- What would happen if you do A instead of B
#     2. Interrogatory- Why did you use A?''',
#     "question_type_name": "Counter questions"}
# ]}

# Running old
# question_guidelines = {
#   "question_guidelines": [
#     {
#       "question_guidelines": '''Guidelines:\n\n
#                                 Design the case as a realistic situation/scenario that requires decision-making, not abstract hypotheticals.\n\n
#                                 Questions should encourage the candidate to analyze trade-offs (performance vs cost, scalability vs maintainability, etc.).\n\n
#                                 Probe on approach and reasoning, not just on the final outcome.\n\n
#                                 You can include asking about the metrics, evaluation criteria, and constraints the candidate might consider.\n\n
#                                 Avoid generic "tell me about a challenge" initiations, rather anchor the case in technical detail relevant to the candidate's skill set or the job role.\n\n
#                                 Example frames:\n\n
#                                 "If you were designing a [system/problem] under [constraints], how would you decide between option A and option B?"\n\n
#                                 "What bottlenecks could appear in [situation], and how would you measure and mitigate them?"\n\n
#                                 "How would you adapt your solution if the scale increased by 10x?''',
#       "question_type_name": "Case study type questions"
#     },
#     {
#       "question_guidelines": '''Guidelines:\n\n
#                                 Start with "why" (motivation, design choice, architecture decision), then follow with "how" (implementation, tools, processes).\n\n
#                                 Frame questions so they feel grounded in the candidate's project details (P1, P2, etc.).\n\n
#                                 Avoid repetitive or generic initiations like "What challenges did you face?" or "How did you solve them?" unless expanded into technical reasoning.\n\n
#                                 Direct QAs should resemble peer-to-peer technical reviews, e.g., "In your project where you implemented X, why did you choose Y approach?"\n\n
#                                 Encourage the candidate to explain about the trade-offs, alternatives considered, and design impact.\n\n
#                                 Keep the phrasing specific and contextualized to avoid vague storytelling answers.\n\n
#                                 Example frames:\n\n
#                                 "In [Project], you mentioned using [technology/tool]. Why was it chosen over alternatives?"\n\n
#                                 "During [Project], how did you ensure performance/scalability/security in the design?"\n\n
#                                 "If you had to extend a [Project] today for handling a [new requirement], what architectural changes would you propose?''',
#       "question_type_name": "Project based questions"
#     },
#     {
#       "question_guidelines": '''Guidelines:\n\n
#                                 The counter questions should be of the types: \n    
#                                 1. Twist- What would happen if you do A instead of B\n
#                                 2. Interrogatory- Why did you use A?\n\n
#                                 The questions should be more realistic counter questions as if asked in a real interview irrespective of the difficulty''',
#       "question_type_name": "Counter questions"
#     }
#   ]
# }

# # Running good except case studies
# question_guidelines = {
#   "question_guidelines": [
#     {
#       "question_guidelines": '''Guidelines:\n\n
#                                 Definition:
#                                 A case study question presents a hypothetical scenario with specific constraints and asks the candidate to analyze, design, or solve a problem. Probe on approach and reasoning, not just on the final outcome. Avoid generic "tell me about a challenge" initiations, rather anchor the case in technical detail relevant to the candidate's skill set or the job role. The first question always sets the stage with the case and constraints. Follow-up questions explore design choices, trade-offs, and decision-making. 
#                                 Example:
#                                 Case: "You are asked to design a system that handles online ticket booking for concerts. The system should support high traffic during peak booking times but must also minimize operational costs. The constraint is that the budget only allows use of basic cloud services, not premium enterprise ones."
#                                 Q1: How would you design the overall system architecture to meet these constraints?
#                                 Q2: What approach would you take to handle sudden spikes in traffic during a ticket release?
#                                 Q3: If the system starts facing delays in payment confirmation, how would you modify the design?''',
#       "question_type_name": "Case study type questions"
#     },
#     {
#       "question_guidelines": '''Guidelines:\n\n
#                                 Definition:
#                                 These questions are grounded in the candidate's past projects. The interviewer picks specific aspects of the project (requirements, challenges, decisions) and explores them in detail.
#                                 Example:
#                                 "In your last project, you mentioned you worked on a system that processed large amounts of user data. What was the most significant performance bottleneck you faced?"
#                                 "You said you chose a queue-based architecture. Why did you prefer that over a simpler request-response model?"
#                                 ''',
#       "question_type_name": "Project based questions"
#     },
#     {
#       "question_guidelines": '''Guidelines:\n\n
#                                 Counter questions build directly on the candidate's previous answer. While making the examples, implicitly create a sample candidate's response related to the focus areas then create the examples. The counter questions are of two types:
#                                 (a) Interrogatory Counter Questions
#                                 These dig deeper into the reasoning behind a choice the candidate made.
#                                 Example:
#                                 Candidate: "I would use caching to reduce response times."
#                                 Counter: "Why did you choose caching over database optimization first? What risks does caching introduce?"
#                                 (b) Twist Counter Questions
#                                 These modify the scenario or constraints slightly and ask the candidate to reconsider.
#                                 Example:
#                                 Original Q: "How would you design a system for online ticket booking under limited budget constraints?"
#                                 Candidate Answer: "I'd use basic cloud services with auto-scaling."
#                                 Twist Counter: "Now suppose the budget is no longer a constraint, but strict compliance with data privacy laws is required. How does your design change?"''',
#       "question_type_name": "Counter questions"
#     }
#   ]
# }
question_guidelines = {
  "question_guidelines": [
    {
      "question_guidelines": '''Guidelines:\n\n
                                Definition:
                                A case study question presents a hypothetical scenario with specific constraints and asks the candidate to analyze, design, or solve a problem. Probe on approach and reasoning, not just on the final outcome. Avoid generic "tell me about a challenge" initiations, rather anchor the case in technical detail relevant to the candidate's skill set or the job role. The first question always sets the stage with the case and constraints. Follow-up questions explore design choices, trade-offs, and decision-making. 
                                Examples:
                                Case: "You are asked to design a system that handles online ticket booking for concerts. The system should support high traffic during peak booking times but must also minimize operational costs. The constraint is that the budget only allows use of basic cloud services, not premium enterprise ones."
                                Q1: How would you design the overall system architecture to meet these constraints?
                                Q2: What approach would you take to handle sudden spikes in traffic during a ticket release?
                                Q3: If the system starts facing delays in payment confirmation, how would you modify the design?

                                Case: "Design a content delivery platform like YouTube that must serve millions of concurrent streams with minimal buffering. Constraint: bandwidth costs must be optimized."
                                Q1: How would you design the architecture for scalable video streaming?
                                Q2: How would you handle personalized recommendations without overloading latency budgets?
                                Q3: How would you ensure reliability during regional outages?

                                Case: "Design an online collaborative document editor (like Google Docs). Constraint: real-time edits must be synced across devices with <200ms delay."
                                Q1: What data model would you use to handle concurrent edits?
                                Q2: How would you ensure fault tolerance?
                                Q3: How would you secure documents while maintaining low latency?

                                Case: "Design a large-scale ride-hailing service (like Uber). Constraint: system must match riders and drivers in under 2 seconds."
                                Q1: How would you design the matching algorithm and backend services?
                                Q2: How would you ensure fair distribution of drivers during surge hours?
                                Q3: How would you scale real-time location updates globally?

                                Case: "Design an AI-powered fraud detection system for a payments company. Constraint: decisions must happen in under 100ms to avoid checkout friction."
                                Q1: How would you structure the feature pipeline?
                                Q2: How would you balance false positives and false negatives?
                                Q3: How would you update models with new fraud patterns in real time?

                                Case: "Design an AI assistant that summarizes long documents in regulated industries. Constraint: explanations must be human-auditable."
                                Q1: How would you architect the pipeline from ingestion to output?
                                Q2: What approaches would you use for explainability?
                                Q3: How would you handle sensitive data storage and compliance?

                                Case: "Design a recommendation engine for a global video streaming platform. Constraint: responses must be <100ms worldwide."
                                Q1: How would you build the training and serving pipeline?
                                Q2: How would you solve cold-start problems for new users?
                                Q3: How would you ensure fairness and diversity in recommendations?

                                Case: "Design a decentralized exchange (DEX) for crypto tokens. Constraint: transaction finality must occur within 5 seconds."
                                Q1: How would you design the matching engine on-chain or off-chain?
                                Q2: How would you mitigate front-running attacks?
                                Q3: How would you ensure scalability under high trading volume?

                                Case: "Design a blockchain-based voting system. Constraint: privacy must be preserved, but auditability is mandatory."
                                Q1: What architecture would you use for secure vote casting?
                                Q2: How would you balance transparency with voter anonymity?
                                Q3: How would you prevent double voting or fraud?

                                Case: "Design an in-memory key-value store like Redis. Constraint: it must support persistence without high latency penalties."
                                Q1: How would you structure data storage and snapshots?
                                Q2: How would you handle replication for high availability?
                                Q3: How would you prevent memory fragmentation under heavy usage?

                                Case: "Design a thread scheduler for an OS kernel. Constraint: it must optimize for both latency and throughput."
                                Q1: What scheduling algorithm would you use?
                                Q2: How would you manage priority inversion?
                                Q3: How would you test your scheduler under diverse workloads?

                                Case: "Design a distributed file storage system like Google Drive. Constraint: strong consistency for file versions must be maintained."
                                Q1: How would you architect the metadata and storage layers?
                                Q2: How would you handle concurrent edits and conflicts?
                                Q3: How would you recover from partial data center outages?

                                Case: "Design a smart home IoT network with 10,000 devices. Constraint: devices must operate with limited power and intermittent connectivity."
                                Q1: How would you design the communication protocol?
                                Q2: How would you secure data transmissions from edge devices?
                                Q3: How would you handle updates to all devices efficiently?

                                Case: "Design a fleet management system for autonomous delivery drones. Constraint: drones must stay connected even in poor network areas."
                                Q1: How would you architect control and telemetry?
                                Q2: How would you ensure safe failover when connectivity drops?
                                Q3: How would you optimize routing in real time?

                                Case: "Design a connected healthcare monitoring system. Constraint: data must be transmitted securely and in near real time."
                                Q1: How would you design the edge device data pipeline?
                                Q2: How would you ensure HIPAA/GDPR compliance?
                                Q3: How would you handle network dropouts for critical alerts?

                                Case: "Design a log analytics platform like Splunk. Constraint: it must ingest 10M events/sec and allow near real-time querying."
                                Q1: How would you architect the ingestion pipeline?
                                Q2: How would you design the storage layer for both cost and speed?
                                Q3: How would you handle query optimization under heavy load?

                                Case: "Design a recommendation system for a global e-commerce company. Constraint: responses must be <100ms for users worldwide."
                                Q1: How would you design the serving layer with caching?
                                Q2: How would you deal with cold start problems?
                                Q3: How would you scale across regions while keeping personalization accurate?

                                Case: "New York City has hired you to determine what optimal route or what destination taxi drivers should go to when they do not have a customer."
                                Q1: What data sources would you use to determine hotspots of passenger demand?
                                Q2: How would you design the system to recommend destinations in real time?
                                Q3: How would you adapt the model as traffic, weather, and events change throughout the day?

                                Case: "Design a real-time analytics dashboard for IoT devices. Constraint: latency must be <1 second for 1M events/sec."
                                Q1: How would you design the stream processing pipeline?
                                Q2: How would you ensure durability of raw event data?
                                Q3: How would you allow flexible queries without affecting throughput?

                                Case: "Design a secure password manager system. Constraint: all data must be zero-knowledge encrypted."
                                Q1: How would you architect storage of secrets?
                                Q2: How would you handle cross-device synchronization?
                                Q3: How would you detect or prevent account takeovers?

                                Case: "Design a DDoS mitigation system for a global web service. Constraint: it must absorb traffic surges 100x normal load."
                                Q1: What layers of defense would you apply?
                                Q2: How would you distinguish malicious vs. legitimate traffic?
                                Q3: How would you test resilience before attacks happen?

                                Case: "Design a secure multi-factor authentication (MFA) system for millions of users. Constraint: low friction while resisting phishing and SIM-swap attacks."
                                Q1: How would you architect enrollment and verification?
                                Q2: How would you secure secret delivery and backups?
                                Q3: How would you balance usability with strong security guarantees?"
                                ''',
      "question_type_name": "Case study type questions"
    },
    {
      "question_guidelines": '''Guidelines:\n\n
                                Definition:
                                These questions are grounded in the candidate's past projects. The interviewer picks specific aspects of the project (requirements, challenges, decisions) and explores them in detail.
                                Example:
                                "In your last project, you mentioned you worked on a system that processed large amounts of user data. What was the most significant performance bottleneck you faced?"
                                "You said you chose a queue-based architecture. Why did you prefer that over a simpler request-response model?"
                                ''',
      "question_type_name": "Project based questions"
    },
    {
      "question_guidelines": '''Guidelines:\n\n
                                Counter questions build directly on the candidate's previous answer. While making the examples, implicitly create a sample candidate's response related to the focus areas then create the examples. The counter questions are of two types:
                                (a) Interrogatory Counter Questions
                                These dig deeper into the reasoning behind a choice the candidate made.
                                Example:
                                Candidate: "I would use caching to reduce response times."
                                Counter: "Why did you choose caching over database optimization first? What risks does caching introduce?"
                                (b) Twist Counter Questions
                                These modify the scenario or constraints slightly and ask the candidate to reconsider.
                                Example:
                                Original Q: "How would you design a system for online ticket booking under limited budget constraints?"
                                Candidate Answer: "I'd use basic cloud services with auto-scaling."
                                Twist Counter: "Now suppose the budget is no longer a constraint, but strict compliance with data privacy laws is required. How does your design change?"''',
      "question_type_name": "Counter questions"
    }
  ]
}
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

with open(r"testing\op51.txt", "w", encoding="utf-8") as f:
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