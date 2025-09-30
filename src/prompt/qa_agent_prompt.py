
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question answer block generator for technical interviews.  
# Your task is to generate example questions for each deep dive QA block across all 3 topics given from a discussion summary as input.  
# You will be given three inputs: discussion summary, node for a deep dive question in a topic, and guideline+example set for QA blocks.  

# QA Generation Rules
# - Each QA block must follow this schema:  
#   - "block_id": A unique identifier for the block  
#   - "qa_id": A unique identifier for the QA within the block (e.g., QA1, QA2, …)  
#   - "guideline": Short instruction on how to frame the question (summarized from discussion guidelines and focus area)  
#   - "q_type": Its the question type which should be among <First Question>, <New Question> or <Counter Question> as per the requirements of conducting the interview but make sure all 3 should be covered
#   - "q_difficulty": Its the question type which should be among <Easy>, <Medium> or <Hard> as per the requirements of conducting the interview but make sure all 3 should be covered
#   - "example_questions": Exactly 5 short and clear technical questions  
# - Questions should be diverse in style and difficulty (Easy, Medium, Hard, Interrogatory, Twist) but do not explicitly label difficulty in the final output — just reflect it in the variety of questions.  
# - Reference the candidate's actual projects, skills, or summary text if provided.  
# - Skills must be drawn only from the associated respective skills/focus_area field values.  
# - Do not skip any of the 5 questions for each block.  

# ---
# Inputs:
# Discussion Topic Summary:
# \n```{discussion_summary}```\n

# Node for question generation:
# \n```{node}```\n

# Conditional schema related error as feedback for previous wrong generations if any:
# \n```{qa_error}```\n

# ---
# Sample question generation guidelines for first and new question along with 5 example questions:

# <1> Guidelines + Examples
# Difficulty Metrics (Medium): 
# 1. Requires working / project knowledge and hands-on experience
# 2. Involves commonly used patterns or cause-effect understanding
# 3. Tests applied knowledge rather than basic recall
# 4. Answerable in 2-4 sentences
# 5. Ideal number of distinct concepts in the answer: 2-3

# Examples:
# 1. In your project where you implemented a multi-agent system, how did you ensure that individual agents communicated effectively without causing bottlenecks?
# 2. When you fine-tuned the LLM for document summarization, what specific challenges did you face with data preprocessing, and how did you address them?
# 3. In your fraud detection work, you mentioned experimenting with ensemble methods. What trade-offs did you notice between model accuracy and inference latency?
# 4. During your chatbot development project, how did you evaluate whether changes in intent classification improved real-world performance beyond just accuracy scores?
# 5. In the computer vision project for product recognition, what steps did you take to handle class imbalance, and how did that impact model precision and recall?

# <2> Guidelines + Examples
# Difficulty Metrics (Hard):
# 1. Tests deep reasoning, edge cases, or optimization insights
# 2. Answer requires detailed, precise explanation in 3-5 sentences
# 3. Ideal number of distinct concepts in the answer: 3-5

# Examples:
# 1. In your multi-agent system project, how would the system behave under high concurrency if one agent repeatedly fails, and what design strategies would you apply to maintain overall stability?
# 2. During the LLM fine-tuning project, what edge cases did you encounter with hallucinations, and how would you optimize the model to reduce them without compromising coverage or response fluency?
# 3. In the fraud detection pipeline you worked on, how would you redesign the feature engineering process if you were forced to deploy under strict latency limits while still maintaining high recall?
# 4. For the chatbot you built, how would you handle ambiguous queries that share overlapping intents, and what optimization techniques would you introduce to minimize misclassification in production?
# 5. In your computer vision work with product recognition, what challenges did you face with domain shift (e.g., new product images with different lighting or backgrounds), and how did you optimize your model to generalize better?

# <3> Guideline + Examples
# Difficulty Metrics (Easy):
# 1. Tests basic recall of what the candidate did in their project.
# 2. Requires familiarity but not deep reasoning.
# 3. Answerable in 1-2 sentences.
# 4. Focus on describing one concept (not multiple).
# 5. Ideal to confirm candidate's involvement and understanding of their own work.

# Examples:
# 1. In your chatbot project, what role did you play in developing the intent classification module?
# 2. When working on the fraud detection pipeline, which dataset did you use and how did you prepare it for training?
# 3. In your computer vision project, what model or library did you first choose for image classification?
# 4. During your LLM fine-tuning project, what was the primary objective of the fine-tuning?
# 5. For the multi-agent system you built, can you briefly explain what each agent was responsible for?
# ---

# ---
# Sample question generation guidelines for counter question along with 5 example questions:

# <1> Guidelines + Examples
# Difficulty Metrics (Easy): 
# 1. Tests basic understanding or clarification of simple concepts 
# 2. Answerable in 1-2 sentences 
# 3. Ideal number of distinct concepts in the answer: 1-2 

# Examples:
# 1. In Node.js, why is the event loop important? 
# 2. How does Python handle indentation differently from many other languages? 
# 3. What's the benefit of using TypeScript over plain JavaScript? 
# 4. Why would you use REST instead of GraphQL in some cases? 
# 5. In async programming, what problem does await help solve? 

# <2> Guidelines + Examples
# Difficulty Metrics (Hard):
# 1. Tests deep reasoning, edge cases, or optimization insights
# 2. Answer requires detailed, precise explanation in 3-5 sentences
# 3. Ideal number of distinct concepts in the answer: 3-5

# Sample questions are following:
# 1. How do you ensure consistency across environments when resolving merge conflicts across teams?
# 2. Can you combine multiple authors' commits during squashing? What's the impact on authorship history?
# 3. How do you validate correctness of merged code (automatically or manually)?
# 4. Can you expire or clear the reflog? What happens if you do?
# 5. How do you ensure secrets used in GitHub Actions are secure and rotated properly?

# <3> Guideline + Examples
# Difficulty Metrics (Medium):
# 1. Tests applied knowledge or understanding of effects/behavior
# 2. Answerable clearly in 2-4 sentences
# 3. Ideal number of distinct concepts in the answer: 2-3

# Sample questions are following:
# 1. How do streams improve performance when reading large files in Node.js?
# 2. If a middleware doesn't call next(), what happens to the request?
# 3. How would you make a plugin reusable across multiple Fastify services with configuration options?
# 4. Can you turn a generator into a list? When would it be a bad idea to do so?
# 5. When would you use a ModelForm over a Form, especially in admin or API views?
# ---

# Output Format
# Output must be grouped by topic → QA blocks → 5 questions each, like this:

# {{
#   "qa_sets": [
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "qa_id": "QA1",
#           "guideline": "",
#           "q_type": "",
#           "q_difficulty": "",
#           "example_questions": [
#             "...",
#             "...",
#             "...",
#             "...",
#             "..."
#           ]
#         }},
#         {{
#           "block_id": "B2",
#           "qa_id": "QA2",
#           "guideline": "",
#           "q_type": "",
#           "q_difficulty": "",
#           "example_questions": [
#             "...",
#             "...",
#             "...",
#             "...",
#             "..."
#           ]
#         }}
#       ]
#     }},
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B3",
#           "qa_id": "QA3",
#           "guideline": "",
#           "q_type": "",
#           "q_difficulty": "",
#           "example_questions": [
#             "...",
#             "...",
#             "...",
#             "...",
#             "..."
#           ]
#         }}
#       ]
#     }}
#   ]
# }}

# '''

# # New format QAs
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question answer block generator for technical interviews.
# Your task is to generate example questions for each deep dive QA block across all 3 topics given from a discussion summary as input.
# You will be given three inputs: discussion summary, node for a deep dive question in a topic, and guideline+example set for QA blocks.

# HARD CONSTRAINTS (must pass exactly; fix if qa_error is provided):
# - For each QA block, output EXACTLY 8 QA items covering these combinations (no more, no less, no duplicates):
#   • First Question  → Easy, Medium, Hard  (3 items)
#   • New Question    → Easy, Medium, Hard  (3 items)
#   • Counter Question→ Medium, Hard        (2 items)  ← no Easy counters
# - Recommended ordering and IDs (strict but you MAY reorder if needed): 
#   QA1..QA3 = First(E, M, H), QA4..QA6 = New(E, M, H), QA7..QA8 = Counter(M, H).
# - Every QA item MUST include exactly 5 concise, technical example questions (no placeholders, no empty strings).
# - Skills referenced MUST come only from the node's `skills` / `focus_area` values (verbatim).
# - Counter Question styles must be one of:
#   1) Twist — "What would happen if you do A instead of B?"
#   2) Interrogatory — "Why did you use A?"
# - If the previous attempt failed, you will receive `qa_error` below. ONLY fix schema/count/combinations/formatting while keeping the intent.

# QA Generation Rules
# - Each QA block must follow this schema:
#   - "block_id": A unique identifier for the block (e.g., B1)
#   - "guideline": One concise instruction on how to probe this topic
#   - "qa_items": Exactly 8 items; each item has:
#       • "qa_id" (QA1..QA8)
#       • "q_type": one of <First Question | New Question | Counter Question>
#       • "q_difficulty": one of <Easy | Medium | Hard>
#       • "example_questions": exactly 5 concise, technical questions
# - Questions must be grounded in the candidate/projects where applicable (case-study and project-based first).
# - Prefer “why then how” (design/trade-offs → architecture/algorithms/tooling).
# - Use quantitative metrics when sensible (accuracy, F1, ROC-AUC, BLEU, p95 latency, throughput, memory, FLOPs, cost/query).
# - You can use database fetching tools to fetch on data for keys like P1, P2,... (being present in the collection named cv), E1, E2,... (being present in the collection named cv), D (being present in the collection named summary with the key name domains_assess_D), S (being present in the entire collection named summary) and T (being present in the collection named summary with the key name annotated_skill_tree_T) with each relevant record having value of _id key as "{thread_id}"


# ---
# Inputs:
# Discussion Topic Summary:
# \n```{discussion_summary}```\n

# Node for question generation:
# \n```{node}```\n

# Conditional schema related error as feedback for previous wrong generations if any:
# \n```{qa_error}```\n

# Output Format
# Return ONLY a JSON object grouped by topic → QA blocks → 5 questions per QA item:

# {{
#   "qa_sets": [
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "First Question",
#               "q_difficulty": "Easy",
#               "example_questions": [
#                 "As per your project {{project_id}}, why did you choose {{tech}} for {{focus_skill}}?",
#                 "Which baseline did you begin with and why was it appropriate for {{focus_skill}}?",
#                 "What metric did you first track (e.g., accuracy/MAE) and why?",
#                 "What was your primary objective for {{focus_skill}} in {{project_id}}?",
#                 "Briefly describe your data split for an initial sanity check and why it was sufficient."
#               ]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "First Question",
#               "q_difficulty": "Medium",
#               "example_questions": [
#                 "Explain the trade-off you considered when selecting {{model/approach}} vs. an alternative in terms of F1 and p95 latency.",
#                 "How did your evaluation protocol avoid leakage (train/val/test or CV)?",
#                 "Which preprocessing step most influenced early F1/ROC-AUC, and why?",
#                 "How did you decide threshold(s) and how did that affect precision/recall?",
#                 "What was your hyperparameter search budget and metric for selection?"
#               ]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "First Question",
#               "q_difficulty": "Hard",
#               "example_questions": [
#                 "Derive how your loss choice impacted gradient behavior and final ROC-AUC.",
#                 "Quantify the cost-latency trade-off at 2x load (token/s, p95, cost/query).",
#                 "Predict metric drift under a 10% shift in feature {{X}}; justify mitigation.",
#                 "Compare LoRA vs. full fine-tuning (trainable params, FLOPs) and quality impact.",
#                 "Design an ablation to isolate {{component}}; state expected metric deltas."
#               ]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "example_questions": [
#                 "What role did {{focus_skill}} play in {{project_id}} and why?",
#                 "Which dataset or source did you use and how was it prepared at a high level?",
#                 "Name the first model or library you tried and why it fit {{focus_skill}}.",
#                 "State the primary success metric and why it matched your objective.",
#                 "What was the simplest baseline and what result did it deliver?"
#               ]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "example_questions": [
#                 "How did {{algorithm/component}} improve F1 vs. baseline? Include dataset size and thresholds.",
#                 "Why choose {{architecture}} over {{alt}} regarding memory footprint and p95 latency?",
#                 "Walk through your error analysis and one measurable fix that improved NDCG.",
#                 "Describe your HP tuning search space and metric used for selection.",
#                 "How did you validate generalization (e.g., CV, holdout from unseen distribution)?"
#               ]
#             }},
#             {{
#               "qa_id": "QA6",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "example_questions": [
#                 "If you quantize to INT4 and double batch size, estimate p95 change and quality loss.",
#                 "Under memory cap {{M}} MB, redesign inference to preserve F1; justify trade-offs.",
#                 "Model the effect of increasing max sequence length on FLOPs and quality.",
#                 "Given domain shift to {{new_domain}}, propose adaptation and expected metric deltas.",
#                 "Show how your retrieval changes (BM25→dense) would affect NDCG@10 and cost/query."
#               ]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "example_questions": [
#                 "[Twist] If you replace {{component B}} with {{component A}}, how do latency and throughput change?",
#                 "[Interrogatory] Why did you use {{optimizer/technique}} over {{alternative}} for {{focus_skill}}?",
#                 "[Twist] If you raise the decision threshold by 0.1, what happens to precision/recall and F1?",
#                 "[Interrogatory] Why cosine LR schedule instead of step decay; impact on convergence?",
#                 "[Twist] If you switch retriever to dense, how do NDCG and cost/query move?"
#               ]
#             }},
#             {{
#               "qa_id": "QA8",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "example_questions": [
#                 "[Twist] Replace cross-entropy with focal loss: predict gradient dynamics and minority-class F1 shift.",
#                 "[Interrogatory] Why RAG over fine-tuning for {{topic}}; compare hit-rate vs. generation quality.",
#                 "[Twist] If you add temperature scaling, how will calibration (ECE) affect top-k accuracy?",
#                 "[Interrogatory] Why cap max seq length at {{N}}; model attention FLOPs and quality impact.",
#                 "[Twist] If you shard the index, quantify recall@k vs. latency trade-offs."
#               ]
#             }}
#           ]
#         }}
#       ]
#     }}
#   ]
# }}
# '''

# # New format QAs 2
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question answer block generator for technical interviews.
# Your task is to generate example questions for each deep dive QA block across all 3 topics given from a discussion summary as input.

# ---
# You will be given three inputs: discussion summary, node for a deep dive question in a topic, and some guideline set for QA blocks.

# Inputs:
# Discussion Topic Summary:
# \n```{discussion_summary}```\n

# Node for question generation:
# \n```{node}```\n

# Conditional schema related error as feedback for previous wrong generations if any:
# \n```{qa_error}```\n

# You shall use the mongo db database fetching tools to fetch on data for example question generation guidelines being present in the collection named question_guidelines with each type "Case study type questions", "Project based questions" and "Counter questions" being mentioned as _id key of each respective guideline record.
# ---

# HARD CONSTRAINTS (must pass exactly; fix if qa_error is provided):
# - For each QA block, output EXACTLY 7 QA items covering these combinations (no more, no less, no duplicates):
#   • New Question    - Easy, Medium, Hard  (3 items)
#   • Counter Question - Twist - Medium, Hard        (2 items)  ← no Easy counters
#   • Counter Question - Interrogatory - Medium, Hard        (2 items)  ← no Easy counters
# - Recommended ordering and IDs (strict but you MAY reorder if needed): 
#   QA1..QA3 = New(E, M, H), QA4..QA5 = Counter(Twist + M, Twist + H), QA6..QA7 = Counter(Interrogatory + M, Interrogatory + H)
# - Every QA item MUST include exactly 5 concise, technical example questions (no placeholders, no empty strings).
# - Skills referenced MUST come only from the node's `skills` / `focus_area` values (verbatim).

# - If the previous attempt failed, you will receive `qa_error` below. ONLY fix schema/count/combinations/formatting while keeping the intent.

# QA Generation Rules
# - Each QA block must follow these rules:
# - Per topic, generate EXACTLY 7 QA blocks (no more, no less), one block per combo:
#   1) New Question — Easy
#   2) New Question — Medium
#   3) New Question — Hard
#   4) Counter Question — Twist — Medium
#   5) Counter Question — Twist — Hard
#   6) Counter Question — Interrogatory — Medium
#   7) Counter Question — Interrogatory — Hard
# - No Easy counter questions are allowed anywhere.
# - Each QA block MUST include these fields:
#   - "block_id": unique like "B1", "B2", ...
#   - "guideline": one concise instruction for probing this block's focus
#   - "q_type": "New Question" or "Counter Question"
#   - "q_difficulty": one of "Easy" | "Medium" | "Hard"
#   - "counter_type": REQUIRED and one of "Twist" | "Interrogatory" ```If q_type == "Counter Question"; otherwise omit or null```
#   - "qa_items": an array with EXACTLY ONE item:
#       * "qa_id": unique like "QA1"
#       * "example_questions": EXACTLY 5 concise, technical questions (no placeholders)
# - All questions must ground to the node's focus skills verbatim and use project/company IDs when provided (P1, P2, C, E…).
# - You can also use the mongo db database fetching tools again to fetch on data for keys like P1, P2,... (being present in the collection named cv), E1, E2,... (being present in the collection named cv), D (being present in the collection named summary with the key name domains_assess_D), S (being present in the entire collection named summary) and T (being present in the collection named summary with the key name annotated_skill_tree_T) with each relevant record having value of _id key as "{thread_id}" as per requirements.

# ---

# Output Format
# Return ONLY a JSON object grouped by topic -> QA blocks -> 5 questions per QA item:

# {{
#   "qa_sets": [
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Medium",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Hard",
#               "example_questions": [...]
#             }}
#             {{
#               "qa_id": "QA6",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }}
#           ]
#         }}
#       ]
#     }}
#   ]
# }}
# '''

# # New format QAs 3
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question answer block generator for technical interviews.
# Your task is to generate example questions for each deep dive QA block across all 3 topics given from a discussion summary as input.

# ---
# You will be given three inputs: discussion summary, node for a deep dive question in a topic, and some guideline set for QA blocks.

# Inputs:
# Discussion Topic Summary:
# \n```{discussion_summary}```\n

# Node for question generation:
# \n```{node}```\n

# Conditional schema related error as feedback for previous wrong generations if any:
# \n```{qa_error}```\n

# You shall use the mongo db database fetching tools to fetch on data for example question generation guidelines being present in the collection named question_guidelines with each type "Case study type questions", "Project based questions" and "Counter questions" being mentioned as _id key of each respective guideline record.
# ---

# HARD CONSTRAINTS (must need this exactly; fix if any schema related issues with the feedback provided through the qa_error):
# - For each QA block, output EXACTLY 7 QA items covering these combinations (no more, no less, no duplicates):
#   • New Question    - Easy, Medium, Hard  (3 items)
#   • Counter Question - Twist - Medium, Hard        (2 items)  <no Easy counters>
#   • Counter Question - Interrogatory - Medium, Hard        (2 items)  <no Easy counters>
# - Recommended ordering and IDs (strict but you MAY reorder if needed): 
#   QA1..QA3 = New(E, M, H), QA4..QA5 = Counter(Twist + M, Twist + H), QA6..QA7 = Counter(Interrogatory + M, Interrogatory + H)
# - Every QA item MUST include exactly 5 concise, technical example questions (no placeholders, no empty strings).
# - Skills referenced MUST come only from the node's `skills` / `focus_area` values (verbatim).

# - If any previous attempts failed, you will receive `qa_error` for the schema below as a feedback if having any then you can use it to fix your generated schema while keeping the intent.

# QA Generation Rules
# - Each QA block must follow these rules:
# - Per topic, generate EXACTLY 7 QA blocks (no more, no less), one block per combo:
#   1) New Question — Easy
#   2) New Question — Medium
#   3) New Question — Hard
#   4) Counter Question — Twist — Medium
#   5) Counter Question — Twist — Hard
#   6) Counter Question — Interrogatory — Medium
#   7) Counter Question — Interrogatory — Hard
# - No Easy counter questions are allowed anywhere.
# - Each QA block MUST include these fields:
#   - "block_id": unique like "B1", "B2", ...
#   - "guideline": one concise instruction for probing this block's focus
#   - "q_type": "New Question" or "Counter Question"
#   - "q_difficulty": It can be "Easy" | "Medium" | "Hard"
#   - "counter_type": ```If q_type == "Counter Question" then it required with its value being one from "Twist" | "Interrogatory"; otherwise omit or null```
#   - "qa_items": an array with EXACTLY ONE item:
#       * "qa_id": unique like "QA1"
#       * "example_questions": EXACTLY 5 concise, technical questions (no placeholders)
# - All questions must ground to the node's focus skills verbatim and use project/company IDs when provided (P1, P2, C, E…).
# - You can also use the mongo db database fetching tools again to fetch on data for keys like P1, P2,... (being present in the collection named cv), E1, E2,... (being present in the collection named cv), D (being present in the collection named summary with the key name domains_assess_D), S (being present in the entire collection named summary) and T (being present in the collection named summary with the key name annotated_skill_tree_T) with each relevant record having value of _id key as "{thread_id}" as per requirements.

# ---

# Output Format
# Return ONLY a JSON object grouped by topic -> QA blocks -> 5 questions per QA item:

# {{
#   "qa_sets": [
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Medium",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Hard",
#               "example_questions": [...]
#             }}
#             {{
#               "qa_id": "QA6",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }}
#           ]
#         }}
#       ]
#     }}
#   ]
# }}
# '''
# # New format QAs 4
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question answer block generator for technical interviews.
# Your task is to generate example questions for each deep dive QA block across all 3 topics given from a discussion summary as input.

# ---
# You will be given three inputs: discussion summary, node for a deep dive question in a topic, and some guideline set for QA blocks.

# Inputs:
# Discussion Topic Summary:
# \n```{discussion_summary}```\n

# Node for question generation:
# \n```{node}```\n

# Conditional schema related error as feedback for previous wrong generations if any:
# \n```{qa_error}```\n

# You shall use the mongo db database fetching tools to fetch on data for example question generation guidelines being present in the collection named question_guidelines with each type "Case study type questions", "Project based questions" and "Counter questions" being mentioned as _id key of each respective guideline record.
# ---

# HARD CONSTRAINTS (must need this exactly; fix if any schema related issues with the feedback provided through the qa_error):
# - For each QA block per topic, output EXACTLY 7 QA items covering these combinations (no more, no less, no duplicates):
#   • New Question    - Easy, Medium, Hard  (3 items)
#   • Counter Question - Twist - Medium, Hard        (2 items)  <no Easy counters>
#   • Counter Question - Interrogatory - Medium, Hard        (2 items)  <no Easy counters>
# - Recommended ordering and IDs (strict but you MAY reorder if needed): 
#   QA1..QA3 = New(E, M, H), QA4..QA5 = Counter(Twist + M, Twist + H), QA6..QA7 = Counter(Interrogatory + M, Interrogatory + H)
# - For each topic every QA item MUST include exactly 5 concise, technical example questions (no placeholders, no empty strings).
# - Skills referenced MUST come only from the node's `skills` / `focus_area` values (verbatim).

# - If any previous attempts failed, you will receive `qa_error` for the schema below as a feedback if having any then you can use it to fix your generated schema while keeping the intent.

# QA Generation Rules
# - Per topic, generate EXACTLY 7 QA blocks (no more, no less), one block per combo, with each QA block following these rules:
#   1) New Question — Easy
#   2) New Question — Medium
#   3) New Question — Hard
#   4) Counter Question — Twist — Medium
#   5) Counter Question — Twist — Hard
#   6) Counter Question — Interrogatory — Medium
#   7) Counter Question — Interrogatory — Hard
# - No Easy counter questions are allowed anywhere.
# - Each QA block MUST include these fields:
#   - "block_id": unique like "B1", "B2", ...
#   - "guideline": one concise instruction for probing this block's focus
#   - "q_type": "New Question" or "Counter Question"
#   - "q_difficulty": It can be "Easy" | "Medium" | "Hard"
#   - "counter_type": ```If q_type == "Counter Question" then it required with its value being one from "Twist" | "Interrogatory"; otherwise omit or null```
#   - "qa_items": an array with EXACTLY ONE item:
#       * "qa_id": unique like "QA1"
#       * "example_questions": EXACTLY 5 concise, technical questions (no placeholders)
# - All questions must ground to the node's focus skills verbatim and use project/company IDs when provided (P1, P2, C, E…).
# - You can also use the mongo db database fetching tools again to fetch on data for keys like P1, P2,... (being present in the collection named cv), E1, E2,... (being present in the collection named cv), D (being present in the collection named summary with the key name domains_assess_D), S (being present in the entire collection named summary) and T (being present in the collection named summary with the key name annotated_skill_tree_T) with each relevant record having value of _id key as "{thread_id}" as per requirements.

# ---

# Output Format
# Return ONLY a JSON object grouped by topic with each topic having QA blocks with each having 5 questions per QA item:

# {{
#   "qa_sets": [
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Medium",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Hard",
#               "example_questions": [...]
#             }}
#             {{
#               "qa_id": "QA6",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }}
#           ]
#         }}
#       ]
#     }},
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Medium",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Hard",
#               "example_questions": [...]
#             }}
#             {{
#               "qa_id": "QA6",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }}
#           ]
#         }}
#       ]
#     }},
#     ...
#   ]
# }}
# '''
# # New format QAs 5
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question answer block generator for technical interviews.
# Your task is to generate example questions for each deep dive QA block across all 3 topics given from a discussion summary as input.

# ---
# You will be given three inputs: discussion summary, node for a deep dive question in each topic, and some guideline set for QA blocks.

# Inputs:
# Discussion Topic Summary:
# \n```{discussion_summary}```\n

# Deep-dive nodes for question generation:
# \n````{deep_dive_nodes}```\n

# Conditional schema related error as feedback for previous wrong generations if any:
# \n```{qa_error}```\n

# You shall use the mongo db database fetching tools to fetch on data for example question generation guidelines being present in the collection named question_guidelines with each type "Case study type questions", "Project based questions" and "Counter questions" being mentioned as _id key of each respective guideline record.
# ---

# HARD CONSTRAINTS (must need this exactly; fix if any schema related issues with the feedback provided through the qa_error):
# - For each QA block per topic, output EXACTLY 7 QA items covering these combinations (no more, no less, no duplicates):
#   • New Question    - Easy, Medium, Hard  (3 items)
#   • Counter Question - Twist - Medium, Hard        (2 items)  <no Easy counters>
#   • Counter Question - Interrogatory - Medium, Hard        (2 items)  <no Easy counters>
# - Recommended ordering and IDs (strict but you MAY reorder if needed): 
#   QA1..QA3 = New(E, M, H), QA4..QA5 = Counter(Twist + M, Twist + H), QA6..QA7 = Counter(Interrogatory + M, Interrogatory + H)
# - For each topic every QA item MUST include exactly 5 concise, technical example questions (no placeholders, no empty strings).
# - Use only skills/focus areas that appear verbatim in the union of all nodes' `skills` / `focus_area`.
# - If any previous attempts failed, you will receive `qa_error` for the schema below as a feedback if having any then you can use it to fix your generated schema while keeping the intent.

# QA Generation Rules
# - Per topic, generate EXACTLY 7 QA blocks (no more, no less), one block per combo, with each QA block following these rules:
#   1) New Question — Easy
#   2) New Question — Medium
#   3) New Question — Hard
#   4) Counter Question — Twist — Medium
#   5) Counter Question — Twist — Hard
#   6) Counter Question — Interrogatory — Medium
#   7) Counter Question — Interrogatory — Hard
# - No Easy counter questions are allowed anywhere.
# - Each QA block MUST include these fields:
#   - "block_id": unique like "B1", "B2", ...
#   - "guideline": one concise instruction for probing this block's focus
#   - "q_type": "New Question" or "Counter Question"
#   - "q_difficulty": It can be "Easy" | "Medium" | "Hard"
#   - "counter_type": ```If q_type == "Counter Question" then it required with its value being one from "Twist" | "Interrogatory"; otherwise omit or null```
#   - "qa_items": an array with EXACTLY ONE item:
#       * "qa_id": unique like "QA1"
#       * "example_questions": EXACTLY 5 concise, technical questions (no placeholders)
# - You can also use the mongo db database fetching tools again to fetch on data for keys like P1, P2,... (being present in the collection named cv), E1, E2,... (being present in the collection named cv), D (being present in the collection named summary with the key name domains_assess_D), S (being present in the entire collection named summary) and T (being present in the collection named summary with the key name annotated_skill_tree_T) with each relevant record having value of _id key as "{thread_id}" as per requirements.
# - All questions must ground to the node's focus skills verbatim and use project, experience, summary etc keys when provided (P1, P2, E1, T). But make sure to not make questions from the projects where the respective project's projectwise summary has no such evidence mentioned in any its field.

# ---

# Output Format
# Return ONLY a JSON object grouped by topic with each topic having QA blocks with each having 5 questions per QA item:

# {{
#   "qa_sets": [
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Medium",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Hard",
#               "example_questions": [...]
#             }}
#             {{
#               "qa_id": "QA6",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }}
#           ]
#         }}
#       ]
#     }},
#     ...
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Medium",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Hard",
#               "example_questions": [...]
#             }}
#             {{
#               "qa_id": "QA6",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }}
#           ]
#         }}
#       ]
#     }}
#   ]
# }}
# '''
# # New format(without first question) QAs 6
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question answer block generator for technical interviews.
# Your task is to generate example questions for each deep dive QA block across all 3 topics given from a discussion summary as input.

# ---
# You will be given three inputs: discussion summary, node for a deep dive question in each topic, and some guideline set for QA blocks.

# Inputs:
# Discussion Topic Summary:
# \n```{discussion_summary}```\n

# Deep-dive nodes for question generation:
# \n````{deep_dive_nodes}```\n

# Conditional schema related error as feedback for previous wrong generations if any:
# \n```{qa_error}```\n

# You shall use the mongo db database fetching tools to fetch on data for example question generation guidelines being present in the collection named question_guidelines with each type "Case study type questions", "Project based questions" and "Counter questions" being mentioned as _id key of each respective guideline record.
# ---

# HARD CONSTRAINTS (must need this exactly; fix if any schema related issues with the feedback provided through the qa_error):
# - For each QA block per topic, output EXACTLY 7 QA items covering these combinations (no more, no less, no duplicates):
#   • New Question    - Easy, Medium, Hard  (3 items)
#   • Counter Question - Twist - Medium, Hard        (2 items)  <no Easy counters>
#   • Counter Question - Interrogatory - Medium, Hard        (2 items)  <no Easy counters>
# - Recommended ordering and IDs (strict but you MAY reorder if needed): 
#   QA1..QA3 = New(E, M, H), QA4..QA5 = Counter(Twist + M, Twist + H), QA6..QA7 = Counter(Interrogatory + M, Interrogatory + H)
# - For each topic every QA item MUST include exactly 5 concise, technical example questions (no placeholders, no empty strings).
# - For any QA try to use combination of as much skills as much as possible in making your example questions but they should still make sense. So make sure to do the following things:
#   1. Use all skills/focus areas only that appear in the combination of all node's `skills` / `focus_area` lists of the Deep dive Nodes and make sure none of them is left out,
#   2. Ensure none of the skills present in the `focus_areas` of the Deep Dive type sequence of each respective topic of the discussion summary is not left out and is used completely
#  also 
# - If any previous attempts failed, you will receive `qa_error` for the schema below as a feedback if having any then you can use it to fix your generated schema while keeping the intent.

# QA Generation Rules
# - Per topic, generate EXACTLY 7 QA blocks (no more, no less), one block per combo, with each QA block following these rules:
#   1) New Question — Easy
#   2) New Question — Medium
#   3) New Question — Hard
#   4) Counter Question — Twist — Medium
#   5) Counter Question — Twist — Hard
#   6) Counter Question — Interrogatory — Medium
#   7) Counter Question — Interrogatory — Hard
# - No Easy counter questions are allowed anywhere.
# - Each QA block MUST include these fields:
#   - "block_id": unique like "B1", "B2", ...
#   - "guideline": one concise instruction for probing this block's focus
#   - "q_type": "New Question" or "Counter Question"
#   - "q_difficulty": It can be "Easy" | "Medium" | "Hard"
#   - "counter_type": ```If q_type == "Counter Question" then it required with its value being one from "Twist" | "Interrogatory"; otherwise omit or null```
#   - "qa_items": an array with EXACTLY ONE item:
#       * "qa_id": unique like "QA1"
#       * "example_questions": EXACTLY 5 concise, technical questions (no placeholders)
# - You can also use the mongo db database fetching tools again to fetch on data for keys like P1, P2,... (being present in the collection named cv), E1, E2,... (being present in the collection named cv), D (being present in the collection named summary with the key name domains_assess_D), S (being present in the entire collection named summary) and T (being present in the collection named summary with the key name annotated_skill_tree_T) with each relevant record having value of _id key as "{thread_id}" as per requirements.
# - All questions must ground to the node's focus skills verbatim and use project, experience, summary etc keys when provided (P1, P2, E1, T). But make sure to not make questions from the projects where the respective project's projectwise summary has no such evidence mentioned in any its field.

# ---

# Output Format
# Return ONLY a JSON object grouped by topic with each topic having QA blocks with each having 5 questions per QA item:

# {{
#   "qa_sets": [
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Medium",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Hard",
#               "example_questions": [...]
#             }}
#             {{
#               "qa_id": "QA6",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }}
#           ]
#         }}
#       ]
#     }},
#     ...
#     {{
#       "topic": "short name",
#       "qa_blocks": [
#         {{
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
#           "qa_items": [
#             {{
#               "qa_id": "QA1",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA2",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA3",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "counter_type": null,
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA4",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Medium",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA5",
#               "q_type": "Counter Question",
#               "counter_type": "Twist",
#               "q_difficulty": "Hard",
#               "example_questions": [...]
#             }}
#             {{
#               "qa_id": "QA6",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }},
#             {{
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": [...]
#             }}
#           ]
#         }}
#       ]
#     }}
#   ]
# }}
# '''

# # New format 7 running with open ai
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question-answer (QA) block generator for technical interviews.
# Your task is to generate example questions for each deep-dive QA block across all topics, given a discussion summary and the topic's deep-dive nodes as input.

# ---
# Inputs:
# Discussion Topic Summary:
# ```@discussion_summary```

# Deep-dive nodes for question generation (ordered for this topic):
# ```@deep_dive_nodes```

# Conditional schema-related error as feedback for previous wrong generations (fix issues while keeping intent):
# ```@qa_error```

# You have access to MongoDB fetching tools (programmatic tools, not shell). Use them to retrieve guideline and context data when needed.

# MONGODB USAGE (STRICT):
# - Use only these tools: mongodb_list_collections, mongodb_query_checker, custom_mongodb_query.
# - NEVER call custom_mongodb_query without "query".
# - For 'cv' and 'summary', ALWAYS use {"_id": "{thread_id}"}.
# - Do not call mongodb_list_collections or mongodb_schema.
# - Validate with mongodb_query_checker BEFORE executing.
# Valid:
#   custom_mongodb_query args={"collection":"summary","query":{"_id":"{thread_id}"}}
#   custom_mongodb_query args={"collection":"cv","query":{"_id":"{thread_id}"}}
#   custom_mongodb_query args={"collection":"question_guidelines",
#     "query":{"_id":{"$in":["Case study type questions","Project based questions","Counter questions"]}}}
# Invalid (do not do this): custom_mongodb_query args={"collection":"summary"}
# When grounding to projects/experiences, first check the summary's projectwise_summary; skip items where fields say "no such evidence".

# ---
# HARD CONSTRAINTS (must be exact; use @qa_error feedback to correct on retries):
# - Let N = the number of deep-dive nodes in @deep_dive_nodes. For this topic, output EXACTLY N QA blocks, one block per deep-dive node, IN THE SAME ORDER as provided.
# - Each QA block MUST contain EXACTLY SEVEN qa_items with these combos (no more, no less, no duplicates):
#   • New Question — Easy
#   • New Question — Medium
#   • New Question — Hard
#   • Counter Question — Twist — Medium
#   • Counter Question — Twist — Hard
#   • Counter Question — Interrogatory — Medium
#   • Counter Question — Interrogatory — Hard
#   (No Easy counter questions anywhere.)
# - Each qa_item MUST include EXACTLY 5 concise, technical, fully-specified example questions (no placeholders, no empty strings, plain text only—no markdown).
# - Skill grounding:
#   1) Use only skills that appear in the union of all deep-dive node skills/focus areas for THIS topic.
#   2) Ensure every skill listed in the discussion summary's deep-dive focus_areas for THIS topic appears across the 7 qa_items of the block set (none left out across the topic).
#   3) Prefer aligning each QA block's questions to the skills/context of its corresponding deep-dive node.
# - Project/experience grounding:
#   - Use P1, P2, …; E1, E2, …; T; D; S only when they exist for @thread_id.
#   - Do NOT create questions from projects whose projectwise_summary fields state “no such evidence” for the relevant aspect.
#   - Outside of questions and guidelines, do NOT output these keys arbitrarily.

# QA Generation Rules
# - For THIS topic, generate EXACTLY N QA blocks (N = number of deep-dive nodes).
# - Each block contains:
#   - "block_id": unique like "B1", "B2", ..., in the same order as deep-dive nodes
#   - "guideline": one concise instruction to probe this block's focus (reflect the mapped deep-dive node)
#   - "qa_items": an array with EXACTLY SEVEN items; for each item:
#       * "qa_id": unique within the block like "QA1", "QA2", ...
#       * "q_type": "New Question" | "Counter Question"
#       * "q_difficulty": "Easy" | "Medium" | "Hard"
#       * "counter_type": present ONLY if q_type == "Counter Question" and ∈ {"Twist","Interrogatory"}; otherwise null/omitted
#       * "example_questions": EXACTLY 5 concise, technical questions (plain text)

# Output Format
# Return ONLY a JSON object grouped by topic, with each topic having QA blocks and each QA item having 5 questions:

# {
#   "qa_sets": [
#     {
#       "topic": "short name",
#       "qa_blocks": [
#         {
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this block's focus skills and measurable outcomes.",
#           "qa_items": [
#             {"qa_id": "QA1","q_type": "New Question","q_difficulty": "Easy","counter_type": null,"example_questions": [...]},
#             {"qa_id": "QA2","q_type": "New Question","q_difficulty": "Medium","counter_type": null,"example_questions": [...]},
#             {"qa_id": "QA3","q_type": "New Question","q_difficulty": "Hard","counter_type": null,"example_questions": [...]},
#             {"qa_id": "QA4","q_type": "Counter Question","q_difficulty": "Medium","counter_type": "Twist","example_questions": [...]},
#             {"qa_id": "QA5","q_type": "Counter Question","q_difficulty": "Hard","counter_type": "Twist","example_questions": [...]},
#             {"qa_id": "QA6","q_type": "Counter Question","q_difficulty": "Medium","counter_type": "Interrogatory","example_questions": [...]},
#             {"qa_id": "QA7","q_type": "Counter Question","q_difficulty": "Hard","counter_type": "Interrogatory","example_questions": [...]}
#           ]
#         },
#         ...
#         {
#           "block_id": "Bn",
#           "guideline": "One sentence on how to probe this block's focus skills and measurable outcomes.",
#           "qa_items": [
#             {"qa_id": "QA1","q_type": "New Question","q_difficulty": "Easy","counter_type": null,"example_questions": [...]},
#             {"qa_id": "QA2","q_type": "New Question","q_difficulty": "Medium","counter_type": null,"example_questions": [...]},
#             {"qa_id": "QA3","q_type": "New Question","q_difficulty": "Hard","counter_type": null,"example_questions": [...]},
#             {"qa_id": "QA4","q_type": "Counter Question","q_difficulty": "Medium","counter_type": "Twist","example_questions": [...]},
#             {"qa_id": "QA5","q_type": "Counter Question","q_difficulty": "Hard","counter_type": "Twist","example_questions": [...]},
#             {"qa_id": "QA6","q_type": "Counter Question","q_difficulty": "Medium","counter_type": "Interrogatory","example_questions": [...]},
#             {"qa_id": "QA7","q_type": "Counter Question","q_difficulty": "Hard","counter_type": "Interrogatory","example_questions": [...]}
#           ]
#         }
#       ]
#     }
#   ]
# }
# '''
# New format 7 try with gemini
QA_BLOCK_AGENT_PROMPT = '''
You are a question-answer (QA) block generator for technical interviews.
Your task is to generate example questions for each deep-dive QA block across all topics, given a discussion summary and the topic's deep-dive nodes as input.

---
Inputs:
Discussion Topic Summary:
```@discussion_summary```

Deep-dive nodes for question generation (ordered for this topic):
```@deep_dive_nodes```

Conditional schema-related error as feedback for previous wrong generations (fix issues while keeping intent):
```@qa_error```

You have access to MongoDB fetching tools (programmatic tools, not shell). Use them to retrieve guideline and context data when needed.

MONGODB USAGE (STRICT):
- Use only these tools: mongodb_list_collections, mongodb_query_checker, custom_mongodb_query.
- NEVER call custom_mongodb_query without "query".
- For 'cv' and 'summary', ALWAYS use {"_id": "@thread_id"}.
- Do not call mongodb_list_collections or mongodb_schema.
- Validate with mongodb_query_checker BEFORE executing.
Valid:
{
  "name": "custom_mongodb_query",
  "args": {
    "collection": "summary",
    "query": {
      "_id": "@thread_id"
    }
  }
}
  {
  "name": "custom_mongodb_query",
  "args": {
    "collection": "summary",
    "query": {
      "_id": "@thread_id"
    }
  }
}

  {
  "name": "custom_mongodb_query",
  "args": {
    "collection": "question_guidelines",
    "query": {
      "_id": {
        "$in": [
          "Case study type questions",
          "Project based questions",
          "Counter questions"
        ]
      }
    }
  }
}
Invalid (do not do this): custom_mongodb_query args={"collection":"summary"}
When grounding to projects/experiences, first check the summary's projectwise_summary; skip items where fields say "no such evidence".

---
HARD CONSTRAINTS (must be exact; use @qa_error feedback to correct on retries):
- Let N = the number of deep-dive nodes in @deep_dive_nodes. For this topic, output EXACTLY N QA blocks, one block per deep-dive node, IN THE SAME ORDER as provided.
- Each QA block MUST contain EXACTLY SEVEN qa_items with these combos (no more, no less, no duplicates):
  • New Question — Easy
  • New Question — Medium
  • New Question — Hard
  • Counter Question — Twist — Medium
  • Counter Question — Twist — Hard
  • Counter Question — Interrogatory — Medium
  • Counter Question — Interrogatory — Hard
  (No Easy counter questions anywhere.)
- Each qa_item MUST include EXACTLY 5 concise, technical, fully-specified example questions (no placeholders, no empty strings, plain text only—no markdown).
- Skill grounding:
  1) Use only skills that appear in the union of all deep-dive node skills/focus areas for THIS topic.
  2) Ensure every skill listed in the discussion summary's deep-dive focus_areas for THIS topic appears across the 7 qa_items of the block set (none left out across the topic).
  3) Prefer aligning each QA block's questions to the skills/context of its corresponding deep-dive node.
- Project/experience grounding:
  - Use P1, P2, …; E1, E2, …; T; D; S only when they exist for @thread_id.
  - Do NOT create questions from projects whose projectwise_summary fields state “no such evidence” for the relevant aspect.
  - Outside of questions and guidelines, do NOT output these keys arbitrarily.

QA Generation Rules
- For THIS topic, generate EXACTLY N QA blocks (N = number of deep-dive nodes).
- Each block contains:
  - "block_id": unique like "B1", "B2", ..., in the same order as deep-dive nodes
  - "guideline": one concise instruction to probe this block's focus (reflect the mapped deep-dive node)
  - "qa_items": an array with EXACTLY SEVEN items; for each item:
      * "qa_id": unique within the block like "QA1", "QA2", ...
      * "q_type": "New Question" | "Counter Question"
      * "q_difficulty": "Easy" | "Medium" | "Hard"
      * "counter_type": present ONLY if q_type == "Counter Question" and ∈ {"Twist","Interrogatory"}; otherwise null/omitted
      * "example_questions": EXACTLY 5 concise, technical questions (plain text)

Output Format
Return ONLY a JSON object grouped by topic, with each topic having QA blocks and each QA item having 5 questions:

{
  "qa_sets": [
    {
      "topic": "short name",
      "qa_blocks": [
        {
          "block_id": "B1",
          "guideline": "One sentence on how to probe this block's focus skills and measurable outcomes.",
          "qa_items": [
            {"qa_id": "QA1","q_type": "New Question","q_difficulty": "Easy","counter_type": null,"example_questions": [...]},
            {"qa_id": "QA2","q_type": "New Question","q_difficulty": "Medium","counter_type": null,"example_questions": [...]},
            {"qa_id": "QA3","q_type": "New Question","q_difficulty": "Hard","counter_type": null,"example_questions": [...]},
            {"qa_id": "QA4","q_type": "Counter Question","q_difficulty": "Medium","counter_type": "Twist","example_questions": [...]},
            {"qa_id": "QA5","q_type": "Counter Question","q_difficulty": "Hard","counter_type": "Twist","example_questions": [...]},
            {"qa_id": "QA6","q_type": "Counter Question","q_difficulty": "Medium","counter_type": "Interrogatory","example_questions": [...]},
            {"qa_id": "QA7","q_type": "Counter Question","q_difficulty": "Hard","counter_type": "Interrogatory","example_questions": [...]}
          ]
        },
        ...
        {
          "block_id": "Bn",
          "guideline": "One sentence on how to probe this block's focus skills and measurable outcomes.",
          "qa_items": [
            {"qa_id": "QA1","q_type": "New Question","q_difficulty": "Easy","counter_type": null,"example_questions": [...]},
            {"qa_id": "QA2","q_type": "New Question","q_difficulty": "Medium","counter_type": null,"example_questions": [...]},
            {"qa_id": "QA3","q_type": "New Question","q_difficulty": "Hard","counter_type": null,"example_questions": [...]},
            {"qa_id": "QA4","q_type": "Counter Question","q_difficulty": "Medium","counter_type": "Twist","example_questions": [...]},
            {"qa_id": "QA5","q_type": "Counter Question","q_difficulty": "Hard","counter_type": "Twist","example_questions": [...]},
            {"qa_id": "QA6","q_type": "Counter Question","q_difficulty": "Medium","counter_type": "Interrogatory","example_questions": [...]},
            {"qa_id": "QA7","q_type": "Counter Question","q_difficulty": "Hard","counter_type": "Interrogatory","example_questions": [...]}
          ]
        }
      ]
    }
  ]
}
'''

# # test new format 8 with first question
# QA_BLOCK_AGENT_PROMPT = '''
# You are a question-answer (QA) block generator for technical interviews.
# Your task is to generate example questions for each deep-dive QA block across all topics, given a discussion summary and the topic's deep-dive nodes as input.

# ---
# Inputs:
# Discussion Topic Summary:
# ```@discussion_summary```

# Deep-dive nodes for question generation (ordered for this topic):
# ```@deep_dive_nodes```

# Conditional schema-related error as feedback for previous wrong generations (fix issues while keeping intent):
# ```@qa_error```

# You have access to MongoDB fetching tools (programmatic tools, not shell). Use them to retrieve guideline and context data when needed.

# MONGODB USAGE (STRICT):
# - Use only these tools: mongodb_list_collections, mongodb_query_checker, custom_mongodb_query.
# - NEVER call custom_mongodb_query without "query".
# - For 'cv' and 'summary', ALWAYS use {"_id":"{thread_id}"}.
# - Do not call mongodb_list_collections or mongodb_schema.
# - Validate with mongodb_query_checker BEFORE executing.
# Valid:
#   custom_mongodb_query args={"collection":"summary","query":{"_id":"{thread_id}"}}
#   custom_mongodb_query args={"collection":"cv","query":{"_id":"{thread_id}"}}
#   custom_mongodb_query args={"collection":"question_guidelines","query":{"_id":{"$in":["Case study type questions","Project based questions","Counter questions"]}}}
# Invalid (do not do this): custom_mongodb_query args={"collection":"summary"}
# When grounding to projects/experiences, first check the summary's projectwise_summary; skip items where fields say "no such evidence".

# ---
# HARD CONSTRAINTS (must be exact; use @qa_error feedback to correct on retries):
# - Let N = the number of deep-dive nodes in @deep_dive_nodes. For this topic, output EXACTLY N QA blocks, one block per deep-dive node, IN THE SAME ORDER as provided.
# - Each QA block MUST contain EXACTLY TEN qa_items with these combos (no more, no less, no duplicates):
#   • First Question       → Easy, Medium, Hard  (3 items)
#   • New Question         → Easy, Medium, Hard  (3 items)
#   • Counter Question — Twist         → Medium, Hard  (2 items)  <no Easy counters>
#   • Counter Question — Interrogatory → Medium, Hard  (2 items)  <no Easy counters>
# - Recommended IDs and order (you MAY reorder if necessary but still produce 10 items):
#   QA1..QA3 = First(E, M, H),
#   QA4..QA6 = New(E, M, H),
#   QA7..QA8 = Counter(Twist M, Twist H),
#   QA9..QA10 = Counter(Interrogatory M, Interrogatory H).
# - Every qa_item MUST include EXACTLY 5 concise, technical, fully-specified example questions (no placeholders, no empty strings, plain text only—no markdown).
# - Field requirements per qa_item:
#   • "q_type": one of {"First Question","New Question","Counter Question"}
#   • "q_difficulty": one of {"Easy","Medium","Hard"}
#   • "counter_type": REQUIRED IFF q_type == "Counter Question" and must be one of {"Twist","Interrogatory"}; otherwise omit or null.
# - Skill grounding:
#   1) Use only skills that appear in the union of all deep-dive node skills/focus_area lists for THIS topic.
#   2) Ensure every skill listed in the discussion summary's deep-dive focus_areas for THIS topic appears across the qa_items for the topic (none left out).
#   3) Prefer aligning each QA block's questions to the skills/context of its corresponding deep-dive node.
# - Project/experience grounding:
#   - Use P1, P2, …; E1, E2, …; T; D; S only when they exist for @thread_id.
#   - Do NOT create questions from projects whose projectwise_summary fields state “no such evidence” for the relevant aspect.
#   - Outside of questions and guidelines, do NOT output these keys arbitrarily.

# QA Generation Rules
# - For THIS topic, generate EXACTLY N QA blocks (N = number of deep-dive nodes).
# - Each block contains:
#   - "block_id": unique like "B1", "B2", ..., in the same order as deep-dive nodes.
#   - "guideline": one concise instruction to probe this block's focus (reflect the mapped deep-dive node).
#   - "qa_items": an array with EXACTLY TEN items; for each item:
#       * "qa_id": unique within the block like "QA1", "QA2", ...
#       * "q_type", "q_difficulty", "counter_type" (per constraints above)
#       * "example_questions": EXACTLY 5 concise, technical questions.

# Output Format
# Return ONLY a JSON object grouped by topic, with each topic having QA blocks and each QA item having 5 questions:

# {
#   "qa_sets": [
#     {
#       "topic": "short name",
#       "qa_blocks": [
#         {
#           "block_id": "B1",
#           "guideline": "One sentence on how to probe this block's focus skills and measurable outcomes.",
#           "qa_items": [
#             {
#               "qa_id": "QA1",
#               "q_type": "First Question",
#               "q_difficulty": "Easy",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA2",
#               "q_type": "First Question",
#               "q_difficulty": "Medium",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA3",
#               "q_type": "First Question",
#               "q_difficulty": "Hard",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA4",
#               "q_type": "New Question",
#               "q_difficulty": "Easy",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA5",
#               "q_type": "New Question",
#               "q_difficulty": "Medium",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA6",
#               "q_type": "New Question",
#               "q_difficulty": "Hard",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA7",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Twist",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA8",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Twist",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA9",
#               "q_type": "Counter Question",
#               "q_difficulty": "Medium",
#               "counter_type": "Interrogatory",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             },
#             {
#               "qa_id": "QA10",
#               "q_type": "Counter Question",
#               "q_difficulty": "Hard",
#               "counter_type": "Interrogatory",
#               "example_questions": ["...", "...", "...", "...", "..."]
#             }
#           ]
#         },
#         ...
#       ]
#     }
#   ]
# }
# '''
