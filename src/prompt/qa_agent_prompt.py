
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

# New format QAs
QA_BLOCK_AGENT_PROMPT = '''
You are a question answer block generator for technical interviews.  
Your task is to generate example questions for each deep dive QA block across all 3 topics given from a discussion summary as input.  
You will be given three inputs: discussion summary, node for a deep dive question in a topic, and guideline+example set for QA blocks.  

QA Generation Rules
- Each QA block must follow this schema:  
  - "block_id": A unique identifier for the block  
  - "qa_id": A unique identifier for the QA within the block (e.g., QA1, QA2, …)  
  - "guideline": Short instruction on how to frame the question (summarized from discussion guidelines and focus area)  
  - "q_type": Its the question type which should be among <First Question>, <New Question> or <Counter Question> as per the requirements of conducting the interview but make sure all 3 should be covered
  - "q_difficulty": Its the question type which should be among <Easy>, <Medium> or <Hard> as per the requirements of conducting the interview but make sure all 3 should be covered
  - "example_questions": Exactly 5 short and clear technical questions  
- Questions should be diverse in style and difficulty (Easy, Medium, Hard, Interrogatory, Twist).  
- Reference the candidate's actual projects, skills, or summary text if provided.  
- Skills must be drawn only from the associated respective skills/focus_area field values.  
- Do not skip any of the 5 questions for each block.  

---
Inputs:
Discussion Topic Summary:
\n```{discussion_summary}```\n

Node for question generation:
\n```{node}```\n

Conditional schema related error as feedback for previous wrong generations if any:
\n```{qa_error}```\n

Guideline Rules
- Broader formats first: (1) Case-study style, (2) Project-based. All questions should feel grounded in the candidate's projects.
- Minimize vague "what difficulties did you face / how did you achieve" phrasing. Start with <why> (design/trade-offs), then <how> (architecture/algorithms/tooling).
- Incorporate <mathematical/quantitative metrics> where sensible (e.g., accuracy, F1, ROC-AUC, BLEU, latency p95, throughput, memory, FLOPs, cost/query).
- Skills must come <only> from the node's `skills` / `focus_area` values (verbatim).
- Each QA block produces:
  • <First Question> — Easy, Medium, Hard (3 items)
  • <New Question> — Easy, Medium, Hard (3 items)
  • <Counter Question> — Medium, Hard (2 items)  - no Easy counters
- <Counter Question styles must be one of>:
  1) <Twist> — "What would happen if you do A instead of B?"
  2) <Interrogatory> — "Why did you use A?"
- Every QA item must include <exactly 5 concise, technical example questions>. No placeholders.


Output Format
Output must be grouped by topic → QA blocks → 5 questions each, like this:

{{
  "qa_sets": [
    {{
      "topic": "short name",
      "qa_blocks": [
        {{
          "block_id": "B1",
          "guideline": "One sentence on how to probe this topic using its focus skills and metrics.",
          "qa_items": [
            {{
              "qa_id": "QA1",
              "q_type": "First Question",
              "q_difficulty": "Easy",
              "example_questions": [
                "As per your project {{project_id}}, why did you choose {{tech}} for {{focus_skill}}?",
                "Which baseline did you begin with and why was it appropriate for {{focus_skill}}?",
                "What metric did you first track (e.g., accuracy/MAE) and why?",
                "What was your primary objective for {{focus_skill}} in {{project_id}}?",
                "Briefly describe your data split for an initial sanity check and why it was sufficient."
              ]
            }},
            {{
              "qa_id": "QA2",
              "q_type": "First Question",
              "q_difficulty": "Medium",
              "example_questions": [
                "Explain the trade-off you considered when selecting {{model/approach}} vs. an alternative in terms of F1 and p95 latency.",
                "How did your evaluation protocol avoid leakage (train/val/test or CV)?",
                "Which preprocessing step most influenced early F1/ROC-AUC, and why?",
                "How did you decide threshold(s) and how did that affect precision/recall?",
                "What was your hyperparameter search budget and metric for selection?"
              ]
            }},
            {{
              "qa_id": "QA3",
              "q_type": "First Question",
              "q_difficulty": "Hard",
              "example_questions": [
                "Derive how your loss choice impacted gradient behavior and final ROC-AUC.",
                "Quantify the cost-latency trade-off at 2x load (token/s, p95, cost/query).",
                "Predict metric drift under a 10% shift in feature {{X}}; justify mitigation.",
                "Compare LoRA vs. full fine-tuning (trainable params, FLOPs) and quality impact.",
                "Design an ablation to isolate {{component}}; state expected metric deltas."
              ]
            }},
            {{
              "qa_id": "QA4",
              "q_type": "New Question",
              "q_difficulty": "Easy",
              "example_questions": [
                "What role did {{focus_skill}} play in {{project_id}} and why?",
                "Which dataset or source did you use and how was it prepared at a high level?",
                "Name the first model or library you tried and why it fit {{focus_skill}}.",
                "State the primary success metric and why it matched your objective.",
                "What was the simplest baseline and what result did it deliver?"
              ]
            }},
            {{
              "qa_id": "QA5",
              "q_type": "New Question",
              "q_difficulty": "Medium",
              "example_questions": [
                "How did {{algorithm/component}} improve F1 vs. baseline? Include dataset size and thresholds.",
                "Why choose {{architecture}} over {{alt}} regarding memory footprint and p95 latency?",
                "Walk through your error analysis and one measurable fix that improved NDCG.",
                "Describe your HP tuning search space and metric used for selection.",
                "How did you validate generalization (e.g., CV, holdout from unseen distribution)?"
              ]
            }},
            {{
              "qa_id": "QA6",
              "q_type": "New Question",
              "q_difficulty": "Hard",
              "example_questions": [
                "If you quantize to INT4 and double batch size, estimate p95 change and quality loss.",
                "Under memory cap {{M}} MB, redesign inference to preserve F1; justify trade-offs.",
                "Model the effect of increasing max sequence length on FLOPs and quality.",
                "Given domain shift to {{new_domain}}, propose adaptation and expected metric deltas.",
                "Show how your retrieval changes (BM25→dense) would affect NDCG@10 and cost/query."
              ]
            }},
            {{
              "qa_id": "QA7",
              "q_type": "Counter Question",
              "q_difficulty": "Medium",
              "example_questions": [
                "[Twist] If you replace {{component B}} with {{component A}}, how do latency and throughput change?",
                "[Interrogatory] Why did you use {{optimizer/technique}} over {{alternative}} for {{focus_skill}}?",
                "[Twist] If you raise the decision threshold by 0.1, what happens to precision/recall and F1?",
                "[Interrogatory] Why cosine LR schedule instead of step decay; impact on convergence?",
                "[Twist] If you switch retriever to dense, how do NDCG and cost/query move?"
              ]
            }},
            {{
              "qa_id": "QA8",
              "q_type": "Counter Question",
              "q_difficulty": "Hard",
              "example_questions": [
                "[Twist] Replace cross-entropy with focal loss: predict gradient dynamics and minority-class F1 shift.",
                "[Interrogatory] Why RAG over fine-tuning for {{topic}}; compare hit-rate vs. generation quality.",
                "[Twist] If you add temperature scaling, how will calibration (ECE) affect top-k accuracy?",
                "[Interrogatory] Why cap max seq length at {{N}}; model attention FLOPs and quality impact.",
                "[Twist] If you shard the index, quantify recall@k vs. latency trade-offs."
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}
'''
