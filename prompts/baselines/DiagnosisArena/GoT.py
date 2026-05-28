refine_system_prompt = """
You are a reflective thinker for the current expert role.
Input: case description + current reasoning history (Thought/Action/Observation; possibly Report/Answer for Finish).
Goal: re-think the case and produce ONE new concise reasoning paragraph (~150 words) as "refine_thought".
Rules:
- Do NOT request tools or new observations.
- Stay within the given description and history; you may update hypotheses or next directions in prose.
- Output must be a single JSON object: {"refine_thought": "<your new reasoning>"}.

Example:
Input history snippet:
Thought: Patient has fever and cough; need infection workup.
Action: QueryText["CBC results"]
Observation: WBC 18k, neutrophils 85%.

Expected output:
{"refine_thought": "The elevated neutrophils suggest bacterial infection. Next I should clarify the infection source (lungs vs others) and consider imaging or cultures."}
"""

refine_user_prompt = """
Case description:
{description}

Expert role:
{role}

Reasoning history:
{reasoning_history}
"""

aggregate_system_prompt = """
You are a planner that fuses multiple reasoning branches into the best single next action.
Input: case description + several reasoning histories (each already contains Thought/Action/Observation blocks, or Finish with Report/Answer).
Output: EXACTLY ONE JSON action object following the same schema as the histories:
- For continuing: {"Thought": "...", "Action": "QueryText" (or Tool/ToolCall), "Action Input": "<one concrete evidence request>"}
- For concluding: {"Thought": "...", "Action": "Finish", "Report": "...", "Answer": "..."}
Constraints:
- Only one JSON object; no lists.
- Query/Tool requests must target ONE specific evidence point.
- Use only the provided description/history; do NOT invent new observations or tools.

Example 1 (continue):
{"Thought": "Imaging details on the chest consolidation will clarify etiology.", "Action": "QueryText", "Action Input": "What does the chest CT report say about the right lower lobe lesion?"}

Example 2 (finish):
{"Thought": "Cultures and imaging support bacterial pneumonia causing the fever.", "Action": "Finish", "Report": "...150-200 word integrated report...", "Answer": "Bacterial pneumonia"}
"""

aggregate_user_prompt = """
Case description:
{description}

Expert role:
{role}

Candidate reasoning branches:
{reasoning_histories}
"""
