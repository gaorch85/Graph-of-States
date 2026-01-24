import pandas as pd
from utils.public_function import llm_generate_response

# ===================== configs =====================
data_path = "Correct-Path-Here"
group_size = 5 # Evaluate every 5 cases
model_path = "gpt-5.1"
temperature = 1
max_tokens = 8192

# ===================== 1. load data =====================
df = pd.read_csv(data_path, encoding="utf-8")

df["prediction"] = df["prediction"].fillna("").str.strip()
df["report"] = df["report"].fillna("").str.strip()
df["answer"] = df["answer"].fillna("").str.strip()
combined_list = (df["prediction"] + ", " + df["report"]).tolist()
answer_list = df["answer"].tolist()

# ===================== 2. Define system prompt =====================
system_prompt = """
You are an expert in diagnosing challenging cases. You will receive a GROUP of student’s answers (each containing final diagnosis: prediction + report) and corresponding reference diagnoses. 
Score EACH diagnosis in the group according to these rules: 
2 = Student’s diagnosis exactly matches the reference diagnosis; 
1 = Student’s diagnosis is a broad category that includes the reference diagnosis; 
0 = Student’s diagnosis does not meet the criteria for 1 or 2. 

Output Format (clearly mark group number and each case score):
Group X:
1. Case 1 - Disease name: score X;
2. Case 2 - Disease name: score X;
...
(If less than 5 cases in the group, list only existing cases)
"""

# ===================== 3. Evaluate =====================
total_cases = len(combined_list)
for group_idx in range(0, total_cases, group_size):
    start = group_idx
    end = min(group_idx + group_size, total_cases)
    current_combined = combined_list[start:end]
    current_answers = answer_list[start:end]
    
    user_prompt = f"""
Group {group_idx // group_size + 1} (Cases {start+1} to {end}):
Here is the student’s answers for this group:
"""
    for i, (stu_ans, ref_ans) in enumerate(zip(current_combined, current_answers), 1):
        user_prompt += f"{i}. Case {start+i}: {stu_ans}\n"
    
    user_prompt += f"""
Here is the reference diagnoses for this group:
"""
    for i, ref_ans in enumerate(current_answers, 1):
        user_prompt += f"{i}. Case {start+i}: {ref_ans}\n"

    try:
        response = llm_generate_response(
            user_prompt=user_prompt,
            model_path=model_path,
            temperature=temperature,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )
        print(f"\n========== Group {group_idx // group_size + 1} (Cases {start+1}-{end}) Score Result ==========")
        print(response)
    except Exception as e:
        print(f"\n❌ Error processing Group {group_idx // group_size + 1}: {str(e)}")
        continue

print("\n✅ All groups processed completed!")