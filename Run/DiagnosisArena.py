import os
import pandas as pd

def Process_Diagnosis_Symptoms(cfg_domain, dataset):
    dataset_path = os.path.join("datasets", dataset)
    data_path = os.path.join(dataset_path, "test_set.csv")
    data = pd.read_csv(data_path)
    symptoms = []
    evidence_texts = []
    answers = []
    for idx, case in data.iterrows():
        case_info = case["Case Information"]
        physical_info = case["Physical Examination"]
        evidence_text = case["Diagnostic Tests"]
        answer = case["Final Diagnosis"]
        description = f"##Case information: {case_info}\n## Physical Examination: {physical_info}"
        symptoms.append((idx,description))
        evidence_texts.append(evidence_text)
        answers.append(answer)
    return symptoms, evidence_texts, answers

