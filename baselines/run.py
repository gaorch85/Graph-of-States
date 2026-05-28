from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .method.multi_agent_CoT import MultiAgentCoT
from .method.multi_agent_FoT import MultiAgentFoT
from .method.multi_agent_GoT import MultiAgentGoT
from .method.multi_agent_ToT import MultiAgentToT
from .method.single_agent_CoT import SingleAgentCoT
from .method.single_agent_FoT import SingleAgentFoT
from .method.single_agent_GoT import SingleAgentGoT
from .method.single_agent_ToT import SingleAgentToT


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-5.1"

METHODS = {
    "SingleAgentCoT": SingleAgentCoT,
    "SingleAgentToT": SingleAgentToT,
    "SingleAgentFoT": SingleAgentFoT,
    "SingleAgentGoT": SingleAgentGoT,
    "MultiAgentCoT": MultiAgentCoT,
    "MultiAgentToT": MultiAgentToT,
    "MultiAgentFoT": MultiAgentFoT,
    "MultiAgentGoT": MultiAgentGoT,
}


def load_diagnosis_data(split: str = "test_set") -> List[Dict[str, str]]:
    csv_name = split if split.endswith(".csv") else f"{split}.csv"
    data_path = BASE_DIR / "datasets" / "DiagnosisArena" / csv_name
    if not data_path.exists():
        raise FileNotFoundError(f"DiagnosisArena data not found at {data_path}")

    data = pd.read_csv(data_path)
    cases = []
    for _, row in data.iterrows():
        cases.append(
            {
                "id": row["id"],
                "description": (
                    f"##Case Information: {row['Case Information']}\n\n"
                    f"## Physical Examination: {row['Physical Examination']}"
                ),
                "evidence": row["Diagnostic Tests"],
                "answer": row["Final Diagnosis"],
            }
        )
    return cases


def run_case(
    cases: List[Dict[str, str]],
    case_index: int,
    method_name: str = "MultiAgentToT",
    model: str = DEFAULT_MODEL,
    data_name: str = "DiagnosisArena",
) -> Dict[str, str]:
    if case_index < 0 or case_index >= len(cases):
        raise IndexError(f"case_index {case_index} is out of range")
    if method_name not in METHODS:
        available = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method '{method_name}'. Available methods: {available}")

    case = cases[case_index]
    method_cls = METHODS[method_name]
    method = method_cls(
        case_index=case_index,
        description=case["description"],
        evidence=case["evidence"],
        model=model,
        data_name=data_name,
    )
    return method.run()


def run_cases(
    cases: List[Dict[str, str]],
    case_indices: Optional[Iterable[int]] = None,
    method_name: str = "MultiAgentToT",
    model: str = DEFAULT_MODEL,
    data_name: str = "DiagnosisArena",
) -> List[Dict[str, str]]:
    indices = list(case_indices) if case_indices is not None else list(range(len(cases)))
    results = []
    for case_index in indices:
        result = run_case(
            cases=cases,
            case_index=case_index,
            method_name=method_name,
            model=model,
            data_name=data_name,
        )
        results.append(
            {
                "case_id": cases[case_index]["id"],
                "prediction": result.get("answer", ""),
                "report": result.get("report", ""),
                "answer": cases[case_index].get("answer", ""),
            }
        )
    return results
