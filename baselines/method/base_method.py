from abc import ABC, abstractmethod
from typing import Dict
from utils.public_function import evidence_retriever, llm_generate_response
from pathlib import Path
from datetime import datetime
import json
import re

class BaseMethod(ABC):
    def __init__(self, case_index: int, description: str, evidence, model: str = "gpt-5.1", data_name: str = "default"):
        self.case_index = case_index
        self.description = description
        self.evidence = evidence
        self.model = model
        self.data = data_name
        self._dataset_key = self._normalize_dataset_key(data_name)

    def _normalize_dataset_key(self, name: str) -> str:
        key = (name or "").strip()
        return key or "DiagnosisArena"

    def dataset_key(self) -> str:
        return self._dataset_key
    
    @abstractmethod
    def run(self) -> Dict[str, str]:
        pass

    def evidence_retriever(self, query: str) -> str:
        res = evidence_retriever(query, self.evidence)
        return res
    

    def llm(self, system_prompt, user_prompt):
        return llm_generate_response(user_prompt=user_prompt, model_path=self.model, temperature=1, max_tokens=4096, system_prompt=system_prompt, return_meta=False)
    
    
    def parse_first_json(self, s: str):
        s = s.strip()
        m = re.search(r"```(?:\w+)?\s*(.*?)\s*```", s, flags=re.S | re.I)
        if m:
            s = m.group(1).strip()

        try:
            return json.loads(s)
        except json.JSONDecodeError:
            print("LLM output is not strict JSON; trying to parse the first JSON object.")
            dec = json.JSONDecoder()
            for i, ch in enumerate(s):
                if ch in ('{', '['):
                    try:
                        obj, end = dec.raw_decode(s, idx=i)
                        print("Parsed first JSON object successfully.")
                        return obj
                    except json.JSONDecodeError:
                        continue
            raise ValueError("No valid JSON object found")

    def parse_thought_json(self, s: str):
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            print("LLM output is not strict JSON; trying to parse JSON objects.")
            json_blocks = []
            depth = 0
            start_idx = -1
            
            for i, char in enumerate(s):
                if char == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start_idx != -1:
                        json_blocks.append(s[start_idx:i+1])
                        start_idx = -1
            
            for block in json_blocks:
                try:
                    obj = json.loads(block)
                    if isinstance(obj, dict) and 'Thought' in obj:
                        print(f"Found a valid JSON object containing Thought.")
                        return obj
                except json.JSONDecodeError:
                    continue
            
            raise ValueError("No valid JSON object containing Thought found")
        
    
    def write_log(
        self,
        log_text: str,
        method: str,
        data_set: str,
        log_dir: Path = Path("logs"),
    ) -> Path:
        log_dir.mkdir(parents=True, exist_ok=True)
        data_type_dir = log_dir / self.dataset_key()
        method_dir = data_type_dir / method
        data_dir = method_dir / data_set
        case_dir = data_dir / f"case_{self.case_index}"
        case_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y_%m%d_%H%M")
        log_path = case_dir / f"{ts}.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"case_index: {self.case_index}\n")
            f.write(log_text.strip() + "\n")
        return log_path
