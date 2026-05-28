from .base_method import BaseMethod
from typing import Dict, Tuple, List
import json
from prompts.baselines.DiagnosisArena import multi_agent_CoT as diag_multi_agent_CoT_prompts


DIAGNOSIS_EXPERT_PROMPTS: Dict[str, str] = {
    "Primary Physician": diag_multi_agent_CoT_prompts.Primary_Physician_system_prompt,
    "Laboratory Physician": diag_multi_agent_CoT_prompts.Laboratory_Physician_system_prompt,
    "Pathologist": diag_multi_agent_CoT_prompts.Pathologist_system_prompt,
    "Radiologist": diag_multi_agent_CoT_prompts.Radiologist_system_prompt,
}

DIAGNOSIS_EXPERT_LIST: List[str] = [
    "Laboratory Physician",
    "Pathologist",
    "Radiologist",
]

EXPERT_PROMPTS: Dict[str, Dict[str, str]] = {
    "DiagnosisArena": DIAGNOSIS_EXPERT_PROMPTS,
}

EXPERT_LISTS: Dict[str, List[str]] = {
    "DiagnosisArena": DIAGNOSIS_EXPERT_LIST,
}


class MultiAgentCoT(BaseMethod):
    def _get_prompt_map(self) -> Dict[str, str]:
        key = self.dataset_key()
        return EXPERT_PROMPTS.get(key, EXPERT_PROMPTS["DiagnosisArena"])

    def _get_expert_list(self) -> List[str]:
        key = self.dataset_key()
        return EXPERT_LISTS.get(key, EXPERT_LISTS["DiagnosisArena"])

    def formatted_reasoning_history(self, reasoning_history: list) -> str:
        if reasoning_history is None or len(reasoning_history) == 0:
            return "<empty>"
        else:
            reasoning_history_str = ""
            for step in reasoning_history:
                if step["Action"] in ["QueryText", "ToolCall", "Tool"]:
                    reasoning_history_str += f'Thought: {step["Thought"]}\n'
                    reasoning_history_str += f'Action: {step["Action"]}["{step["Action Input"]}"]\n'
                    reasoning_history_str += f'Observation: {step["Observation"]}\n\n'
                else:
                    reasoning_history_str += f'Thought: {step["Thought"]}\n'
                    reasoning_history_str += f'Action: {step["Action"]}\n'
                    reasoning_history_str += f'report: {step["Report"]}\n'
            return reasoning_history_str

    def ReAct(self, expert: str, max_steps: int = 10) -> Tuple[str, int, str]:
        role_prompt = self._get_prompt_map().get(expert, "")
        base_system_prompt = role_prompt + "\n" + diag_multi_agent_CoT_prompts.ReAct_system_prompt

        reasoning_history = []
        final_report = ""
        current_step = 0

        while True:
            current_step += 1
            system_prompt = base_system_prompt
            reasoning_history_str = self.formatted_reasoning_history(reasoning_history)
            user_prompt = diag_multi_agent_CoT_prompts.ReAct_user_prompt.format(
                description=self.description,
                reasoning_history=reasoning_history_str,
            )

            MAX_RETRY = 5
            data = None
            for i in range(MAX_RETRY):
                response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
                print(f"[ReAct] {expert} - Step {current_step} Response:", response)
                try:
                    data = self.parse_thought_json(response)
                    break
                except Exception as e:
                    print(f"[ReAct] {expert} - Attempt {i+1} failed:", e)

            if data is None:
                raise RuntimeError(f"ReAct for expert '{expert}' failed to produce valid JSON after {MAX_RETRY} attempts.")

            if data.get("Action") in ["QueryText", "ToolCall", "Tool"]:
                question = data.get("Action Input", "")
                observation = self.evidence_retriever(query=question)
                reasoning_history.append({
                    "Thought": data.get("Thought", ""),
                    "Action": data.get("Action", "QueryText"),
                    "Action Input": question,
                    "Observation": observation,
                })
                if current_step >= max_steps:
                    finish_system_prompt = role_prompt + "\n" + diag_multi_agent_CoT_prompts.ReAct_finish_system_prompt
                    finish_user_prompt = diag_multi_agent_CoT_prompts.ReAct_finish_user_prompt.format(
                        description=self.description,
                        reasoning_history=self.formatted_reasoning_history(reasoning_history),
                    )
                    finish_data = None
                    for i in range(MAX_RETRY):
                        response = self.llm(
                            system_prompt=finish_system_prompt,
                            user_prompt=finish_user_prompt
                        )
                        try:
                            finish_data = self.parse_thought_json(response)
                            break
                        except Exception as e:
                            print(f"[ReAct-Finish] {expert} - Attempt {i+1} failed:", e)

                    if finish_data is None:
                        raise RuntimeError(f"Forced finish for expert '{expert}' failed to produce valid JSON.")

                    reasoning_history.append({
                        "Action": "Forced Finish",
                        "Thought": finish_data.get("Thought", ""),
                        "Report": finish_data.get("Report", ""),
                    })
                    final_report = finish_data.get("Report", "")
                    print(f"[ReAct] {expert}: Reached max steps {max_steps}; forcing finish")
                    break
                else:
                    continue

            elif data.get("Action") == "Finish":
                reasoning_history.append({
                    "Action": data.get("Action", "Finish"),
                    "Thought": data.get("Thought", ""),
                    "Report": data.get("Report", ""),
                })
                final_report = data.get("Report", "")
                print(f"[ReAct] {expert}: finished after {current_step} tool calls")
                break

            else:
                reasoning_history.append({
                    "Action": data.get("Action", "Unknown"),
                    "Thought": data.get("Thought", ""),
                    "Report": data.get("Report", ""),
                })
                final_report = data.get("Report", "")
                print(f"[ReAct] {expert}: received unknown Action; treating as finished.")
                break

        completed_reasoning_process = self.formatted_reasoning_history(reasoning_history)
        return completed_reasoning_process, current_step, final_report

    def execute_tasks(self):
        analysis_reports = {}
        reasoning_histories = {}
        expert_list = self._get_expert_list()
        self._expert_list = expert_list
        for expert in expert_list:
            print(f"[execute_tasks] Running ReAct for expert: {expert}")
            completed_reasoning_process, steps, final_report = self.ReAct(
                expert=expert,
                max_steps=3,
            )
            analysis_reports[expert] = {
                "steps": steps,
                "report": final_report,
            }
            reasoning_histories[expert] = completed_reasoning_process
        return analysis_reports, reasoning_histories


    def summary(self, analysis_reports) -> Tuple[str, str]:
        system_prompt = diag_multi_agent_CoT_prompts.Primary_Physician_system_prompt +  diag_multi_agent_CoT_prompts.summary_system_prompt
        expert_reports_str = json.dumps(
            analysis_reports, ensure_ascii=False, indent=2
        )
        user_prompt = diag_multi_agent_CoT_prompts.summary_user_prompt.format(
            description=self.description,
            expert_reports=expert_reports_str,
        )

        MAX_RETRY = 5
        data = None
        for i in range(MAX_RETRY):
            response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                data = self.parse_first_json(response)
                break
            except Exception as e:
                print("[summary] Attempt", i + 1, "failed:", e)

        if data is None:
            raise RuntimeError("summary failed to produce valid JSON.")

        final_answer = data.get("Answer", "")
        final_report = data.get("Report", "")

        return final_answer, final_report


    def run(self) -> Dict[str, str]:
        analysis_reports, reasoning_histories = self.execute_tasks()
        final_answer, final_report = self.summary(analysis_reports)
        expert_list = getattr(self, "_expert_list", None) or self._get_expert_list()

        log_text_parts = [
            "=== Experts ===",
            json.dumps(expert_list, ensure_ascii=False, indent=2),
            "",
            "=== Expert Analysis Reports ===",
            json.dumps(analysis_reports, ensure_ascii=False, indent=2),
            "",
            "=== ReAct Reasoning Histories ===",
        ]
        for expert, reasoning_history in reasoning_histories.items():
            log_text_parts.append(f"[{expert}]")
            log_text_parts.append(reasoning_history)
            log_text_parts.append("")
        log_text_parts += [
            "=== Final Answer ===",
            final_answer,
            "",
            "=== Final Report ===",
            final_report,
        ]
        log_text = "\n".join(log_text_parts)
        self.write_log(log_text, "MultiAgentCoT", self.data)

        res = {
            "answer": final_answer,
            "report": final_report
        }
        return res
