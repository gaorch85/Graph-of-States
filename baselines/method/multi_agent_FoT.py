from .base_method import BaseMethod
from typing import Dict, List, Tuple
import json
from prompts.baselines.DiagnosisArena import multi_agent_ToT as diag_multi_agent_ToT_prompts
from prompts.baselines.DiagnosisArena import FoT as FoT_prompts


DIAGNOSIS_EXPERT_PROMPTS: Dict[str, str] = {
    "Primary Physician": diag_multi_agent_ToT_prompts.Primary_Physician_system_prompt,
    "Laboratory Physician": diag_multi_agent_ToT_prompts.Laboratory_Physician_system_prompt,
    "Pathologist": diag_multi_agent_ToT_prompts.Pathologist_system_prompt,
    "Radiologist": diag_multi_agent_ToT_prompts.Radiologist_system_prompt,
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


class MultiAgentFoT(BaseMethod):

    def _get_prompt_map(self) -> Dict[str, str]:
        key = self.dataset_key()
        return EXPERT_PROMPTS.get(key, EXPERT_PROMPTS["DiagnosisArena"])

    def _get_expert_list(self) -> List[str]:
        key = self.dataset_key()
        return EXPERT_LISTS.get(key, EXPERT_LISTS["DiagnosisArena"])

    def evaluate(self, expert: str, reasoning_history: str) -> Tuple[float, str]:
        system_prompt = diag_multi_agent_ToT_prompts.evaluate_system_prompt
        user_prompt = diag_multi_agent_ToT_prompts.evaluate_user_prompt.format(
            description=self.description,
            reasoning_history=reasoning_history
        )
        response = self.llm(system_prompt, user_prompt)
        data = self.parse_first_json(response)
        score_raw = data.get("Score", 0)
        try:
            score = float(score_raw)
        except Exception:
            score = 0.0
        detailed_log = f"\n External Expert Judge ({expert}):\n [Thought]:  {data.get('Thought', '')}\n [Score]: {score}"
        print(f"[{expert}][evaluate] Score: {score}")
        return score, detailed_log

    def evaluate_list(self, expert: str, reasoning_history_list: List[str]):
        res = []
        for cur in reasoning_history_list:
            cur_res = self.evaluate(expert, cur)
            res.append(cur_res)
        return res

    def expand(self, expert: str, reasoning_history: str):
        role_prompt = self._get_prompt_map().get(expert, "")
        system_prompt = role_prompt + "\n" + diag_multi_agent_ToT_prompts.ToT_system_prompt
        user_prompt = diag_multi_agent_ToT_prompts.ToT_user_prompt.format(
            description=self.description,
            reasoning_history=reasoning_history
        )
        
        MAX_RETRY = 5
        data = None
        for i in range(MAX_RETRY):
            response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                data = self.parse_thought_json(response)
                break
            except Exception as e:
                print(f"[expand] {expert} Attempt {i+1} failed:", e)
        
        if data is None:
            raise RuntimeError(f"expand for expert '{expert}' failed to produce valid JSON after {MAX_RETRY} attempts.")
        return data
        
    def expand_list(self, expert: str, reasoning_history: str, num: int) -> List[Dict]:
        res = []
        for _ in range(num):
            cur = self.expand(expert, reasoning_history)
            res.append(cur)
        return res

    def node_to_str(self, reasoning_history: str, current_info: Dict) -> str:
        addition = ""
        if current_info.get("Action") in ["QueryText", "ToolCall", "Tool"]:
            addition += f'Thought: {current_info.get("Thought", "")}\n'
            addition += f'Action: {current_info.get("Action", "QueryText")}["{current_info.get("Action Input", "")}"]\n'
            addition += f'Observation: {current_info.get("Observation", "")}\n\n'

        else:
            addition += f'Thought: {current_info.get("Thought", "")}\n'
            addition += f'Action: {current_info.get("Action", "")}\n'
            addition += f'report: {current_info.get("Report", "")}\n'
            if "Answer" in current_info:
                addition += f'answer: {current_info.get("Answer", "")}\n'
            addition += "\n"

        return reasoning_history + addition
    
    def node_list_to_str(self, reasoning_history: str, current_info_list: List[Dict]) -> List[str]:
        res = []
        for info in current_info_list:
            info_str = self.node_to_str(reasoning_history, info)
            res.append(info_str)
        return res
    
    def node_list_get_observation(self, node_list: List[Dict]):
        for node in node_list:
            if node.get("Action") in ["QueryText", "ToolCall", "Tool"]:
                observation = self.evidence_retriever(node.get("Action Input", ""))
                node["Observation"] = observation
                print(f"[observation] {node}")
        return None
    
    def get_report_from_best_open(self, expert: str, reasoning_history: str):
        role_prompt = self._get_prompt_map().get(expert, "")
        system_prompt = role_prompt + "\n" + diag_multi_agent_ToT_prompts.finish_system_prompt
        user_prompt = diag_multi_agent_ToT_prompts.finish_user_prompt.format(
            description=self.description,
            reasoning_history=reasoning_history
        )
        MAX_RETRY = 5
        finish_data = None
        for i in range(MAX_RETRY):
            response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                finish_data = self.parse_thought_json(response)
                break
            except Exception as e:
                print(f"[finish] {expert} Attempt {i+1} failed:", e)
        
        if finish_data is None:
            raise RuntimeError(f"Forced finish for expert '{expert}' failed to produce valid JSON.")
        
        return finish_data
    
    def eff_score(self, node: Dict, depth_weight: float = 0.03):
        return node["score"] + depth_weight * node["depth"]

    def is_better(self, a: Dict, b: Dict, depth_weight: float = 0.02):
        return self.eff_score(a, depth_weight) > self.eff_score(b, depth_weight)

    def DFS(self, expert: str, max_steps: int = 12):
        EXPAND_NUM = 2
        accept_threshold = 0.9
        kill_threshold = 0.5

        root = {
            "reasoning_history": "",
            "score": 1.0,
            "depth": 0
        }

        stack = [root]
        candidate_finish = []
        best_open = None

        current_step = 0

        while stack and current_step < max_steps:
            current_step += 1
            current_node = stack.pop()
            current_depth = current_node["depth"]
            print(f"[DFS] {expert} Expanding node with score {current_node['score']}, depth {current_depth}")

            children = self.expand_list(expert, current_node["reasoning_history"], EXPAND_NUM)
            self.node_list_get_observation(children)
            children_str = self.node_list_to_str(current_node["reasoning_history"], children)
            children_score = self.evaluate_list(expert, children_str)

            to_push = []

            for idx in range(0, EXPAND_NUM):
                child = children[idx]
                child_score = children_score[idx][0]
                if child.get("Action") == "Finish":
                    print(f"[DFS] {expert} Finish node found，score: {child_score}")
                    if child_score >= accept_threshold:
                        print(f"[DFS] {expert} Finish node found with score {child_score}, above {accept_threshold}; returning immediately")
                        final_report = child.get("Report", "")
                        detailed_trace = children_str[idx]
                        return detailed_trace, current_step, final_report

                    candidate_finish.append({
                        "report": child.get("Report", ""),
                        "current_path_trace": children_str[idx],
                        "score": child_score
                    })

                elif child_score < kill_threshold:
                    print(f"[DFS] {expert} Current child score: {child_score}, is below threshold {kill_threshold}; dropping")
                    continue
                
                else:
                    child_node = {
                        "reasoning_history": children_str[idx],
                        "score": child_score,
                        "depth": current_depth + 1,
                    }
                    to_push.append(child_node)

                    if best_open is None:
                        best_open = child_node
                    else:
                        if self.is_better(child_node, best_open):
                            print(f"[DFS] {expert} Updated best_open")
                            print(f"[DFS] {expert} New best_open: score: {child_node['score']} depth: {child_node['depth']}")
                            print(f"[DFS] {expert} Previous best_open: score: {best_open['score']} depth: {best_open['depth']}")
                            best_open = child_node

            to_push.sort(key=lambda x: x["score"])
            for item in to_push:
                stack.append(item)

        
        if candidate_finish:
            best_finish = max(candidate_finish, key=lambda x: x["score"])
            final_report = best_finish["report"]
            detailed_trace = best_finish["current_path_trace"]
            
        elif best_open is not None:
            print(f"[DFS] {expert} No finish candidate; forcing answer from best_open with score: {best_open['score']}")
            res = self.get_report_from_best_open(expert, best_open["reasoning_history"])
            res["Action"] = res.get("Action", "Forced Finish")
            final_report = res.get("Report", "")
            detailed_trace = self.node_to_str(best_open["reasoning_history"], res)

        else:
            print(f"[DFS] {expert} Case failed: no finish candidate or best_open")
            final_report = "No Report"
            detailed_trace = "No Trace"

        return detailed_trace, current_step, final_report


    def execute_tasks(self, max_steps=6):
        analysis_reports = {}
        reasoning_histories = {}
        expert_list = self._get_expert_list()
        self._expert_list = expert_list
        for expert in expert_list:
            print(f"[execute_tasks] Running DFS for expert: {expert}")
            detailed_trace, steps, final_report = self.DFS(
                expert=expert,
                max_steps=max_steps,
            )
            analysis_reports[expert] = {
                "steps": steps,
                "report": final_report,
            }
            reasoning_histories[expert] = detailed_trace
        return analysis_reports, reasoning_histories


    def summary(self, analysis_reports) -> Tuple[str, str]:
        system_prompt = diag_multi_agent_ToT_prompts.Primary_Physician_system_prompt +  diag_multi_agent_ToT_prompts.summary_system_prompt
        expert_reports_str = json.dumps(
            analysis_reports, ensure_ascii=False, indent=2
        )
        user_prompt = diag_multi_agent_ToT_prompts.summary_user_prompt.format(
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

    def aggregate(self, final_answer_list: List[str], final_report_list: List[str]):
        system_prompt = FoT_prompts.aggreate_FoT_system_prompt
        solutions_str = ""
        for idx, (ans, rep) in enumerate(zip(final_answer_list, final_report_list)):
            solutions_str += f"Solution {idx+1}:\nAnswer: {ans}\nReport: {rep}\n\n"
        
        user_prompt = FoT_prompts.aggreate_FoT_user_prompt.format(
            solutions=solutions_str
        )
        
        response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
        data = self.parse_first_json(response)
        final_answer = data["Answer"]
        final_report = data["Report"]
        return final_answer, final_report

    def run(self) -> Dict[str, str]:
        N = 4
        
        final_answer_list = []
        final_report_list = []
        
        for _i in range(N):
            print(f"FoT Sample {_i + 1}/{N}")
            analysis_reports, reasoning_histories = self.execute_tasks(max_steps=3)
            final_answer, final_report = self.summary(analysis_reports)
            expert_list = getattr(self, "_expert_list", None) or self._get_expert_list()

            log_text_parts = [
                "=== Experts ===",
                json.dumps(expert_list, ensure_ascii=False, indent=2),
                "",
                "=== Expert Analysis Reports ===",
                json.dumps(analysis_reports, ensure_ascii=False, indent=2),
                "",
                "=== ToT Reasoning Histories ===",
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
            self.write_log(log_text, "MultiAgentFoT", self.data)
            
            final_answer_list.append(final_answer)
            final_report_list.append(final_report)

        final_answer, final_report = self.aggregate(final_answer_list, final_report_list)
        res = {
            "answer": final_answer,
            "report": final_report
        }
        return res
