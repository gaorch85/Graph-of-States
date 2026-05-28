from .base_method import BaseMethod
from typing import Dict, List, Tuple
from prompts.baselines.DiagnosisArena import single_agent_ToT as diag_single_agent_ToT_prompts
import prompts.baselines.DiagnosisArena.GoT as got_prompts


class SingleAgentGoT(BaseMethod):
    
    def evaluate(self, reasoning_history: str):
        system_prompt = diag_single_agent_ToT_prompts.evaluate_system_prompt
        user_prompt = diag_single_agent_ToT_prompts.evaluate_user_prompt.format(
            description = self.description,
            reasoning_history = reasoning_history
        )
        response = self.llm(system_prompt, user_prompt)
        data = self.parse_first_json(response)
        score = data["Score"]
        detailed_log = f"\n External Expert Judge:\n [Thought]:  {data['Thought']}\n [Score]: {data['Score']}"
        print(f"Score: {data['Score']}")
        return score, detailed_log
    
    def evaluate_list(self, reasoning_history_list: List[str]):
        res = []
        for cur in reasoning_history_list:
            cur_res = self.evaluate(cur)
            res.append(cur_res)
        return res

    def expand(self, reasoning_history: str):
        system_prompt = diag_single_agent_ToT_prompts.system_prompt
        user_prompt = diag_single_agent_ToT_prompts.user_prompt.format(
            description = self.description,
            reasoning_history = reasoning_history
        )
        
        MAX_RETRY = 5
        for i in range(MAX_RETRY):
            response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                data = self.parse_thought_json(response)
                break
            except Exception as e:
                print(f"Attempt {i+1} failed:", e)
        
        return data
        
    def expand_list(self, reasoning_history: str, num) -> List[Dict]:
        res = []
        for _ in range(num):
            cur = self.expand(reasoning_history)
            res.append(cur)
        return res
    

    def _format_reasoning_list(self, reasoning_histories: List[str]) -> str:
        blocks = []
        for idx, content in enumerate(reasoning_histories, start=1):
            blocks.append(f"[Branch {idx}]\n{content}".strip())
        return "\n\n".join(blocks)

    def aggregate_node(self, reasoning_histories: List[str]) -> Dict:
        system_prompt = got_prompts.aggregate_system_prompt
        user_prompt = got_prompts.aggregate_user_prompt.format(
            description=self.description,
            reasoning_histories=self._format_reasoning_list(reasoning_histories),
            role="SingleAgent"
        )
        data = None
        MAX_RETRY = 3
        for i in range(MAX_RETRY):
            response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                data = self.parse_thought_json(response)
                break
            except Exception as e:
                print(f"[aggregate_node] Attempt {i+1} failed:", e)
        if data is None:
            print("[aggregate_node] fallback to expand on the first branch.")
            data = self.expand(reasoning_histories[0])
        return data

    
    def refine_node(self, reasoning_history: str) -> str:
        system_prompt = got_prompts.refine_system_prompt
        user_prompt = got_prompts.refine_user_prompt.format(
            description=self.description,
            reasoning_history=reasoning_history,
            role="SingleAgent"
        )
        MAX_RETRY = 3
        for i in range(MAX_RETRY):
            response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                data = self.parse_first_json(response)
                refine_text = data.get("refine_thought", "").strip()
                if refine_text:
                    addition = f"Thought: {refine_text}\n\n"
                    return reasoning_history + addition
            except Exception as e:
                print(f"[refine_node] Attempt {i+1} failed:", e)
            print(f"[refine_node] Attempt {i+1} returned empty text.")
        return reasoning_history

            
    def decide_action(self, node_list: List[Dict], allow_refine: bool = True, allow_aggregate: bool = True) -> Tuple[str, List[Dict], List[Tuple[float, str]]]:
        if not node_list:
            return "expand", [], []
        reasoning_histories = [node["reasonning_history"] for node in node_list]
        eval_results = self.evaluate_list(reasoning_histories)
        for node, (score, _) in zip(node_list, eval_results):
            node["score"] = score

        sorted_nodes = sorted(node_list, key=lambda x: x["score"], reverse=True)
        if allow_aggregate and len(sorted_nodes) >= 2 and sorted_nodes[1]["score"] >= 0.65:
            return "aggregate", sorted_nodes[:2], eval_results
        if allow_refine and sorted_nodes and sorted_nodes[0]["score"] < 0.7:
            return "refine", [sorted_nodes[0]], eval_results
        return "expand", [sorted_nodes[0]], eval_results        


    def node_to_str(self, reasoning_history: str, current_info: Dict) -> str:
        new_reasoning_str = ""
        if current_info.get("Action") in ["QueryText", "ToolCall", "Tool"]:
            new_reasoning_str += f'Thought: {current_info["Thought"]}\n'
            new_reasoning_str += f'Action: {current_info["Action"]}["{current_info["Action Input"]}"]\n'
            new_reasoning_str += f'Observation: {current_info["Observation"]}\n\n'

        else:
            new_reasoning_str += f'Thought: {current_info["Thought"]}\n'
            new_reasoning_str += f'Action: {current_info["Action"]}\n'
            new_reasoning_str += f'report: {current_info["Report"]}\n'
            new_reasoning_str += f'answer: {current_info["Answer"]}\n\n'

        return reasoning_history + new_reasoning_str
    
    def node_list_to_str(self, reasoning_history: str, current_info_list: List[Dict]) -> List[str]:
        res = []
        for info in current_info_list:
            info_str = self.node_to_str(reasoning_history, info)
            res.append(info_str)
        return res
    
    def node_list_get_observation(self, node_list: List[Dict]):
        for node in node_list:
            if node.get("Action") in ["QueryText", "ToolCall", "Tool"]:
                observation = self.evidence_retriever(node["Action Input"])
                node["Observation"] = observation
                print(node)
        return None
    
    def get_answer_from_best_open(self, reasoning_history: str):
        system_prompt = diag_single_agent_ToT_prompts.finish_system_prompt
        user_prompt = diag_single_agent_ToT_prompts.finish_user_prompt.format(
            description = self.description,
            reasoning_history = reasoning_history
        )
        MAX_RETRY = 5
        for i in range(MAX_RETRY):
            response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                data = self.parse_thought_json(response)
                break
            except Exception as e:
                print(f"Attempt {i+1} failed:", e)
        
        return data
    
    def eff_score(self, node, depth_weight=0.03):
        return node["score"] + depth_weight * node["depth"]

    def is_better(self, a, b, depth_weight=0.02):
        return self.eff_score(a, depth_weight) > self.eff_score(b, depth_weight)

    def DFS(self, max_steps: int = 15):
        EXPAND_NUM = 2
        accept_threshold = 0.9
        kill_threshold = 0.5

        frontier = [
            {
                "reasonning_history": "",
                "score": 1.0,
                "depth": 0
            }
        ]
        candidate_finish = []
        best_open = None
        current_step = 0
        allow_refine = True
        force_expand_only = False

        while frontier and current_step < max_steps:
            current_step += 1
            action, selected_nodes, _ = self.decide_action(
                frontier,
                allow_refine=allow_refine,
                allow_aggregate=not force_expand_only
            )
            if not selected_nodes:
                break

            if action == "refine":
                target = selected_nodes[0]
                target["reasonning_history"] = self.refine_node(target["reasonning_history"])
                print("[GoT] refining current branch")
                allow_refine = False
                force_expand_only = False
                continue

            if action == "aggregate":
                chosen = selected_nodes
                combined_history = "\n\n".join(node["reasonning_history"] for node in chosen)
                agg_action = self.aggregate_node([node["reasonning_history"] for node in chosen])
                if agg_action.get("Action") in ["QueryText", "ToolCall", "Tool"]:
                    observation = self.evidence_retriever(agg_action.get("Action Input", ""))
                    agg_action["Observation"] = observation
                agg_trace = self.node_to_str(combined_history, agg_action)
                agg_score, _ = self.evaluate(agg_trace)
                new_depth = max(node.get("depth", 0) for node in chosen) + 1
                for node in chosen:
                    if node in frontier:
                        frontier.remove(node)

                if agg_action.get("Action") == "Finish":
                    print(f"[GoT] aggregate produced Finish，score={agg_score}")
                    if agg_score >= accept_threshold:
                        final_answer = agg_action.get("Answer", "")
                        final_report = agg_action.get("Report", "")
                        detailed_trace = agg_trace
                        detailed_log = f"Final Answer: {final_answer}\n\nFinal Report: {final_report}\n\n Steps: {current_step}/{max_steps}\n\n\n\n[ReAct Detailed Trace]\n{detailed_trace}"
                        return detailed_log, final_answer, final_report
                    candidate_finish.append({
                        "answer": agg_action.get("Answer", ""),
                        "report": agg_action.get("Report", ""),
                        "current_path_trace": agg_trace,
                        "score": agg_score
                    })
                else:
                    new_node = {
                        "reasonning_history": agg_trace,
                        "score": agg_score,
                        "depth": new_depth
                    }
                    frontier.append(new_node)
                    if best_open is None or self.is_better(new_node, best_open):
                        best_open = new_node
                allow_refine = False
                force_expand_only = True
                continue

            target = selected_nodes[0]
            if target in frontier:
                frontier.remove(target)
            current_depth = target.get("depth", 0)
            print(f"Expanding node with score {target.get('score', 'N/A')}, depth {current_depth}")

            children = self.expand_list(target["reasonning_history"], EXPAND_NUM)
            self.node_list_get_observation(children)
            children_str = self.node_list_to_str(target["reasonning_history"], children)
            children_score = self.evaluate_list(children_str)

            to_push = []
            for child_info, child_trace, (score, _) in zip(children, children_str, children_score):
                if child_info["Action"] == "Finish":
                    print(f"Finish node found，score: {score}")
                    if score >= accept_threshold:
                        final_answer = child_info["Answer"]
                        final_report = child_info["Report"]
                        detailed_trace = child_trace
                        detailed_log = f"Final Answer: {final_answer}\n\nFinal Report: {final_report}\n\n Steps: {current_step}/{max_steps}\n\n\n\n[ReAct Detailed Trace]\n{detailed_trace}"
                        return detailed_log, final_answer, final_report

                    candidate_finish.append({
                        "answer": child_info["Answer"],
                        "report": child_info["Report"],
                        "current_path_trace": child_trace,
                        "score": score
                    })
                elif score < kill_threshold:
                    print(f"Current child score: {score}, is below threshold {kill_threshold}; dropping")
                    continue
                else:
                    child_node = {
                        "reasonning_history": child_trace,
                        "score": score,
                        "depth": current_depth + 1,
                    }
                    to_push.append(child_node)
                    if best_open is None or self.is_better(child_node, best_open):
                        best_open = child_node

            to_push.sort(key=lambda x: x["score"])
            frontier.extend(to_push)
            allow_refine = True
            force_expand_only = False

        if candidate_finish:
            best_finish = max(candidate_finish, key=lambda x: x["score"])
            final_answer = best_finish["answer"]
            final_report = best_finish["report"]
            detailed_trace = best_finish["current_path_trace"]
        elif best_open is not None:
            print(f"No finish candidate; forcing answer from best_open with score: {best_open['score']}")
            res = self.get_answer_from_best_open(best_open["reasonning_history"])
            res["Action"] = "Forced Finish"
            final_answer = res["Answer"]
            final_report = res["Report"]
            detailed_trace = self.node_to_str(best_open["reasonning_history"], res)
        else:
            print("Case failed: no finish candidate or best_open")
            final_answer = "No Answer"
            final_report = "No Report"
            detailed_trace = "No Trace"

        detailed_log = f"Final Answer: {final_answer}\n\nFinal Report: {final_report}\n\n Steps: {current_step}/{max_steps}\n\n\n\n[ReAct Detailed Trace]\n{detailed_trace}"
        return detailed_log, final_answer, final_report


    def run(self) -> Dict[str, str]:
        max_steps = 5

        detailed_log, final_answer, final_report = self.DFS(max_steps)

        self.write_log(detailed_log, "SingleAgentGoT", self.data)
        print("Detailed log saved successfully!")
        res = {
            "answer": final_answer,
            "report": final_report
        }

        return res
