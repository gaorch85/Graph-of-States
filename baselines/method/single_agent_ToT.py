from .base_method import BaseMethod
from typing import Dict, List
from prompts.baselines.DiagnosisArena import single_agent_ToT as diag_single_agent_ToT_prompts


class SingleAgentToT(BaseMethod):
    
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

    def DFS(self, max_steps = 15):
        EXPAND_NUM = 2
        accept_threshold = 0.9
        kill_threshold = 0.5

        root = {
            "reasonning_history": "",
            "score": 1.0,
            "depth": 0
        }

        stack = [root]
        candiadate_finish = []
        best_open = None

        current_step = 0

        while stack and current_step < max_steps:
            current_step += 1
            current_node = stack.pop()
            current_depth = current_node["depth"]
            print(f"Expanding node with score {current_node['score']}, depth {current_depth}")

            children = self.expand_list(current_node["reasonning_history"], EXPAND_NUM) 
            self.node_list_get_observation(children)
            children_str = self.node_list_to_str(current_node["reasonning_history"], children)
            children_score = self.evaluate_list(children_str)

            to_push = []

            for _ in range(0, EXPAND_NUM):
                if children[_]["Action"] == "Finish":
                    print("Finish node found")
                    print(f"score: {children_score[_][0]}")
                    if children_score[_][0] >= accept_threshold:
                        print(f"Finish node found with score {children_score[_][0]}, above 0.9; returning immediately")
                        final_answer = children[_]["Answer"]
                        final_report = children[_]["Report"]
                        detailed_trace = children_str[_]
                        detailed_log = f"Final Answer: {final_answer}\n\nFinal Report: {final_report}\n\n Steps: {current_step}/{max_steps}\n\n\n\n[ReAct Detailed Trace]\n{detailed_trace}"
                                            
                        return detailed_log, final_answer, final_report


                    candiadate_finish.append({
                        "answer": children[_]["Answer"],
                        "report": children[_]["Report"],
                        "current_path_trace": children_str[_],
                        "score": children_score[_][0]
                    })

                
                elif children_score[_][0] < kill_threshold:
                    print(f"Current child score: {children_score[_][0]}, is below threshold {kill_threshold}; dropping")
                    continue
                
                else:
                    child_node = {
                        "reasonning_history": children_str[_],
                        "score": children_score[_][0],
                        "depth": current_depth + 1,
                    }
                    to_push.append(child_node)

                    if best_open is None:
                        best_open = child_node
                    else:
                        if self.is_better(child_node, best_open):
                            print("Updated best_open")
                            print(f"New best_open: score: {child_node['score']} depth: {child_node['depth']}")
                            print(f"Previous best_open: score: {best_open['score']} depth: {best_open['depth']}")
                            best_open = child_node

            to_push.sort(key=lambda x: x["score"])
            for item in to_push:
                stack.append(item)

            
        
        if candiadate_finish:
            best_finish = max(candiadate_finish, key=lambda x: x["score"])
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

        self.write_log(detailed_log, "SingleAgentToT", self.data)
        print("Detailed log saved successfully!")
        res = {
            "answer": final_answer,
            "report": final_report
        }

        return res
