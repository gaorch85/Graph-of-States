from .base_method import BaseMethod
from typing import Dict
from prompts.baselines.DiagnosisArena import single_agent_CoT as diag_single_agent_CoT

class SingleAgentCoT(BaseMethod):

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
                    reasoning_history_str += f'answer: {step["Answer"]}\n\n'
            return reasoning_history_str


    def ReAct(self, max_steps: int = 10) -> str:
        system_prompt = diag_single_agent_CoT.system_prompt
        reasoning_history = []
        final_answer = ""
        final_report = ""
        current_step = 0
        while True:
            current_step += 1
            reasoning_history_str = self.formatted_reasoning_history(reasoning_history)
            user_prompt = diag_single_agent_CoT.user_prompt.format(
                description=self.description,
                reasoning_history=reasoning_history_str,
            )

            MAX_RETRY = 5
            for i in range(MAX_RETRY):
                response = self.llm(system_prompt=system_prompt, user_prompt=user_prompt)
                try:
                    data = self.parse_thought_json(response)
                    break
                except Exception as e:
                    print(f"Attempt {i+1} failed:", e)

            action_type = data.get("Action")
            if action_type in ["QueryText", "ToolCall", "Tool"]:
                question = data.get("Action Input", "")
                observation = self.evidence_retriever(query=question)
                reasoning_history.append({
                    "Thought": data.get("Thought", ""),
                    "Action": action_type,
                    "Action Input": question,
                    "Observation": observation,
                })
                if current_step >= max_steps:
                    system_prompt_finish = diag_single_agent_CoT.finish_system_prompt
                    user_prompt = diag_single_agent_CoT.finish_user_prompt.format(
                        description=self.description,
                        reasoning_history=self.formatted_reasoning_history(reasoning_history),
                    )
                    MAX_RETRY = 5
                    for i in range(MAX_RETRY):
                        response = self.llm(system_prompt=system_prompt_finish, user_prompt=user_prompt)
                        try:
                            data = self.parse_thought_json(response)
                            break
                        except Exception as e:
                            print(f"Attempt {i+1} failed:", e)
                    reasoning_history.append({
                        "Action": "Forced Finish",
                        "Thought": data["Thought"],
                        "Report": data["Report"],
                        "Answer": data["Answer"],
                    }) 
                    final_answer, final_report = data["Answer"], data["Report"]
                    print(f"Reached max steps {max_steps}; forcing finish") 
                    break
                else:
                    continue

            elif action_type == "Finish":
                reasoning_history.append({
                    "Action": data.get("Action"),
                    "Thought": data.get("Thought", ""),
                    "Report": data.get("Report", ""),
                    "Answer": data.get("Answer", ""),
                }) 
                final_answer, final_report = data.get("Answer", ""), data.get("Report", "")
                print(f"ReAct finished after {current_step} tool calls") 
                break
                
        completed_reasoning_process = self.formatted_reasoning_history(reasoning_history)

        return completed_reasoning_process, current_step, final_answer, final_report


    def run(self) -> Dict[str, str]:
        max_steps = 5
        completed_reasoning_process, current_step, final_answer, final_report = self.ReAct(max_steps=max_steps)
        
        detailed_log = f"Final Answer: {final_answer}\n\nFinal Report: {final_report}\n\nTool Call: {current_step}/{max_steps}\n\n\n\n[ReAct Detailed Trace]\n{completed_reasoning_process}"

        self.write_log(detailed_log, "SingleAgentCoT", self.data)
        print("Detailed log saved successfully!")
        res = {
            "answer": final_answer,
            "report": final_report
        }

        return res
