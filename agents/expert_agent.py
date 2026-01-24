import json
from utils.public_function import *
from utils.IT_tools import Telemetry
from prompts.prompt_generation import *
from pathlib import Path
from datetime import datetime

class Expert_Agent:
    """
    
    """

    def __init__(self, name, graph, fsm, case_id,  dataset, log_path, cfg_domain, llm_args, evidence_text, max_retrieval_steps):
        self.name = name
        self.graph = graph
        self.fsm = fsm
        self.case_id = case_id
        self.dataset = dataset
        self.log_path = log_path
        self.cfg_domain = cfg_domain
        self.llm_args = llm_args
        self.evidence_text = evidence_text
        self.max_retrieval_steps = max_retrieval_steps
        self.retrieval_history = []
        self.telemetry = None
        if self.dataset == "Micro" and isinstance(evidence_text, dict):
            try:
                self.telemetry = Telemetry.from_case_resources(case_id=case_id, resources=evidence_text)
            except Exception as e:
                print(f"Failed to init telemetry for case {case_id}: {e}")
                self.telemetry = None
    

    def run(self, frontier):
       
        tool_names = self.cfg_domain.get("tool_names")
        tool_descriptions = self.cfg_domain.get("tool_descriptions")
        
        if self.dataset == "Micro":
            return self._run_micro(frontier)

        
        max_retrieve_step = self.max_retrieval_steps 
        retrieve_step = 0
        final_analysis = None
        retrieval_history = self.retrieval_history 
        decision_result: Optional[dict] = None  

        if len(tool_names) == 1 and tool_names[0] == "Evidence_Text":
            tool_prompt_base = "We provide only pre-prepared auxiliary evidence texts for you, and you must extract related information from the evidence texts."
           
            while retrieve_step < max_retrieve_step and final_analysis is None:
                
                if retrieve_step >= max_retrieve_step - 1:  
                    tool_prompt = f"{tool_prompt_base}. NOTE: You have reached the maximum number of retrieval steps (max={max_retrieve_step}), your decision MUST be 'analyze' (even if information is insufficient)."
                else:
                    tool_prompt = tool_prompt_base + "Your decision MUST be 'retrieve' only to gather more complete informations."

               
                print(f"\nStep {retrieve_step + 1} - Decision Making")
                log_to_file(f"\nStep {retrieve_step + 1} - Decision Making",log_path=self.log_path)
                system_p1, user_p1 = construct_expert_analyze_prompt(
                    dataset=self.dataset,
                    expert_name=self.name,
                    tool_prompt=tool_prompt,
                    belief=self.graph.belief,
                    history=retrieval_history,
                    frontier=frontier,
                    task_stage="decision_making"  
                )
                
               
                response_p1, meta_p1 = llm_generate_response(
                    user_prompt=user_p1,
                    model_path=self.llm_args["model_path"],
                    temperature=self.llm_args["temperature"],
                    max_tokens=self.llm_args["max_tokens"],
                    system_prompt=system_p1,
                    return_meta=self.llm_args["return_meta"]
                )
                print(f"Step {retrieve_step + 1} Decision: {response_p1}")

               
                log_to_file(f"Step {retrieve_step + 1} Decision: {response_p1}\n\n",log_path=self.log_path)
                
                try:
                    decision_result = parse_json_response(response_p1.strip())
                    
                    if not all(k in decision_result for k in ["type", "decision"]):
                        raise ValueError("Missing mandatory fields (type/decision) in decision response")
                   
                    if decision_result["decision"] not in ["retrieve", "analyze"]:
                        raise ValueError(f"Invalid decision value: {decision_result['decision']} (only 'retrieve'/'analyze' allowed for type1)")
                    current_type = decision_result["type"]
                    current_decision = decision_result["decision"]
                    print(f"Step {retrieve_step + 1} parsed result: type={current_type}, decision={current_decision}")
                    log_to_file(f"Step {retrieve_step + 1} parsed result: type={current_type}, decision={current_decision}\n\n",log_path=self.log_path)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Failed parsing: {e}")
                    log_to_file(f"Failed parsing: {e}\n\n",log_path=self.log_path)
                    retrieve_step += 1
                    continue

                
                print(f"Step {retrieve_step + 1} - Content Generation")
                log_to_file(f"Step {retrieve_step + 1} - Content Generation\n\n",log_path=self.log_path)
                system_p2, user_p2 = construct_expert_analyze_prompt(
                    dataset=self.dataset,
                    expert_name=self.name,
                    tool_prompt=tool_prompt,
                    belief=self.graph.belief,
                    history=retrieval_history,
                    frontier=frontier,
                    task_stage="content_generation", 
                    decision_result=decision_result  
                )
                
                
                response_p2, meta_p2 = llm_generate_response(
                    user_prompt=user_p2,
                    model_path=self.llm_args["model_path"],
                    temperature=self.llm_args["temperature"],
                    max_tokens=self.llm_args["max_tokens"],
                    system_prompt=system_p2,
                    return_meta=self.llm_args["return_meta"]
                )
                print(f"Step {retrieve_step + 1} reponse: {response_p2}")
                log_to_file(f"Step {retrieve_step + 1} reponse: {response_p2}\n\n",log_path=self.log_path)


                
                try:
                    final_response_dict = parse_json_response(response_p2.strip())
                    if not all(k in final_response_dict for k in ["type", "decision", "content"]):
                        raise ValueError("Missing mandatory fields (type/decision/content) in final response")
                    if (final_response_dict["type"] != current_type) or (final_response_dict["decision"] != current_decision):
                        raise ValueError(f"Content stage result inconsistent with decision stage: expected (type={current_type}, decision={current_decision}), got (type={final_response_dict['type']}, decision={final_response_dict['decision']})")
                    content = final_response_dict["content"].strip()
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Failed Parsing: {e}")
                    log_to_file(f"Failed Parsing: {e}\n\n",log_path=self.log_path)
                    retrieve_step += 1
                    continue

               
                if current_decision == "retrieve":
                    retrieve_step += 1
                    if not content:
                        print("Empty query, terminated!")
                        log_to_file("Empty query, terminated!\n",log_path=self.log_path)
                        break
                    evidence_output = evidence_retriever(query=content, evidence_texts=self.evidence_text)
                    retrieval_history.append((content, evidence_output))
                    print(f"Step {retrieve_step} Observations: {evidence_output}")
                    log_to_file(f"Step {retrieve_step} Observations: {evidence_output}\n\n",log_path=self.log_path)

                elif current_decision == "analyze":
                    final_analysis = content
                    break

                else:
                    print(f"Wrong decision value: {current_decision}，terminated!")
                    log_to_file(f"Wrong decision value: {current_decision}，terminated!\n\n",log_path=self.log_path)
                    break

            
            if final_analysis is None:
                final_analysis = "Insufficient information to complete the analysis (max retrieval steps reached or invalid decision/response)"

            return final_analysis
        else:
            pass

    def _run_micro(self, frontier):
       
        telemetry = self.telemetry
        if telemetry is None:
            return "Telemetry snapshots are missing for this case."

        max_retrieve_step = self.max_retrieval_steps
        retrieve_step = 0
        final_analysis = None
        retrieval_history = self.retrieval_history
        decision_result: Optional[dict] = None

        while retrieve_step < max_retrieve_step and final_analysis is None:
            tool_prompt = telemetry.build_tool_prompt()
            if retrieve_step >= max_retrieve_step - 1:
                tool_prompt += "\nYou are at the final retrieval step, try to conclude if information is enough."

            
            print(f"\nStep {retrieve_step + 1} - Decision Making")
            log_to_file(f"\n Step {retrieve_step + 1} - Decision Making", log_path=self.log_path)
            system_p1, user_p1 = construct_expert_analyze_prompt(
                dataset=self.dataset,
                expert_name=self.name,
                tool_prompt=tool_prompt,
                belief=self.graph.belief,
                history=retrieval_history,
                frontier=frontier,
                task_stage="decision_making",
            )
            response_p1, meta_p1 = llm_generate_response(
                user_prompt=user_p1,
                model_path=self.llm_args["model_path"],
                temperature=self.llm_args["temperature"],
                max_tokens=self.llm_args["max_tokens"],
                system_prompt=system_p1,
                return_meta=self.llm_args["return_meta"],
            )
            print(f"[] Step {retrieve_step + 1} Decision: {response_p1}")
            log_to_file(f"[] Step {retrieve_step + 1} Decision: {response_p1}\n\n", log_path=self.log_path)

            try:
                decision_result = parse_json_response(response_p1.strip())
                if not all(k in decision_result for k in ["type", "decision"]):
                    raise ValueError("Missing mandatory fields (type/decision) in decision response")
                if decision_result["decision"] not in ["retrieve", "tool_call", "analyze"]:
                    raise ValueError(f"Invalid decision value: {decision_result['decision']}")
                current_type = decision_result["type"]
                current_decision = decision_result["decision"]
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Decision parsing failed: {e}")
                log_to_file(f"Decision parsing failed: {e}\n\n", log_path=self.log_path)
                retrieve_step += 1
                continue

           
            print(f"[] Step {retrieve_step + 1} - Content Generation")
            log_to_file(f"[] Step {retrieve_step + 1} - Content Generation\n\n", log_path=self.log_path)
            system_p2, user_p2 = construct_expert_analyze_prompt(
                dataset=self.dataset,
                expert_name=self.name,
                tool_prompt=tool_prompt,
                belief=self.graph.belief,
                history=retrieval_history,
                frontier=frontier,
                task_stage="content_generation",
                decision_result=decision_result,
            )
            response_p2, meta_p2 = llm_generate_response(
                user_prompt=user_p2,
                model_path=self.llm_args["model_path"],
                temperature=self.llm_args["temperature"],
                max_tokens=self.llm_args["max_tokens"],
                system_prompt=system_p2,
                return_meta=self.llm_args["return_meta"],
            )
            print(f"[] Step {retrieve_step + 1} Content Generation: {response_p2}")
            log_to_file(f"[] Step {retrieve_step + 1} Content Generation: {response_p2}\n\n", log_path=self.log_path)

            try:
                final_response_dict = parse_json_response(response_p2.strip())
                if not all(k in final_response_dict for k in ["type", "decision", "content"]):
                    raise ValueError("Missing mandatory fields (type/decision/content) in final response")
                if (final_response_dict["type"] != current_type) or (final_response_dict["decision"] != current_decision):
                    raise ValueError(
                        f"Content stage result inconsistent with decision stage: expected (type={current_type}, decision={current_decision}), got (type={final_response_dict['type']}, decision={final_response_dict['decision']})"
                    )
                content = final_response_dict["content"].strip()
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed parsing: {e}")
                log_to_file(f"Failed parsing: {e}\n\n", log_path=self.log_path)
                retrieve_step += 1
                continue

            
            if current_decision in ["tool_call", "retrieve"]:
                retrieve_step += 1
                if not content:
                    print("Empty tool invokes")
                    log_to_file("Empty tool invokes\n", log_path=self.log_path)
                    break
                tool_output = telemetry.dispatch_tool(content)
                retrieval_history.append((content, tool_output))
                print(f"[] Step {retrieve_step} Tool observations: {tool_output}")
                log_to_file(f"[] Step {retrieve_step} Tool observations: {tool_output}\n\n", log_path=self.log_path)
                continue

            if current_decision == "analyze":
                final_analysis = content
                break

            break

        if final_analysis is None:
            final_analysis = "Insufficient information to complete the analysis (max retrieval steps reached or invalid decision/response)"

        return final_analysis
