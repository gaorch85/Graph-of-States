import json
from prompts.prompt_generation import *
from utils.public_function import *
from pathlib import Path
from datetime import datetime

class Central_Agent:
   
    def __init__(self, name, graph, fsm, dataset, log_path, case_id, case_symptom, llm_args, cfg_domain, experts, evidence_text):

        self.name = name
        self.graph = graph
        self.fsm = fsm
        self.dataset = dataset
        self.log_path = log_path
        self.case_id = case_id
        self.case_symptom = case_symptom
        self.llm_args =llm_args
        self.session_max_steps = llm_args["session_max_steps"]
        self.cfg_domain = cfg_domain
        self.experts = experts
        self.evidence_text = evidence_text
        self.graphs = []
    


    def ingest(self):

        total_input_tokens = 0
        total_output_tokens = 0


     
        system_prompt, user_prompt = construct_ingest_prompt_basic_node(dataset=self.dataset, expert_name=self.name, symptom=self.case_symptom)
        response, meta = llm_generate_response(user_prompt=user_prompt, model_path=self.llm_args["model_path"], temperature=self.llm_args["temperature"], max_tokens=self.llm_args["max_tokens"], system_prompt=system_prompt, return_meta=self.llm_args["return_meta"])
        print(f"Response: {response}")
        log_to_file(f"#### Here are the reponse of ingest function -- generating basic symptom and evidence nodes:\n\n {response}\n\n",log_path=self.log_path)

        response_dict = parse_json_response(response)
        symptom_node = response_dict["symptom_node"]
        isolated_evidence = response_dict["isolated_evidence"]
        total_input_tokens += meta["prompt_tokens"]
        total_output_tokens += meta["completion_tokens"]
        self.symptom_node_id = self.graph.add_node(node_type="Signal", label=symptom_node)
        self.isolated_evidence_ids = []
        for evidence in isolated_evidence:
            isolated_evidence_id = self.graph.add_node(node_type="Evidence", label=evidence)
            self.isolated_evidence_ids.append(isolated_evidence_id)
        
        self.graphs.append(self.graph.to_dict())

        self.graph.pretty_print()
        
    
        system_prompt, user_prompt = construct_ingest_prompt_L1_Hypo(dataset=self.dataset, expert_name=self.name, graph=self.graph, symptom_node_id=self.symptom_node_id, isolated_evidence_ids=self.isolated_evidence_ids)
        response, meta = llm_generate_response(user_prompt=user_prompt, model_path=self.llm_args["model_path"], temperature=self.llm_args["temperature"], max_tokens=self.llm_args["max_tokens"], system_prompt=system_prompt, return_meta=self.llm_args["return_meta"])
        print(f"Response: {response}")
        log_to_file(f"#### Here are the reponse of ingest function -- generating L1 hypos and corresponding relationship between evidence and hypos:\n\n {response}\n\n",log_path=self.log_path)
        
        response_dict = parse_json_response(response)
        candidates = response_dict["candidates"]
        edges = response_dict["edges"]     

        total_input_tokens += meta["prompt_tokens"]
        total_output_tokens += meta["completion_tokens"]   
        
     
        hypo_id2real_id = {}  
        for hypo in candidates:
            real_node_id = self.graph.add_node(
                node_type="Hypothesis",
                label=hypo["label"],
                score=hypo["confidence"],
                attrs={"why": hypo["why"], "level": 1}
            )
            
            hypo_id2real_id[hypo["id"]] = real_node_id
        
       
        for hypo_id in hypo_id2real_id.values():
            self.graph.add_edge(
                src=self.symptom_node_id,         
                dst=hypo_id,         
                edge_type="refines",       
            )

    
        for edge in edges:
            evidence_id = edge["src"]  
            original_dst = edge["dst"]  
            relation = edge["relation"]  
            real_dst_id = hypo_id2real_id[original_dst]
            self.graph.add_edge(
                src=evidence_id,          
                dst=real_dst_id,        
                edge_type=relation,        
            )
        
        self.graph.update_level_nodes()
        self.graphs.append(self.graph.to_dict())
        self.graph.pretty_print()


    
    def run(self):
    

        step = 0
        self.graph.generate_belief_text()
      
        while True:
            
            step += 1
            self.fsm.tick_step(1)

            
            self.frontier= self.extract_frontier()
            print(self.frontier)
            log_to_file(f"#### Step{step}: Finding the current frontier \n\n {self.frontier}\n",log_path=self.log_path)

            
            plan = self.plan()
            
            analyses = []
            if plan != []:
                analyses = self.act(plan)

            
            system_prompt, user_prompt = construct_generate_proposal_prompt(dataset=self.dataset, expert_name=self.name, analyses=analyses, graph_description=self.graph.to_dict(), belief=self.graph.belief)
            response, meta = llm_generate_response(user_prompt=user_prompt, model_path=self.llm_args["model_path"], temperature=self.llm_args["temperature"], max_tokens=self.llm_args["max_tokens"], system_prompt=system_prompt, return_meta=self.llm_args["return_meta"])
            print(f"Response: {response}")
            log_to_file(f"#### Step{step}: Central agent decided to update the graph after receiving the analyses from expert agent\n\n {response}",log_path=self.log_path)
           
    
            response_dict = parse_json_response(response)
            edits = response_dict.get("edit", [])  
            nodes = response_dict["nodes"]
            edges = response_dict["edges"]

           
           
            for edit_item in edits:
           
                target_node_id = edit_item["node_id"]  
                new_confidence = edit_item["confidence"]
                new_why = edit_item["why"]
                
                
                if not self.graph.has_node(target_node_id):
                  
                    continue
                
             
                self.graph.update_node(
                    node_id=target_node_id,
                    score=new_confidence, 
                    why={"why": new_why} 
                )




           
            temp_id2real_id = {} 
            for node in nodes:
                temp_node_id = node["id"]  
                node_type = node["node_type"]  # Evidence/Hypothesis
                node_label = node["label"]

                if node_type == "Evidence":
                    real_node_id = self.graph.add_node(
                        node_type=node_type,
                        label=node_label,
                    )
                elif node_type == "Hypothesis":
                   
                    node_confidence = node["confidence"]
                    node_why = node["why"]
                    real_node_id = self.graph.add_node(
                        node_type=node_type,
                        label=node_label,
                        score=node_confidence,
                        attrs={"why": node_why, "level": self.frontier["level"]}  
                    )

                    self.graph.add_edge(
                        src=self.symptom_node_id,        
                        dst=real_node_id,         
                        edge_type="refines",     
                    )
                else:
                    continue
                

               
                temp_id2real_id[temp_node_id] = real_node_id
                
            self.graphs.append(self.graph.to_dict())

    
            for edge in edges:
                src_temp_id = edge["src"]
                dst_temp_id = edge["dst"]
                relation = edge["relation"]
                
               
                src_real_id = temp_id2real_id.get(src_temp_id, src_temp_id)
               
                dst_real_id = temp_id2real_id.get(dst_temp_id, dst_temp_id)
                
              
                self.graph.add_edge(
                    src=src_real_id,
                    dst=dst_real_id,
                    edge_type=relation 
                )
            
            self.graph.update_level_nodes()
            
          
            self.backtrack_frontier_chain()

          


           
            self.frontier= self.extract_frontier()
            print(self.frontier)

            log_to_file(f"#### Step{step}: The new frontier after the central agent updating the graph \n\n {self.frontier}",log_path=self.log_path)
            
            self.graph.pretty_print()
            self.graphs.append(self.graph.to_dict())
            self.graph.generate_belief_text()

           
            advance = self.fsm.maybe_transit(G = self.graph)
            if step >= self.session_max_steps:
                advance = True

          
            if advance == True:
             
                Report_Flag = "False"
                if step >= self.session_max_steps:
                    Report_Flag = "True"
                system_prompt, user_prompt = construct_report_or_refine_prompt(dataset=self.dataset, expert_name=self.name, Report_Flag=Report_Flag,  graph_description=self.graph.to_dict(), belief=self.graph.belief, frontier=self.frontier)
                response, meta = llm_generate_response(user_prompt=user_prompt, model_path=self.llm_args["model_path"], temperature=self.llm_args["temperature"], max_tokens=self.llm_args["max_tokens"], system_prompt=system_prompt, return_meta=self.llm_args["return_meta"])
                print(f"Response: {response}")
                log_to_file(f"#### Step{step}: Central agent decide to report or dive deeper\n\n {response}\n\n",log_path=self.log_path)

                response_dict = parse_json_response(response)
                type = response_dict["type"]

                if type == 1:
                    answer = response_dict["answer"]
                    report = response_dict["report"]
                    return answer, report
                else:
                   
                    candidates = response_dict["candidates"]
                    self.frontier_id = self.frontier["node_id"]
                    candidate_ids = []
                    for candidate in candidates:
                        candidate_id = self.graph.add_node(node_type="Hypothesis", label=candidate["label"],score = candidate["confidence"], attrs={"why": candidate["why"], "level": self.frontier["level"] + 1})
                        candidate_ids.append(candidate_id)
                    for candidate_id in candidate_ids:
                        self.graph.add_edge(self.frontier_id, candidate_id, edge_type="refines")
                    self.fsm.set_state(self.fsm.state + 1)

                    self.graphs.append(self.graph.to_dict())
            
           


    def extract_frontier(self):
        """

        Example:
            {'node_id': '1400679e-a0c3-4588-8096-a55ff194b007', 'label': 'Benign adnexal/appendageal tumor', 'why': 'Congenital, slow-growing alopecic scalp plaque with waxy exophytic nodule and yellow dermoscopic hue without vessels favors benign adnexal tumor (e.g., nevus sebaceus).', 'score': 0.7, 'level': 1, 'supports': 5, 'refutes': 0}
        """
        G = self.graph
        if G is None or not getattr(G, "nodes", None):
            return []

        
        cur_level = None
        if self.fsm is not None:
            try:
                cur_level = self.fsm.get_state()
            except Exception:
                cur_level = None

        hypos: List[Tuple[str, Dict[str, Any]]] = []
        for nid, n in G.nodes.items():
            if n.get("type") != "Hypothesis":
                continue
            lvl = n.get("attrs", {}).get("level")
            if cur_level is not None and lvl != cur_level:
                continue
            hypos.append((nid, n))

        if not hypos:
            return []

        hypos.sort(key=lambda x: float(x[1].get("score", 0.0)), reverse=True)
        # topk = int(self.cfg.get("frontier_k", 3))
        picked = hypos[:1]

       
        frontier: Dict[str, Any] = {}
        for nid, n in picked:
            supports = 0
            refutes = 0
            for (src, dst), e in getattr(G, "edges", {}).items():
                if dst == nid and e.get("type") == "support":
                    supports += 1
                if dst == nid and e.get("type") == "refute":
                    refutes += 1
            frontier = {
                    "node_id": nid,
                    "label": n.get("label"),
                    "why": n.get("attrs").get("why"),
                    "score": float(n.get("score", 0.0)),
                    "level": n.get("attrs", {}).get("level"),
                    "supports": supports,
                    "refutes": refutes,
                }
            
        return frontier
    

    def plan(self):
    
        frontier = self.frontier
        expert_descriptions = self.cfg_domain.get("expert_descriptions", {})

        system_prompt, user_prompt = construct_call_expert_prompt(dataset=self.dataset, expert_name=self.name, frontier=frontier, expert_descriptions=expert_descriptions)
        response, meta = llm_generate_response(user_prompt=user_prompt, model_path=self.llm_args["model_path"], temperature=self.llm_args["temperature"], max_tokens=self.llm_args["max_tokens"], system_prompt=system_prompt, return_meta=self.llm_args["return_meta"])
        print(f"Response: {response}")
        log_to_file(f"#### Here are the plan of central expert: \n\n {response}\n\n",log_path=self.log_path)


        response_dict = parse_json_response(response)
        plan = response_dict
        return plan

    def act(self, plan):
       

        analyses = []

    

        for item in plan:
            expert_name = item["expert_name"]

          
            matching_experts = [agent for agent in self.experts if agent.name == expert_name]
            if not matching_experts:
                raise ValueError(f"Not founded, current list：{[a.name for a in self.experts]}")
            expert = matching_experts[0]

      
            analysis = expert.run(self.frontier)

     
            analyses.append({"expert_name": expert_name, "analysis": analysis})


        return analyses

    def backtrack_frontier_chain(self):
      
        graph = self.graph
        frontier = self.frontier
      
        current_frontier_id = frontier["node_id"]  
        current_level = frontier["level"]         
        
        
       
        if not graph.has_node(current_frontier_id):
            return
        node_info = graph.nodes[current_frontier_id]
        if node_info["type"] != "Hypothesis":
            return

      
        level_chain = list(range(1, current_level + 1))
        level_to_node_id = {}
        def find_ancestor_node(level, current_node_id):
           
            if level == current_level:
                return current_node_id
           
            parent_edges = [
            {
                "src": src,
                "dst": dst,
                "type": attrs["type"],
                "attrs": attrs["attrs"]
            }
            for (src, dst), attrs in graph.edges.items()  
            if dst == current_node_id 
        ]
            if not parent_edges:
                return None
            parent_id = parent_edges[0]["src"]
          
            parent_level = graph.nodes[parent_id]["attrs"]["level"] if parent_id in graph.nodes else -1
            if parent_level == level:
                return parent_id
            return find_ancestor_node(level, parent_id)
       
        for level in level_chain:
            node_id = find_ancestor_node(level, current_frontier_id)
            if node_id:
                level_to_node_id[level] = node_id

       
        invalid_start_level = None 
        for level in level_chain:
            if level not in level_to_node_id:
                continue
            node_id = level_to_node_id[level]
            
            if not graph.is_highest_conf_in_level(node_id):
                invalid_start_level = level
                break  

       
        if invalid_start_level is not None:
            print(f"Level: {invalid_start_level} Node: {level_to_node_id[invalid_start_level]} not the hypothesis with highest confidence, backtracking invoked")
            log_to_file(f"Level: {invalid_start_level} Node: {level_to_node_id[invalid_start_level]} not the hypothesis with highest confidence, backtracking invoked\n\n",log_path=self.log_path)
            self.fsm.set_state(invalid_start_level)
           
            for delete_level in range(invalid_start_level + 1, current_level + 1):
                graph.delete_level_nodes(delete_level)
                print(f"Level {delete_level} deteled")
                log_to_file(f"Level {delete_level} deteled\n\n",log_path=self.log_path)

    
