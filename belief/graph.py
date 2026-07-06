from typing import Dict, Optional, List, Tuple
import uuid
import json
import math
from utils.public_function import llm_generate_response, parse_json_response

class BeliefGraph:

    def __init__(self):
    
        self.nodes: Dict[str, Dict] = {}  # {node_id: {node_type, label, score, attrs}}
        self.edges: Dict[Tuple[str, str], Dict] = {}  # {(src_node_id, dst_node_id): {edge_type, conf, attrs}}
        self.start_signal_id: Optional[str] = None  
        self.belief = ""
        self.idx = 1
        self.level_nodes = {}

    def add_node(self, node_type: str, label: str, score: float = 1.0,  attrs: Optional[Dict] = None) -> str:
       
        valid_types = {"Signal", "Evidence", "Hypothesis"}
        if node_type not in valid_types:
            raise ValueError(f"Node type must be in {valid_types}，input type：{node_type}")
        
        node_id = "node-" + str(self.idx)
        self.idx += 1
        self.nodes[node_id] = {
            "type": node_type,
            "label": label,
            "score": score,  
            "attrs": attrs or {}
        }
        return node_id

    def get_node(self, node_id: str) -> Optional[Dict]:
     
        return self.nodes.get(node_id)

    def update_node(self, node_id: str, score: float, why: str):
      
        if node_id not in self.nodes:
            return
        self.nodes[node_id]["score"] = score
        self.nodes[node_id]["attrs"]["why"] = why
    
    def has_node(self, node_id):
        return node_id in self.nodes

  
    def add_edge(self, src: str, dst: str, edge_type: str, attrs: Optional[Dict] = None):
        valid_edges = {"support", "refute", "refines", "causal"}
        if edge_type not in valid_edges:
            print(f"Edge type must be in {valid_edges}, input type：{edge_type}")
            return

        if src not in self.nodes or dst not in self.nodes:
            raise ValueError("target node does not exist")
        self.edges[(src, dst)] = {
            "type": edge_type,
            "attrs": attrs or {}
        }

    def get_edge(self, src: str, dst: str) -> Optional[Dict]:
        return self.edges.get((src, dst))

    def update_edge_conf(self, src: str, dst: str, new_conf: float):
        if (src, dst) in self.edges:
            self.edges[(src, dst)]["conf"] = max(1e-6, min(1 - 1e-6, new_conf))

    def apply_evidence(self, evidence_id: str, hypo_id: str, relation: str, strength: float, provenance: Optional[Dict] = None):
        
        if relation == "support":
            self.link_support(evidence_id, hypo_id, strength, provenance)
        elif relation == "refute":
            self.link_refute(evidence_id, hypo_id, strength, provenance)
        else:
            raise ValueError("relation must be 'support' or 'refute'")

    def link_support(self, evidence_id: str, hypo_id: str,  provenance: Optional[Dict] = None):
    
        self.add_edge(evidence_id, hypo_id, "support", provenance)
        self.nodes[hypo_id]["attrs"]["has_evidence"] = True

    def link_refute(self, evidence_id: str, hypo_id: str, strength: float, provenance: Optional[Dict] = None):
      
        self.add_edge(evidence_id, hypo_id, "refute", provenance)
        self.nodes[hypo_id]["attrs"]["has_evidence"] = True

    def link_refines(self, src_id: str, dst_id: str, prob: float, base_prior: float = 1.0, attrs: Dict = None):
        src_type = self.nodes[src_id]["type"]
        self.add_edge(src_id, dst_id, "refines", prob, attrs)



    def link_causal(self, src_signal: str, dst_signal: str, conf: float, attrs: Optional[Dict] = None):
        self.add_edge(src_signal, dst_signal, "causal", attrs)

   
    def to_dict(self) -> Dict:
      
        edge_list = []
        for (src, dst), e in self.edges.items():
            edge_list.append({
                "src": src,
                "dst": dst,** e,
            })
        return {
            "nodes": self.nodes,
            "edges": edge_list,
            "start_signal_id": self.start_signal_id
        }

    def pretty_lines(self, max_label_len: int = 60, max_attrs_len: int = 80) -> List[str]:
       
        def _short(txt: str, n: int) -> str:
            s = str(txt)
            return s if len(s) <= n else (s[:n-1] + "…")

        def _node_label(n: dict) -> str:
            return str(n.get("label") or "-")

        def _attrs_str(obj: dict, n: int) -> str:
            if not obj:
                return ""
            try:
                js = json.dumps(obj, ensure_ascii=False)
            except Exception:
                js = str(obj)
            return _short(js, n)

        lines: List[str] = ["=== Nodes ==="]
        for nid, n in self.nodes.items():
            attrs = n.get("attrs", {}) or {}
            lvl = attrs.get("level", "-")  
            label = _short(_node_label(n), max_label_len)
            score = float(n.get("score", 0.0))
            attrs_part = _attrs_str(attrs, max_attrs_len)
            line = f"[{nid[:6]}] {n['type']}[{lvl}] | {label} | score={score:.3f}"
            if attrs_part:
                line += f" | attrs={attrs_part}"
            lines.append(line)

        lines.append("=== Edges ===")
        for (src, dst), e in self.edges.items():
            src_node = self.nodes.get(src, {})
            dst_node = self.nodes.get(dst, {})
            src_label = _short(_node_label(src_node), max_label_len)
            dst_label = _short(_node_label(dst_node), max_label_len)

            edge_type = e.get("type", "?")
            conf = float(e.get("conf", 0.0))
            e_attrs = e.get("attrs", {}) or {}
            attrs_part = _attrs_str(e_attrs, max_attrs_len)

            line = (
                f"{src[:6]}[{src_label}] -> {dst[:6]}[{dst_label}]"
                f" | {edge_type} | conf={conf:.3f}"
            )
            if attrs_part:
                line += f" | attrs={attrs_part}"
            lines.append(line)

        return lines

    def pretty_print(self, max_label_len: int = 60, max_attrs_len: int = 80) -> None:
      
        for line in self.pretty_lines(max_label_len=max_label_len, max_attrs_len=max_attrs_len):
            print(line)
  
    def generate_belief_text(self):
        # Concise system prompt: clarify task and output format
        system_prompt = """You need to generate a coherent text summarizing the current reasoning consensus based on the given inference graph nodes and edges.
Requirements:
1. Moderate length (50-200 words) with clear logic;
2. Output only JSON format with key "belief" (no extra content);
3. Focus on core relationships between Signals, Evidence, and Hypotheses, ignoring redundant details."""

        # User prompt: pass structured graph data (simplified format for LLM understanding)
        graph_data = self.to_dict()
        user_prompt = f"""Please generate the reasoning consensus text based on the following inference graph information:
{json.dumps(graph_data, ensure_ascii=False, indent=2)}

Example Output:
{{"belief": "The patient has a fever, and laboratory tests show an elevated white blood cell count. This evidence strongly supports the conclusion that the patient has a bacterial infection. Additionally, fever is causally linked to fatigue, and the bacterial infection can be further specified as an infection caused by Streptococcus pneumoniae."}}"""
        response, meta = llm_generate_response(user_prompt=user_prompt, model_path="deepseek-v4-pro", temperature=0.5, max_tokens=2048, system_prompt=system_prompt, return_meta=True)
        print(f"Response: {response}")
        response_dict = parse_json_response(response)
        self.belief = response_dict["belief"]

   
    def update_level_nodes(self):
        self.level_nodes = {}
        for node_id, node_info in self.nodes.items():
           
            if node_info["type"] != "Hypothesis" or "level" not in node_info["attrs"]:
                continue
            level = node_info["attrs"]["level"]
            confidence = node_info.get("score", 0.0)
            
            if level not in self.level_nodes:
                self.level_nodes[level] = []
            self.level_nodes[level].append((node_id, confidence))
        
      
        for level in self.level_nodes:
            self.level_nodes[level].sort(key=lambda x: x[1], reverse=True)
    
  
    def delete_node(self, node_id):
       
        if node_id in self.nodes:
            del self.nodes[node_id]
       
        self.edges = {
            (src, dst): attrs 
            for (src, dst), attrs in self.edges.items()  
            if src != node_id and dst != node_id  
        }
      
        self.update_level_nodes()

       
    def delete_level_nodes(self, target_level):
       
        level_node_ids = []
        if target_level in self.level_nodes:
            level_node_ids = [node_id for node_id, _ in self.level_nodes[target_level]]
        
      
        for node_id in level_node_ids:
            self.delete_node(node_id)
        
       
        self.update_level_nodes()

   
    def is_highest_conf_in_level(self, node_id):
        if node_id not in self.nodes:
            return False
        node_info = self.nodes[node_id]
        if node_info["type"] != "Hypothesis" or "level" not in node_info["attrs"]:
            return False
        
        level = node_info["attrs"]["level"]
        if level not in self.level_nodes or len(self.level_nodes[level]) == 0:
            return False
        
       
        highest_node_id = self.level_nodes[level][0][0]
        return highest_node_id == node_id
