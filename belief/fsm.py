from typing import Dict, List, Tuple, Optional, Union

class BeliefFSM:
    def __init__(self, thresholds: Optional[Dict] = None):

        default_thresholds = {
            "gap_delta": 0.15,    
            "min_support": 1,     
            "max_steps": 5       
        }
        self.thresholds = {**default_thresholds, **(thresholds or {})}

       
        self.state: Union[int, str] = 1  
        self.history: List[Union[int, str]] = [self.state]  
        self._stage_steps: Dict[int, int] = {self.state: 0}  

   
    def get_state(self) -> Union[int, str]:
    
        return self.state

    def is_final_state(self) -> bool:
     
        return self.state == "report"

    def reset(self) -> None:
     
        self.state = 1
        self.history = [self.state]
        self._stage_steps = {self.state: 0}

    def set_state(self, new_state: Union[int, str]) -> None:
        if isinstance(new_state, int):
            if new_state < 1:
                raise ValueError(f"state must ≥1，Input: {new_state}")

            if new_state not in self._stage_steps:
                self._stage_steps[new_state] = 0
        elif new_state != "report":
            raise ValueError(f"state must be postive integer or 'report'，Input：{new_state}")
        
        self.state = new_state
        self.history.append(new_state)

    def tick_step(self, k: int = 1) -> None:

        if k < 1 or not isinstance(k, int):
            raise ValueError(f"increment must be positive integer, Input：{k}")
        
        cur_state = self.state
        if cur_state not in self._stage_steps:
            self._stage_steps[cur_state] = 0
        self._stage_steps[cur_state] += k

    def maybe_transit(self, G):
        advanced = self._advance_state(G)
        return advanced




    def _advance_state(self, G) -> bool:
        cur_state = self.state
        cur_candidates = self._top_hypos(G, level=cur_state)
        if not cur_candidates:
            self.set_state("report")
            return True


        top1_id, top1_score = cur_candidates[0]


        cur_steps = self._stage_steps.get(cur_state, 0)
        gap_th = self.thresholds.get("gap_delta", 0.0)
        sup_th = self.thresholds.get("min_support", 0)
        max_steps = self.thresholds.get("max_steps", 10**9)


        gap, _ = self._gap_and_top1(cur_candidates)
        sup_cnt = self._count_support_edges(G, top1_id)


        if gap >= gap_th and sup_cnt >= sup_th:
            return True


        if cur_steps >= max_steps:
            return True

        return False


    def _top_hypos(self, G, level: int, k: Optional[int] = None) -> List[Tuple[str, float]]:

        candidates: List[Tuple[str, float]] = []
        for nid, n in G.nodes.items():
            if n.get("type") != "Hypothesis":
                continue
            node_level = int(n.get("attrs", {}).get("level", -1))
            if node_level != level:
                continue
            score = float(n.get("score", 0.0))
            candidates.append((nid, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates if k is None else candidates[:k]

    def _gap_and_top1(self, candidates: List[Tuple[str, float]]) -> Tuple[float, Tuple[str, float]]:
  
        if not candidates:
            return 0.0, ("", 0.0)
        top1_id, top1_score = candidates[0]
        if len(candidates) == 1:
            return top1_score, (top1_id, top1_score)
        top2_score = candidates[1][1]
        gap = top1_score - top2_score
        return gap, (top1_id, top1_score)

    def _count_support_edges(self, G, hypo_id: str) -> int:
        support_count = 0
        for (src_id, dst_id), edge in G.edges.items():
            if dst_id != hypo_id:
                continue
            if edge.get("type") != "support":
                continue
            src_node = G.nodes.get(src_id, {})
            if src_node.get("type") == "Evidence":
                support_count += 1
        return support_count