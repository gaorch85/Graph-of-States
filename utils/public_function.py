import os
from datetime import datetime
from typing import Optional, Any, Dict, List, Set, Tuple
import math
import pickle
import re
from copy import deepcopy
import json
import http.client
from dataclasses import dataclass



def _strip_json_fence(text: str) -> str:
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fence_match:
        return fence_match.group(1).strip()
    if text.lower().startswith("json"):
        return text[4:].strip()
    return text


def _extract_complete_json_values(text: str) -> List[Any]:
    """Scan text and collect every parseable JSON object/array fragment."""
    decoder = json.JSONDecoder()
    values: List[Any] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\n\r,":
            idx += 1
        if idx >= len(text) or text[idx] not in "{[":
            idx += 1
            continue
        try:
            value, end = decoder.raw_decode(text, idx)
            values.append(value)
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return values


def _looks_truncated(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped[-1] in "}]":
        try:
            json.loads(stripped)
            return False
        except json.JSONDecodeError:
            pass
    return stripped.count("{") > stripped.count("}") or stripped.count("[") > stripped.count("]")


def _score_parsed(value: Any) -> Tuple[int, int, int]:
    if isinstance(value, dict):
        list_items = sum(len(v) for v in value.values() if isinstance(v, list))
        return (2, list_items, len(value))
    if isinstance(value, list):
        return (1, len(value), 0)
    return (0, 0, 0)


def _pick_best_parsed(candidates: List[Any]) -> Any:
    return max(candidates, key=_score_parsed)


def _salvage_truncated_json(text: str) -> Any:
    """Recover as much structure as possible from truncated or malformed JSON."""
    if not _looks_truncated(text):
        return None

    decoder = json.JSONDecoder()

    # Try appending common closing brackets for cut-off responses.
    for suffix in ("", "}", "]}", "\"]}", "\"}]}", "\"}", "\"]", "}]", "\"}]"):
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    # Recover {"key": [{...}, {...}, <truncated>]}
    array_match = re.search(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*\[', text)
    if array_match:
        key = array_match.group(1)
        items = _extract_complete_json_values(text[array_match.end():])
        if items:
            return {key: items}

    # Recover a top-level array with complete elements only.
    stripped = text.lstrip()
    if stripped.startswith("["):
        items = _extract_complete_json_values(stripped)
        if items:
            return items

    # Fall back to any complete JSON values found in the text.
    values = _extract_complete_json_values(text)
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return values

    # Last resort: parse the longest valid prefix starting at the first brace.
    start = text.find("{")
    if start != -1:
        for end in range(len(text), start, -1):
            chunk = text[start:end].rstrip().rstrip(",")
            if not chunk:
                continue
            try:
                return decoder.raw_decode(chunk)[0]
            except json.JSONDecodeError:
                continue

    return None


def parse_json_response(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        raise ValueError("Empty response to parse as JSON")

    text = _strip_json_fence(str(raw).strip())
    if not text:
        raise ValueError("Empty response to parse as JSON")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        candidates: List[Any] = []

        for idx, ch in enumerate(text):
            if ch not in ("{", "["):
                continue
            try:
                candidates.append(decoder.raw_decode(text, idx)[0])
            except json.JSONDecodeError:
                continue

        salvaged = _salvage_truncated_json(text)
        if salvaged is not None:
            candidates.append(salvaged)

        if candidates:
            return _pick_best_parsed(candidates)

        raise exc





@dataclass
class UsageStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    price_rmb: float = 0.0

    def update(self, meta: Dict[str, Any], price: float) -> None:
        self.calls += 1
        self.prompt_tokens += meta.get("prompt_tokens", 0)
        self.completion_tokens += meta.get("completion_tokens", 0)
        self.total_tokens += meta.get("total_tokens", 0)
        self.price_rmb += price


_usage_stats = UsageStats()


def _record_usage(meta: Dict[str, Any], price: float) -> None:
    _usage_stats.update(meta, price)
    print(
        "[LLM-API] Generation Completed | "
        f"Model：{meta.get('model', '')} | "
        f"Input token nums：{meta.get('prompt_tokens', 0)} | "
        f"Output token nums：{meta.get('completion_tokens', 0)} | "
        f"price(CNY)：{price:.8f}"
    )


def print_usage_summary() -> None:
    print(
        "[LLM-API] Total | "
        f"Invoke times：{_usage_stats.calls} | "
        f"Input token nums：{_usage_stats.prompt_tokens} | "
        f"Output token nums：{_usage_stats.completion_tokens} | "
        f"Total token nums：{_usage_stats.total_tokens} | "
        f"price(CNY)：{_usage_stats.price_rmb:.8f}"
    )

def get_usage_summary() -> None:
    return (
        "[LLM-API] Total | "
        f"Invoke times：{_usage_stats.calls} | "
        f"Input token nums：{_usage_stats.prompt_tokens} | "
        f"Output token nums：{_usage_stats.completion_tokens} | "
        f"Total token nums：{_usage_stats.total_tokens} | "
        f"price(CNY)：{_usage_stats.price_rmb:.8f}"
    )



def llm_generate_response(
    user_prompt: str,
    model_path: str,  
    temperature: float = 1.0,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = "",
    return_meta: bool = False,
    api_base: str = "api.chatanywhere.tech",  
    api_endpoint: str = "/v1/chat/completions",  
) -> Any:

    api_key = "Your-API-Key"
    if not api_key:
        raise ValueError("API key must not be empty")
    if not model_path:
        raise ValueError("model_path（Model name）Must not be empty！like'gpt-3.5-turbo'")

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})

    payload = json.dumps({
        "model": model_path,
        "messages": messages,
        "temperature": temperature,
        "stream": False  
    })

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15"
    }

    try:
        conn = http.client.HTTPSConnection(api_base, timeout=300)   
        conn.request("POST", api_endpoint, payload, headers)

        res = conn.getresponse()

        response_data = res.read().decode("utf-8")
        conn.close()


        response_json = json.loads(response_data)


        if "error" in response_json:
            raise RuntimeError(f"API call error：{response_json['error']['message']}")

    except json.JSONDecodeError as e:
        raise RuntimeError(f"API response parse error（Not JSON format）：{response_data}") from e
    except http.client.HTTPException as e:
        raise RuntimeError(f"API link error：{str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"API call error：{str(e)}") from e

    choice = response_json["choices"][0]
    # print("choice:", choice)
    if "message" not in choice or "content" not in choice["message"]:
        print("API return error", response_json)
        generated_text = ""
    else:
        generated_text = choice["message"]["content"].strip()

    meta = {
        "finish_reason": choice["finish_reason"],
        "model": model_path,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_tokens": response_json["usage"]["prompt_tokens"],
        "completion_tokens": response_json["usage"]["completion_tokens"],
        "total_tokens": response_json["usage"]["total_tokens"],
        "response": generated_text
    }

    if model_path == "gpt-5.1":
        price = 0.00875 * meta["prompt_tokens"] / 1000 + 0.07 * meta["completion_tokens"] / 1000
    elif model_path == "gpt-3.5-turbo":
        price = 0.0035 * meta["prompt_tokens"] / 1000 + 0.0105 * meta["completion_tokens"] / 1000
    elif model_path == "o3":
        price = 0.014 * meta["prompt_tokens"] / 1000 + 0.056 * meta["completion_tokens"] / 1000

    _record_usage(meta, price)

    if return_meta:
        return generated_text, meta
    return generated_text




def softmax_normalize(candidates):
    """
    candidates: List[Tuple[str, str|None, str, float]]
    return:     List[Tuple[str, str|None, str, float]] with last value normalized to probabilities
    """

    scores = [c[-1] for c in candidates]
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps) if exps else 1.0
    probs = [e / total for e in exps]

    out = []
    for (label, comp, why, _), p in zip(candidates, probs):
        out.append((label, comp, why, p))
    return out


def collect_lineage_subgraph(G, node_id: str) -> Dict[str, Any]:
    """
    return format：
    {
      "node_id": { ...full_node_payload, "id": node_id },
      "ancestors_via_refines": [ ...full_node_payload... ],  
      "evidence": {
        "support": [ {"src":..., "dst":..., ...full_edge_payload...}, ... ],
        "refute":  [ {"src":..., "dst":..., ...full_edge_payload...}, ... ]
      },
      "alarm_causal": {
        "nodes": { alarm_id: full_node_payload_with_id, ... },
        "edges": [ {"src":..., "dst":..., ...full_edge_payload...}, ... ]
      },
      "subgraph": {
        "nodes": { node_id: full_node_payload_with_id, ... },
        "edges": [ {"src":..., "dst":..., ...full_edge_payload...}, ... ]
      }
    }
    """


    def _coalesce(*values):
        for val in values:
            if isinstance(val, str):
                val = val.strip()
            if val not in (None, ""):
                return val
        return None

    def compact_alarm_attrs(raw: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}

        attrs_inner = raw.get("attrs")
        if isinstance(attrs_inner, dict):
            data = attrs_inner.get("data")
            if isinstance(data, dict):
                alarm_data = data
            else:
                alarm_data = attrs_inner
        else:
            alarm_data = {}

        labels = alarm_data.get("labels") if isinstance(alarm_data.get("labels"), dict) else {}

        info = {
            "alarm_id": raw.get("alarm_id"),
            "name": raw.get("name"),
            "message": raw.get("message"),
            "severity": raw.get("severity"),
            "alert_name": _coalesce(alarm_data.get("alertName"), labels.get("alertname"), raw.get("name")),
            "instance": _coalesce(alarm_data.get("instance"), labels.get("instance"), labels.get("hostname"), labels.get("host_id")),
            "metric": _coalesce(alarm_data.get("metricKey"), labels.get("metricKey"), labels.get("metrics_name_en"), labels.get("metrics_name_zh")),
            "value": _coalesce(labels.get("itemvalue"), alarm_data.get("value")),
            "unit": _coalesce(labels.get("metric_unit"), alarm_data.get("unit")),
            "startAt": _coalesce(alarm_data.get("startAt"), alarm_data.get("startsAt"), labels.get("startAt")),
            "triggerAt": _coalesce(alarm_data.get("triggerAt")),
            "description": _coalesce(labels.get("description"), labels.get("description_en"), alarm_data.get("description"), alarm_data.get("descriptionEn")),
        }
        return {k: v for k, v in info.items() if v not in (None, "")}

    def node_payload(nid: str) -> Dict[str, Any]:
        out = deepcopy(G.nodes[nid])   
        out["id"] = nid
        if out.get("type") == "Alarm":
            out["attrs"] = compact_alarm_attrs(out.get("attrs") or {})
        return out

    def edge_payload(src: str, dst: str) -> Dict[str, Any]:
        e = deepcopy(G.edges[(src, dst)])  
        e["src"] = src
        e["dst"] = dst
        return e

    def parents_of_refines(cur: str) -> List[str]:
        out = []
        for (src, dst), e in G.edges.items():
            if dst == cur and e.get("type") == "refines":
                out.append(src)
        return out

    lineage_nodes: Set[str] = {node_id}
    lineage_edges_refines: List[Tuple[str, str]] = []
    visited: Set[str] = set()
    stack: List[str] = [node_id]

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)

        for p in parents_of_refines(cur):
            lineage_nodes.add(p)
            lineage_edges_refines.append((p, cur))
            if G.nodes[p].get("type") != "Alarm":
                stack.append(p)

    support_edges, refute_edges = [], []
    evidence_nodes: Set[str] = set()
    for (src, dst), e in G.edges.items():
        if dst in lineage_nodes and e.get("type") in ("support", "refute"):
            rec = edge_payload(src, dst)
            evidence_nodes.add(src)
            (support_edges if e.get("type") == "support" else refute_edges).append(rec)


    alarm_nodes = {
        nid for nid in lineage_nodes if G.nodes.get(nid, {}).get("type") == "Alarm"
    }
    alarm_causal_edges: List[Dict[str, Any]] = []
    alarm_causal_node_ids: Set[str] = set()
    if alarm_nodes:
        for (src, dst), e in G.edges.items():
            if e.get("type") != "alarm_causal":
                continue
            if src in alarm_nodes or dst in alarm_nodes:
                alarm_causal_edges.append(edge_payload(src, dst))
                alarm_causal_node_ids.add(src)
                alarm_causal_node_ids.add(dst)

    alarm_causal_nodes = {
        nid: node_payload(nid)
        for nid in alarm_causal_node_ids
    }


    sub_nodes: Dict[str, Dict[str, Any]] = {}
    for nid in lineage_nodes.union(evidence_nodes):
        sub_nodes[nid] = node_payload(nid)

    sub_edges: List[Dict[str, Any]] = []

    for src, dst in lineage_edges_refines:
        sub_edges.append(edge_payload(src, dst))

    sub_edges.extend(support_edges)
    sub_edges.extend(refute_edges)


    lineage_json = {
        "node_id": node_payload(node_id),
        "ancestors_via_refines": [
            node_payload(nid) for nid in lineage_nodes if nid != node_id
        ],
        "evidence": {
            "support": support_edges,
            "refute": refute_edges
        },
        "alarm_causal": {
            "nodes": alarm_causal_nodes,
            "edges": alarm_causal_edges,
        },
        "subgraph": {
            "nodes": sub_nodes,
            "edges": sub_edges
        }
    }
    return lineage_json


def lineage_to_json_str(lineage: Dict[str, Any]) -> str:

    return json.dumps(lineage, ensure_ascii=False)


def save_pkl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def log_to_file(log_msg: str, log_path: str = "project_log.txt"):
    try:
        
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_log = f"[{time_str}] | {log_msg.strip()}\n"
        
        with open(log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(final_log)
    
    except Exception as e:
        print(f"❌ Log write failure：{str(e)}")
    


def evidence_retriever(query: str, evidence_texts) -> str:

    system_prompt = """
You are a precise evidence retriever with only one core task: 
First understand the information need behind the input query (whether it's a keyword or a complete sentence), then extract all relevant text fragments from the given evidence texts that answer or relate to this need, and return them exactly as they appear (no rephrasing, summarization, or additional explanation).

### Rules:
1. Output ONLY the matched evidence text(s) (concatenate multiple fragments with newline if needed) – no other words, punctuation, or comments;
2. If NO relevant content is found in the evidence texts for the query's information need, return an empty string ("");
3. Strictly preserve the original wording of the matched evidence texts (do not modify any characters);
4. Return all matching content (do not omit any relevant fragments);
5. Focus on the core information need of the query, not just literal keyword matching (e.g., "query about brain CT diagnosis" should match fragments with brain CT diagnostic results).

### Examples:
Example 1 (Query: Multiple matching fragments):
- Evidence texts: ["Sarah was at library 18:00-20:00", "Library is 10min from classroom", "David lost watch at 19:00"]
- Input query: "library"
- Output: "Sarah was at library 18:00-20:00\nLibrary is 10min from classroom"""

    user_prompt = f"""
Here are the evidence texts: {evidence_texts}.\n
Here is the input query: {query}.
"""
    response, meta = llm_generate_response(user_prompt=user_prompt, model_path="gpt-5.1", temperature=0, max_tokens=2048, system_prompt=system_prompt, return_meta=True)

    return response














import json
import os

def generate_causal_graph_html(graph_list, html_path):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    
    # 
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Causal Graph Visualization</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Font Awesome -->
    <link href="https://cdn.jsdelivr.net/npm/font-awesome@4.7.0/css/font-awesome.min.css" rel="stylesheet">
    <!-- D3.js -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#3b82f6',
                        secondary: '#64748b',
                        signal: '#ef4444',
                        evidence: '#22c55e',
                        hypothesis: '#f97316',
                        background: '#f8fafc'
                    },
                    fontFamily: {
                        sans: ['Inter', 'system-ui', 'sans-serif']
                    }
                }
            }
        }
    </script>
    
    <style type="text/tailwindcss">
        @layer utilities {
            .node-card {
                @apply p-3 rounded-lg shadow-md transition-all duration-300 cursor-pointer;
            }
            .node-card:hover {
                @apply shadow-lg transform -translate-y-1;
            }
            .signal-node {
                @apply bg-red-100 border-l-4 border-signal;
            }
            .evidence-node {
                @apply bg-green-100 border-l-4 border-evidence;
            }
            .hypothesis-node {
                @apply bg-orange-100 border-l-4 border-hypothesis;
            }
            .nav-button {
                @apply px-4 py-2 rounded-md bg-primary text-white hover:bg-blue-600 transition-colors duration-200;
            }
            .nav-button:disabled {
                @apply bg-gray-300 cursor-not-allowed;
            }
            .tooltip {
                @apply absolute bg-white p-2 rounded shadow-lg z-50 max-w-xs;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.2s;
            }
        }
    </style>
</head>
<body class="bg-background min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-6">
            <h1 class="text-3xl font-bold text-gray-800">Causal graph</h1>
            <p class="text-gray-600">Causal graph</p>
        </header>

        <div class="flex flex-col lg:flex-row gap-6">

            <div class="lg:w-1/4 bg-white p-4 rounded-lg shadow-md">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Navi</h2>
                <div class="flex justify-between items-center mb-6">
                    <button id="prevBtn" class="nav-button">
                        <i class="fa fa-arrow-left mr-2"></i>Before
                    </button>
                    <span id="counter" class="text-lg font-medium text-gray-700">1 / """ + str(len(graph_list)) + """</span>
                    <button id="nextBtn" class="nav-button">
                        Next<i class="fa fa-arrow-right ml-2"></i>
                    </button>
                </div>
                
                <div class="mb-6">
                    <h3 class="text-lg font-medium mb-2 text-gray-700">Legend</h3>
                    <div class="space-y-2">
                        <div class="flex items-center">
                            <div class="w-4 h-4 rounded-full bg-signal mr-2"></div>
                            <span>Signal (Signal)</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-4 h-4 rounded-full bg-evidence mr-2"></div>
                            <span>Evidence (Evidence)</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-4 h-4 rounded-full bg-hypothesis mr-2"></div>
                            <span>Hypothesis (Hypothesis)</span>
                        </div>
                    </div>
                </div>
                
                <div>
                    <h3 class="text-lg font-medium mb-2 text-gray-700">statis</h3>
                    <div id="nodeStats" class="text-sm text-gray-600">
                    </div>
                </div>
            </div>
            
            <div class="lg:w-3/4 bg-white p-4 rounded-lg shadow-md">
                <div class="flex justify-between items-center mb-4">
                    <h2 id="graphTitle" class="text-xl font-semibold text-gray-700">Causal graph 1</h2>
                    <div>
                        <button id="zoomInBtn" class="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 mr-2">
                            <i class="fa fa-search-plus"></i>
                        </button>
                        <button id="zoomOutBtn" class="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 mr-2">
                            <i class="fa fa-search-minus"></i>
                        </button>
                        <button id="resetZoomBtn" class="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300">
                            <i class="fa fa-refresh"></i>
                        </button>
                    </div>
                </div>
                

                <div id="graphContainer" class="border border-gray-200 rounded-lg h-[600px] overflow-hidden">
   
                </div>
                
       
                <div id="nodeDetails" class="mt-4 p-4 border border-gray-200 rounded-lg hidden">
                    <h3 class="text-lg font-medium mb-2 text-gray-700">Node</h3>
                    <div id="nodeDetailsContent">
                      
                    </div>
                </div>
            </div>
        </div>
    </div>
    

    <div id="tooltip" class="tooltip"></div>
    
    <script>
       
        const graphData = """ + json.dumps(graph_list) + """;
        let currentGraphIndex = 0;
        let svg, g, simulation, link, node;
        let width, height;
        let zoom;
        
       
        document.addEventListener('DOMContentLoaded', function() {
            initGraph();
            updateNavigation();
            updateNodeStats();
            setupEventListeners();
        });
        
       
        function initGraph() {
            const container = document.getElementById('graphContainer');
            width = container.clientWidth;
            height = container.clientHeight;
            
         
            svg = d3.select('#graphContainer')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
      
            zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', function(event) {
                    g.attr('transform', event.transform);
                });
            
            svg.call(zoom);
            
      
            g = svg.append('g');
            
         
            loadGraph(currentGraphIndex);
        }
        
      
        function loadGraph(index) {
            const graph = graphData[index];
            
       
            document.getElementById('graphTitle').textContent = 'Causal graph ' + (index + 1);
            
      
            g.selectAll('*').remove();
            
       
            const nodes = [];
            for (const nodeId in graph.nodes) {
                if (graph.nodes.hasOwnProperty(nodeId)) {
                    nodes.push({
                        id: nodeId,
                        ...graph.nodes[nodeId]
                    });
                }
            }
            
 
            let links = (graph.edges || []).map(function(e) {
                return {
                    source: e.source || e.src,
                    target: e.target || e.dst,
                    type: e.type,
                    attrs: e.attrs || {}
                };
            });
            
           
            links = links.filter(function(link) {
                const sourceExists = nodes.some(function(n) { return n.id === link.source; });
                const targetExists = nodes.some(function(n) { return n.id === link.target; });
                
                if (!sourceExists) {
                    console.warn('Source node not found:', link.source);
                }
                if (!targetExists) {
                    console.warn('Target node not found:', link.target);
                }
                
                return sourceExists && targetExists;
            });
            
         
            simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(function(d) { return d.id; }).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(70));
            
  
            link = g.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(links)
                .enter()
                .append('line')
                .attr('stroke-width', 2)
                .attr('stroke', function(d) { return getLinkColor(d.type); })
                .attr('stroke-opacity', 0.6);
            
  
            const nodeGroups = g.append('g')
                .attr('class', 'nodes')
                .selectAll('.node-group')
                .data(nodes)
                .enter()
                .append('g')
                .attr('class', 'node-group')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
            
    
            nodeGroups.append('circle')
                .attr('r', 20)
                .attr('fill', function(d) { return getNodeColor(d.type); })
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
            
        
            nodeGroups.append('text')
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'central')
                .attr('fill', '#fff')
                .attr('font-size', '12px')
                .text(function(d) { return getNodeIcon(d.type); });
     
            nodeGroups.append('text')
                .attr('dy', 40)
                .attr('text-anchor', 'middle')
                .attr('fill', '#333')
                .attr('font-size', '12px')
                .text(function(d) { return truncateText(d.label, 20); });
            

            nodeGroups.on('click', showNodeDetails);
            
  
            nodeGroups.on('mouseover', function(event, d) {
                const tooltip = d3.select('#tooltip');
                tooltip.transition()
                    .duration(200)
                    .style('opacity', .9);
                tooltip.html(
                    '<div class="font-medium">' + d.type + ': ' + truncateText(d.label, 50) + '</div>' +
                    (d.score !== undefined ? '<div>Score: ' + d.score + '</div>' : '')
                )
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', function() {
                d3.select('#tooltip').transition()
                    .duration(500)
                    .style('opacity', 0);
            });
            
     
            simulation.on('tick', function() {
                link
                    .attr('x1', function(d) { return d.source.x; })
                    .attr('y1', function(d) { return d.source.y; })
                    .attr('x2', function(d) { return d.target.x; })
                    .attr('y2', function(d) { return d.target.y; });
                
                nodeGroups
                    .attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });
            });
        }
        
 
        function getLinkColor(type) {
            switch(type) {
                case 'support': return '#22c55e'; 
                case 'refute': return '#ef4444'; 
                default: return '#999';
            }
        }

        function getNodeColor(type) {
            switch(type) {
                case 'Signal': return '#ef4444';
                case 'Evidence': return '#22c55e'; 
                case 'Hypothesis': return '#f97316'; 
                default: return '#64748b'; 
            }
        }
        

        function getNodeIcon(type) {
            switch(type) {
                case 'Signal': return 'S';
                case 'Evidence': return 'E';
                case 'Hypothesis': return 'H';
                default: return '?';
            }
        }
        

        function truncateText(text, maxLength) {
            if (!text) return '';
            return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
        }
        

        function showNodeDetails(event, d) {
            const detailsPanel = document.getElementById('nodeDetails');
            const detailsContent = document.getElementById('nodeDetailsContent');
            
       
            let html = '<div class="mb-2"><span class="font-medium">Type:</span><span class="px-2 py-1 rounded text-white" style="background-color: ' + getNodeColor(d.type) + '">' + d.type + '</span></div>';
            html += '<div class="mb-2"><span class="font-medium"Label:</span><p class="mt-1">' + d.label + '</p></div>';
            
          
            if (d.score !== undefined) {
                html += '<div class="mb-2"><span class="font-medium">Confidence:</span><span>' + d.score + '</span></div>';
            }
            
           
            if (d.attrs && Object.keys(d.attrs).length > 0) {
                html += '<div class="mb-2"><span class="font-medium">Attributes:</span><ul class="mt-1">';
                
                for (const [key, value] of Object.entries(d.attrs)) {
                    html += '<li><span class="font-medium">' + key + ':</span> ' + value + '</li>';
                }
                
                html += '</ul></div>';
            }
            
            detailsContent.innerHTML = html;
            detailsPanel.classList.remove('hidden');
        }
        

        function updateNavigation() {
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            const counter = document.getElementById('counter');
            
            prevBtn.disabled = currentGraphIndex === 0;
            nextBtn.disabled = currentGraphIndex === graphData.length - 1;
            counter.textContent = (currentGraphIndex + 1) + ' / ' + graphData.length;
        }
        

        function updateNodeStats() {
            const graph = graphData[currentGraphIndex];
            const nodes = graph.nodes;
            
            let signalCount = 0;
            let evidenceCount = 0;
            let hypothesisCount = 0;
            
            for (const nodeId in nodes) {
                const node = nodes[nodeId];
                switch(node.type) {
                    case 'Signal': signalCount++; break;
                    case 'Evidence': evidenceCount++; break;
                    case 'Hypothesis': hypothesisCount++; break;
                }
            }
            
            const statsHtml = '<div class="mb-1">Total nodes: ' + Object.keys(nodes).length + '</div>' +
                             '<div class="mb-1">Signal Node: ' + signalCount + '</div>' +
                             '<div class="mb-1">Evidence Node: ' + evidenceCount + '</div>' +
                             '<div>Hypothesis Node: ' + hypothesisCount + '</div>';
            
            document.getElementById('nodeStats').innerHTML = statsHtml;
        }
        
        function setupEventListeners() {

            document.getElementById('prevBtn').addEventListener('click', function() {
                if (currentGraphIndex > 0) {
                    currentGraphIndex--;
                    loadGraph(currentGraphIndex);
                    updateNavigation();
                    updateNodeStats();
                    document.getElementById('nodeDetails').classList.add('hidden');
                }
            });
            
            document.getElementById('nextBtn').addEventListener('click', function() {
                if (currentGraphIndex < graphData.length - 1) {
                    currentGraphIndex++;
                    loadGraph(currentGraphIndex);
                    updateNavigation();
                    updateNodeStats();
                    document.getElementById('nodeDetails').classList.add('hidden');
                }
            });
            
  
            document.getElementById('zoomInBtn').addEventListener('click', function() {
                svg.transition().call(zoom.scaleBy, 1.3);
            });
            
            document.getElementById('zoomOutBtn').addEventListener('click', function() {
                svg.transition().call(zoom.scaleBy, 0.7);
            });
            
            document.getElementById('resetZoomBtn').addEventListener('click', function() {
                svg.transition().call(zoom.transform, d3.zoomIdentity);
            });
       
            window.addEventListener('resize', function() {
                const container = document.getElementById('graphContainer');
                width = container.clientWidth;
                height = container.clientHeight;
                
                svg.attr('width', width).attr('height', height);
                simulation.force('center', d3.forceCenter(width / 2, height / 2));
                simulation.alpha(0.3).restart();
            });
        }
        
  
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
    </script>
</body>
</html>"""
    

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)