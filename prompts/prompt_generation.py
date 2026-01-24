import os
import importlib.util


def construct_ingest_prompt_basic_node(dataset, expert_name, symptom):

    system_prompt_path = os.path.join("prompts", dataset, "system_prompt.py")
    spec = importlib.util.spec_from_file_location(f"{expert_name}_system_prompt", system_prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    system_prompt = getattr(prompt_module, f"{expert_name}_system_prompt")

    task_prompt = """
- Goal: 
  In general abductive reasoning scenarios, extract 1 core "symptom node" (surface phenomenon directly perceivable by users/external observers, no subjective reasoning) and multiple "isolated evidence nodes" (objective factual details from text supporting the symptom) from symptom text, then output structured JSON.

- Constraints:
  1. Only 1 symptom node: concise summary of core surface phenomenon, no inference/analytical info;
  2. Isolated evidence: objective facts directly extractable from text, no subjective assumptions/reasoning;
  3. Output only valid JSON (no extra text/comments/format errors), strictly follow the specified format;
  4. Compatible with multiple scenarios (microservice root cause analysis, medical diagnosis, criminal investigation, etc.);
  5. Isolated evidence retains key info (time/location/person/value/event), no core detail omission;
  6. Return empty array for "isolated_evidence" if no valid evidence exists;
  7. Isolated evidence nodes shall be **reasonably aggregated** to avoid over-fragmentation (merge related/contextually linked factual details into a single evidence node), with the total number in (1, 5);

- Instructions:
  1. Read symptom text thoroughly, first identify "core phenomenon directly perceivable by users/external observers", summarize as 1 symptom node;
  2. Extract all objective facts supporting the symptom from text, then **aggregate related factual details into cohesive isolated evidence nodes** (avoid splitting trivial details into separate nodes); ensure the total number of isolated evidence nodes in (1, 5);
  3. Output JSON with strictly matched field names, no redundant content;
  4. Ensure JSON is parsable (no syntax errors/line breaks/special character interference).

- Examples:
  Example 1 (Criminal Investigation):
  Input Symptom Text: David's watch went missing on July 18 morning; on July 17 evening, David left his watch on the classroom desk, no one was in the classroom when he went out, but he saw Sarah and Lin enter the classroom.
  Output JSON:
  {
    "symptom_node": "David's watch was lost on July 18 morning",
    "isolated_evidence": [
      "On July 17 evening, David placed his watch on the classroom desk, no one was in the classroom when he went out, and he saw Sarah and Lin enter the classroom"
    ]
  }

  Example 2 (Microservice Root Cause Analysis):
  Input Symptom Text: emailservice0 service alerted continuously since 9:05 (response timeout); Host 122.10.212.112 (Pod location) memory usage >90% since 8:50, CPU usage stable at 40%.
  Output JSON:
  {
    "symptom_node": "emailservice0 showed response timeout alerts at 9:05",
    "isolated_evidence": [
      "Host 122.10.212.112 (Pod location) had memory usage >90% since 8:50 and CPU usage stable at 40%"
    ]
  }

  Example 3 (Medical Diagnosis):
  Input Symptom Text: Patient had fever for 3 days (max 39.2°C); physical exam: pharyngeal congestion, grade Ⅱ tonsillar enlargement; blood test: WBC 12×10^9/L.
  Output JSON:
  {
    "symptom_node": "Patient had continuous fever for 3 days (max 39.2°C)",
    "isolated_evidence": [
      "Patient's physical exam showed pharyngeal congestion and grade Ⅱ tonsillar enlargement; blood test indicated WBC count 12×10^9/L"
    ]
  }

  Example 4 (No Isolated Evidence):
  Input Symptom Text: User reported mobile phone cannot turn on normally.
  Output JSON:
  {
    "symptom_node": "User's mobile phone cannot turn on normally",
    "isolated_evidence": []
  }
  }
"""

    system_prompt = system_prompt + task_prompt
    
    user_prompt = f"""
Below are the symptoms text, please follow the above instructions to \n
{symptom}
"""

    return system_prompt, user_prompt


def construct_ingest_prompt_L1_Hypo(dataset, expert_name, graph, symptom_node_id, isolated_evidence_ids):
    """
        Generate L1 Hypothesiss based on symptom node and isolated evidence

        Return:
            system_prompt, user_prompt
    """
    system_prompt_path = os.path.join("prompts", dataset, "system_prompt.py")
    spec = importlib.util.spec_from_file_location(f"{expert_name}_system_prompt", system_prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    system_prompt = getattr(prompt_module, f"{expert_name}_system_prompt")

    task_prompt = """
- Goal: 
  Based on domain knowledge, 1 core symptom node and isolated evidence (ID + label), generate coarse-grained candidate hypotheses; create edges for evidence to clarify support/refute relationships with candidates, output structured JSON. Core rule: All evidence must attach to at least one candidate (top-level coarse-grained inferences only).

- Constraints:
  1. Coarse-grained candidates (2-4): Focus on "category/core direction" (no fine-grained conclusions);
  2. Candidate fields:
     - "id": Unique short ID (e.g., h001), "label": ≤30 chars description, 
     - "confidence": 0.0-1.0 float (hypothesis confidence),
     - "why": ≤150 chars summary of evidence support/refute logic for this candidate;
  3. Edge fields:
     - "relation": Only "support"/"refute";
     - "src": Evidence ID (e.g., ev001), "dst": Candidate ID (e.g., h001);
  4. Output only valid JSON (strict format, no extra text/errors);
  5. Candidate "label" concise (≤30 chars, no redundancy).

- Instructions:
  1. Clarify symptom + evidence core info, derive coarse-grained candidates (add unique ID e.g., h001);
  2. For each evidence, define support/refute relationships with candidates (use ID in edges);
  3. In candidate "why", summarize logic of all associated evidence for this candidate;
  4. Verify JSON validity (no syntax errors).

- Examples:
  Example 1 (Criminal Investigation):
  Input:
  - Symptom: David's watch lost on July 18 morning
  - Evidence:
    ev001: David placed watch on classroom desk (July 17 evening)
    ev002: No one in classroom when David left (July 17 evening)
    ev003: David saw Sarah/Lin enter classroom (July 17 evening)
  Output JSON:
  {
    "candidates": [
      {"id":"h001","label":"Watch taken by classroom-related personnel","confidence":0.7,"why":"ev001 confirms watch was in classroom (accessible to insiders); ev003 shows Sarah/Lin (classroom personnel) entered, supporting this hypothesis"},
      {"id":"h002","label":"Watch stolen by external personnel","confidence":0.2,"why":"ev002 supports (empty classroom) but ev003 refutes (insiders present)"},
      {"id":"h003","label":"David misplaced the watch","confidence":0.1,"why":"No direct evidence supports/refutes this hypothesis"}
    ],
    "edges": [
      {"src":"ev001","dst":"h001","relation":"support"},
      {"src":"ev002","dst":"h002","relation":"support"},
      {"src":"ev003","dst":"h002","relation":"refute"},
      {"src":"ev003","dst":"h001","relation":"support"}
    ]
  }

  Example 2 (Microservice Root Cause Analysis):
  Input:
  - Symptom: emailservice0 timeout alerts at 9:05
  - Evidence:
    ev004: Host 122.10.212.112 memory >90% (8:50+)
    ev005: Host 122.10.212.112 CPU 40% (stable)
  Output JSON:
  {
    "candidates": [
      {"id":"h001","label":"Service timeout from insufficient resources","confidence":0.85,"why":"ev004 supports (high memory usage causes resource contention); ev005 refutes (normal CPU usage)"},
      {"id":"h002","label":"Service timeout from abnormal code","confidence":0.1,"why":"ev005 refutes (no CPU anomalies indicating code issues)"},
      {"id":"h003","label":"Service timeout from network latency","confidence":0.05,"why":"No evidence supports/refutes this hypothesis"}
    ],
    "edges": [
      {"src":"ev004","dst":"h001","relation":"support"},
      {"src":"ev005","dst":"h001","relation":"refute"},
      {"src":"ev005","dst":"h002","relation":"refute"}
    ]
  }
"""

    system_prompt = system_prompt + task_prompt

    evidence_text = ""
    for evidence_id in isolated_evidence_ids:
      evidence_text = evidence_text + f"ID: {evidence_id}. " + f"Label: {graph.nodes[evidence_id]["label"]}.\n"
    user_prompt = f"""
Here is the symptom node: {graph.nodes[symptom_node_id]["label"]}
Below are the isolated evidence nodes with corresponding IDs:
{evidence_text}
"""
    
    # print(system_prompt)
    # print('\n\n\n\n')
    # print(user_prompt)

    return system_prompt, user_prompt


def construct_call_expert_prompt(dataset, expert_name, frontier, expert_descriptions):


    system_prompt_path = os.path.join("prompts", dataset, "system_prompt.py")
    spec = importlib.util.spec_from_file_location(f"{expert_name}_system_prompt", system_prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    system_prompt = getattr(prompt_module, f"{expert_name}_system_prompt")



    task_prompt = """
- Goal: 
  Based on the core issue of the current frontier hypothesis, combined with the responsibility descriptions of each agent, determine whether to call agents; if needed, list the specific agents to be called, and finally output a structured JSON list (return empty list if no call is required).

- Constraints:
  1. Output only JSON format, the JSON is a list where each element is {"expert_name": "Agent Name"};
  2. Only call agents strongly related to the core issue of the frontier, with the number controlled at 1-3 (avoid redundant calls);
  3. Must focus on the to-be-verified/analyzed points of the frontier and not call irrelevant agents;
  4. Do not output empty JSON, you MUST based on current information to call corresponding agent (at least one);

- Instructions:
  1. First extract the core to-be-analyzed/verified issue of the frontier (e.g., "Whether the scalp mass is a malignant tumor", "Whether service timeout is caused by network issues");
  2. Match agents in agent_descriptions who may be helpful solve the issue (cover all potentially relevant agents, not limited to strongly related ones);
  3. List the matched agents to ensure the agent's responsibilities may be helpful to solve the frontier issue;
  4. Output the JSON list in the specified format.

- Examples:
  Example 1 (Medical Scenario - Need to call agents):
  Input:
  - frontier: {"hypo_id":"1400679e-a0c3-4588-8096-a55ff194b007","label":"Benign adnexal/appendageal tumor","why":"Congenital, slow-growing alopecic scalp plaque with waxy exophytic nodule and yellow dermoscopic hue without vessels favors benign adnexal tumor (e.g., nevus sebaceus).","score":0.7,"level":1,"supports":5,"refutes":0}
  - agent_descriptions: [
      {"expert_name":"Laboratory Physician","description":"Responsible for analysis of blood/pathological test data and interpretation of abnormal indicators"},
      {"expert_name":"Pathologist","description":"Responsible for analysis of pathological section features and judgment of lesion nature"},
      {"expert_name":"Radiologist","description":"Responsible for analysis of CT/MRI imaging features of lesions"}
    ]
  Output JSON:
  [
    {"expert_name":"Pathologist"},
    {"expert_name":"Radiologist"}
  ]

  Example 2 (Microservice Scenario - Need to call agents):
  Input:
  - frontier: {"hypo_id":"7d6c5b4a-3f2e-1d0c-9b8a-76543210fedc","label":"Application response timeout is suspected to be caused by network issues","why":"Response latency spikes only in cross-region calls, local calls normal (10ms vs 500ms), packet loss rate unknown.","score":0.6,"level":1,"supports":2,"refutes":1}
  - agent_descriptions: [
      {"expert_name":"Network Expert","description":"Responsible for troubleshooting network latency, packet loss, bandwidth bottlenecks and other network issues"},
      {"expert_name":"Application Development Expert","description":"Responsible for troubleshooting application layer issues such as code logic and interface call exceptions"}
    ]
  Output JSON:
  [
    {"expert_name":"Network Expert"}
  ]
"""

    system_prompt = system_prompt + task_prompt

    user_prompt = f"""
This is the frontier: {frontier}\n
These are expert descriptions: {expert_descriptions}.\n
Please follow the instructions and complete the task.
"""
    

    return system_prompt, user_prompt


def construct_expert_analyze_prompt(
    dataset: str,
    expert_name: str,
    tool_prompt: str,
    belief: str,
    history: list,
    frontier,
    task_stage: str,  
    decision_result: dict = None  
) -> tuple[str, str]:

    system_prompt_path = os.path.join("prompts", dataset, "system_prompt.py")
    spec = importlib.util.spec_from_file_location(f"{expert_name}_system_prompt", system_prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    system_prompt = getattr(prompt_module, f"{expert_name}_system_prompt")

    if task_stage == "decision_making":
        task_prompt = '''
1. Goal
Focus ONLY on two core judgments:
  1. Determine analysis type (type1/type2) strictly based on tool description;
  2. Determine action decision (retrieve/tool_call/analyze) based on type + information sufficiency;
  3. Output structured JSON with ONLY two mandatory fields: "type", "decision".

2. Constraints (Mandatory Compliance)
  1. Type Determination:
     - type1: Tool description only allows access to "predefined evidence texts" (no executable tools);
     - type2: Tool description supports "executable tools" (including combination of evidence texts + tools);
  2. Information Sufficiency (Align with Professional Experience):
     - "Sufficient": Current belief + retrieve/tool call history → covers ALL core verification/ruling-out items required for the frontier hypothesis in current reasoning;
     - "Insufficient": Any core verification/ruling-out item required for the frontier hypothesis in current reasoning is missing (partial coverage ≠ sufficient);
  3. Decision Mapping:
     - type1 + insufficient → decision: "retrieve";
     - type1 + sufficient → decision: "analyze";
     - type2 + insufficient → decision: "tool_call";
     - type2 + sufficient → decision: "analyze";
  4. Output Rule: Return ONLY standard JSON with "type" (1/2) and "decision" (retrieve/tool_call/analyze) — no extra text, comments or formatting.

3. Instructions
  Core Premise: Regardless of the decision to be made, you must conduct in-depth analysis based on the current belief, historical retrieval/tool call records, and the frontier (most credible hypothesis). For decisions leading to "retrieve" or "tool_call", ensure the judgment directly serves to supplement information for verifying or ruling out the frontier.
  
  Execution Steps:
  Step 1: Carefully read the tool description to determine the analysis type (type1/type2);
  Step 2: Based on your professional experience, list all core verification/ruling-out items required for the frontier hypothesis in current reasoning;
  Step 3: Compare the listed core items with current belief + historical records to identify missing key information;
  Step 4: Map to the corresponding decision according to "type + information sufficiency" rule;
  Step 5: Output the required JSON (strictly follow the output rule).

4. Examples (Decision-Making Stage)
  Example 1 (type1 + retrieve):
  Input:
  - Tool description: "Only pre-prepared evidence texts are available (no other tools)";
  - Current belief: "Patient has neurological symptoms, suspected cerebral vascular abnormality";
  - History: []
  Output JSON:
  {"type": 1, "decision": "retrieve"}

  Example 2 (type2 + tool_call):
  Input:
  - Tool description: "Support calling laboratory test tools + access to pre-prepared evidence texts";
  - Current belief: "Suspected diabetes, fasting blood glucose not tested";
  - History: []
  Output JSON:
  {"type": 2, "decision": "tool_call"}

  Example 3 (type1 + analyze):
  Input:
  - Tool description: "Only pre-prepared evidence texts are available (no other tools)";
  - Current belief: "Patient has neurological symptoms, suspected cerebral vascular abnormality";
  - History: [("I need brain CT and cerebral angiography results", "Brain CT: normal; cerebral angiography: mild stenosis in right MCA")]
  Output JSON:
  {"type": 1, "decision": "analyze"}
'''
        user_prompt = f"""
Tool description: {tool_prompt}
Current belief: {belief}
Retrieve/tool call history: {history if history else "No historical records"}

Please strictly follow the task prompt to output JSON with "type" and "decision" (no extra content).
"""

    elif task_stage == "content_generation":
        if not decision_result or not all(key in decision_result for key in ["type", "decision"]):
            raise ValueError("decision_result must contain 'type' and 'decision' fields for content_generation stage")
        
        type_val = decision_result["type"]
        decision_val = decision_result["decision"]

        print("here we are")
        task_prompt = f'''
1. Goal
Generate content that fully matches the given decision ({decision_val}) with three core requirements: ① Strictly comply with length limits; ② Format as naturally coherent plain text (no structured formats); ③ Professional and informationally complete.

2. Constraints (Mandatory; Invalid if Violated)
- Hard Length Limit: Content < 400 characters. 
- Centered on the current most credible frontier hypothesis, first identify all core items required for its verification or elimination, then compare these core items with the current belief and historical records to pinpoint missing key information, and ultimately determine the tools to be called or the information to be retrieved next based on the analysis type and information sufficiency.
- Exclusive Format Requirement: Content must be a single paragraph of naturally coherent text. STRUCTURED FORMATS ARE FORBIDDEN (e.g., dictionaries, lists, bullet points). Only paragraph-style expression is allowed.
- Information Compliance: Do not duplicate historical retrieval/tool calls; Prioritize supplementing critical missing information; Strictly base on "current belief + historical data" — no fabrication.
- Output Format: Return ONLY standard JSON with three mandatory fields: "type", "decision", "content". No extra text, comments, or blank lines.

3. Instructions (Execute by Decision Type)
- Regardless of decision type, you must conduct in-depth analysis based on the current belief, historical retrieval/tool call records, and the frontier (most credible hypothesis); for "retrieve" or "tool_call", generate tools that align with your current analysis status to assist in verifying or ruling out the frontier.
- If decision is "retrieve" (Type 1 only): Content = Professional retrieval statement + Brief explanation of criticality. Example: "Need the patient's cerebral angiography report (core basis for diagnosing cerebrovascular abnormalities)".
- If decision is "tool_call" (Type 2 only): Content = Executable tool command (strictly match tool format) . Example: {{"tool": "shell", "command": "ps aux"}}
- If decision is "analyze" (Both Type 1 & Type 2): Content = Analysis report integrating "belief + all historical data", completed in one paragraph. Include core conclusion + confirmation of "all key information covered". Example: "Combined with the suspicion of cerebrovascular abnormalities, CT shows uniform density, and angiography reveals mild stenosis of the right middle cerebral artery. This stenosis may cause neurological symptoms; clinical follow-up is recommended. All key imaging information is covered."

4. Examples (For Format & Length Reference Only)
Example 1: Type 1 + Retrieve
Output JSON: {{"type":1,"decision":"retrieve","content":"Need the patient's brain CT plain scan results and cerebral angiography report (core examinations for confirming intracranial vascular abnormalities)"}}

Example 2: Type 2 + Tool_call
Output JSON: {{"type":2,"decision":"tool_call","content":"{{"tool": "shell", "command": "ps aux"}}"}}

Example 3: Type 1 + Analyze
Output JSON: {{"type":1,"decision":"analyze","content":"Combined with the belief of suspected cerebrovascular abnormalities, retrieved CT shows uniform density and cerebral angiography reveals mild stenosis of the right middle cerebral artery. This stenosis may cause neurological symptoms; follow-up is recommended. All key imaging information is covered."}}

Example 4: Type 2 + Analyze
Output JSON: {{"type":2,"decision":"analyze","content":"Metric shows "up" is from 1 to 0, indicating a potential MySQL down issue. And the "mysql.up" metric is disappeared after the alarm, which also supports this conclusion, because if MySQL is down, the monitoring system cannot get these metrics. Based on historical experience, the possibility of a problem with the monitoring system has been ruled out."}}
'''

        user_prompt = f"""
Tool description: {tool_prompt}
Current belief: {belief}
Current frontier - most suspicious hypothesis: {frontier}
Retrieve/tool call history: {history if history else "No historical records"}
Current decision result (from previous stage): {decision_result}

Please strictly follow the task prompt to generate content and output the complete JSON (type + decision + content).
"""

    else:
        raise ValueError("task_stage must be one of ['decision_making', 'content_generation']")

    system_prompt = system_prompt + task_prompt
    # print("Generated system prompt:", system_prompt)
    return system_prompt, user_prompt


def construct_generate_proposal_prompt(dataset, expert_name, belief, graph_description, analyses=[]):

    system_prompt_path = os.path.join("prompts", dataset, "system_prompt.py")
    spec = importlib.util.spec_from_file_location(f"{expert_name}_system_prompt", system_prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    system_prompt = getattr(prompt_module, f"{expert_name}_system_prompt")

    task_prompt = """
- Goal:
  Based on the current reasoning consensus (belief), expert analyses (if available), and existing reasoning graph (including nodes and edges), conduct in-depth logical reasoning and deduction to:
  1. Edit existing nodes in the graph (update "confidence" and "why" field of Hypothesis nodes) based on new information from expert analyses;
  2. Identify unmined information to generate new nodes (Evidence/Hypothesis) and edges (support/refute) for updating the reasoning graph;
  Strictly avoid generating new Hypothesis nodes with duplicate labels to existing ones (update existing nodes via edit instead).

- Constraints:
  1. Output only standard JSON with mandatory top-level fields: "edit", "nodes", "edges" (keep keys even if values are empty arrays);
     - "edit": Array of existing node update records (only for Hypothesis nodes); empty if no nodes need editing;
     - "nodes": Array of newly inferred nodes (Evidence/Hypothesis); empty if no new nodes;
     - "edges": Array of newly inferred edges; empty if no new edges;
  2. Node specifications:
     - Evidence node (new only): Must contain "id" (ev+3-digit), "label" (specific, verifiable factual description, e.g., "test result shows positive for influenza A", "device temperature reaches 85°C"), "node_type":"Evidence";
     - Hypothesis node (new only): Must contain "id" (h+3-digit), "label" (specific positive speculation about the core question, emphasizing "what it is" rather than "what it is not" — i.e., a clear "may be XX" or "is speculated to be XX" statement, not an exclusionary or uncertain claim), "node_type":"Hypothesis", "confidence" (0.0-1.0), "why";
  3. Edge specifications (new only): Must contain "src" (new/existing node id), "dst" (new/existing node id), "relation" (only "support" or "refute"); 
  4. Edit field specifications:
     - Each edit record must contain "node_id" (existing Hypothesis node id), "confidence" (0.0-1.0 float), "why" (updated reasoning integrating new evidence/analyses);
     - "why" must explicitly reference new evidence and expert analyses to explain the confidence change;
  5. All reasoning (edit/generation) must be based on belief/analyses/graph; no ungrounded fabrication;
  6. New nodes must represent unmined information (no duplicate labels/IDs with existing nodes);
  7. Node labels must be accurate and concise; "why" fields (new/edit) must be concise and natural reasoning texts that clearly link to relevant information from belief, expert analyses, and existing graph, without directly referencing node IDs.

- Instructions:
  1. Parse existing graph, belief, and expert analyses to identify:
     a. Existing Hypothesis nodes that need confidence/why updates (due to new evidence/analyses);
     b. Unmined evidence/hypotheses for new node generation;
  2. Generate "edit" field:
     - For existing Hypothesis nodes with changed confidence/why: Create edit records with new values ("why" must include new evidence/analyses);
  3. Generate "nodes" field:
     - New Evidence nodes: Extract/infer factual evidence not in the graph from belief/analyses;
     - New Hypothesis nodes: Propose unique-labeled inferential conclusions not in the graph (no duplicates with existing);
  4. Generate "edges" field:
     - Establish support/refute relations between new nodes, or new nodes and existing nodes;
  5. Output only JSON with "edit", "nodes", "edges" (no syntax errors/extra content);
  6. If no edits/new nodes/new edges: Set all three fields as empty arrays.

- Examples:
  Example 1 (Edit existing node + add new evidence nodes + add refute edges)
  Input:
  - current belief: "David's watch was lost in classroom on July 17 evening; Sarah has alibi (was at library) confirmed by surveillance"
  - expert analyses: ["Surveillance footage proves Sarah was at library from 18:00-20:00 on July 17 (no access to classroom)"]
  - existing graph (graph): {
      "nodes": [
        {"id":"ev001","label":"David placed watch on classroom desk (July 17 evening)","node_type":"Evidence"},
        {"id":"h001","label":"Sarah took David's watch","node_type":"Hypothesis","confidence":0.9,"why":"Sarah was seen entering classroom on July 17 evening"}
      ],
      "edges": [
        {"src":"ev001","dst":"h001","relation":"support"}
      ]
    }
  Output JSON:
  {
    "edit": [
      {
        "node_id": "h001",
        "confidence": 0.1,
        "why": "Original reasoning (Sarah entering classroom) is overridden by new evidence: surveillance footage confirms Sarah was at the library from 18:00 to 20:00 on July 17 (no access to the classroom), and the library is 10 minutes away from the classroom (leaving no time for Sarah to access the classroom). These facts refute the claim that Sarah took the watch, so confidence is reduced from 0.9 to 0.1."
      }
    ],
    "nodes": [
      {"id":"ev002","label":"Surveillance footage shows Sarah was at library 18:00-20:00 on July 17","node_type":"Evidence"},
      {"id":"ev003","label":"Library is 10 minutes away from classroom (no time for Sarah to access classroom)","node_type":"Evidence"}
    ],
    "edges": [
      {"src":"ev002","dst":"h001","relation":"refute"},
      {"src":"ev003","dst":"h001","relation":"refute"}
    ]
  }

  Example 2 (No edit + add new evidence/hypothesis nodes + add support edges)
  Input:
  - current belief: "Patient has fever (38.5°C), cough, and chest pain; chest X-ray shows infiltrates in right lung"
  - expert analyses: ["Infiltrates on chest X-ray are typical of pneumonia; sputum culture pending"]
  - existing graph (graph): {
      "nodes": [
        {"id":"ev001","label":"Patient's temperature is 38.5°C","node_type":"Evidence"},
        {"id":"h001","label":"Patient has infectious lung disease","node_type":"Hypothesis","confidence":0.8,"why":"Fever and cough indicate infectious lung disease"}
      ],
      "edges": [
        {"src":"ev001","dst":"h001","relation":"support"}
      ]
    }
  Output JSON:
  {
    "edit": [],
    "nodes": [
      {"id":"ev002","label":"Patient has cough and chest pain","node_type":"Evidence"},
      {"id":"ev003","label":"Chest X-ray shows infiltrates in right lung","node_type":"Evidence"},
      {"id":"h002","label":"Patient has bacterial pneumonia in right lung","node_type":"Hypothesis","confidence":0.75,"why":"The patient’s cough and chest pain, combined with right lung infiltrates shown on chest X-ray and expert analysis indicating that such infiltrates are typical of pneumonia, support bacterial pneumonia in the right lung as a specific subtype of infectious lung disease. Confidence is 0.75 because sputum culture results are still pending."}
    ],
    "edges": [
      {"src":"ev002","dst":"h001","relation":"support"},
      {"src":"ev003","dst":"h001","relation":"support"},
      {"src":"ev002","dst":"h002","relation":"support"},
      {"src":"ev003","dst":"h002","relation":"support"}
    ]
  }
"""
    system_prompt = system_prompt + task_prompt
    user_prompt = f"""
Here is the current belief:
{belief}
Here are the analyses of other experts: 
{analyses}
Here is the existing graph:
{graph_description}
"""
    
    return system_prompt, user_prompt


def construct_report_or_refine_prompt(dataset, expert_name, belief, graph_description, Report_Flag, frontier):

    system_prompt_path = os.path.join("prompts", dataset, "system_prompt.py")
    spec = importlib.util.spec_from_file_location(f"{expert_name}_system_prompt", system_prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    system_prompt = getattr(prompt_module, f"{expert_name}_system_prompt")

    task_prompt = """
- Goal:
  The central agent first determines the action based on the Report_Flag and the refinement level of the current frontier:
  1. If Report_Flag is True (mandatory report generation): Generate a final answer and a complete reasoning report based on the current belief, existing graph, and frontier;
  2. If Report_Flag is False (non-mandatory): 
     - First evaluate if the current frontier has sufficient granularity to answer the core question (no further refinement possible);
     - If sufficient: Generate a final answer and a complete reasoning report;
     - If insufficient: Generate more fine-grained candidate nodes (hypotheses) that refine the current frontier (e.g., if frontier is "infectious fever", generate "bacterial infection fever", "viral infection fever", "mycoplasma infection fever", etc.).

- Constraints:
  1. Output only standard JSON with no extra content; all output JSON must contain a mandatory top-level field "type" (integer) to identify the output category:
     - Type 1 (Report-type): Mandatory when Report_Flag=True or frontier is sufficiently refined; contains fields "type":1, "answer" (final conclusion, concise text), "report" (complete reasoning chain, detailed text based on belief, graph, and frontier);
       - The answer should be within 30 chars, and the report should be within 500 chars;
       - The report shall be a naturally coherent single paragraph that clearly and completely describes the reasoning process.
     - Type 2 (Candidate-type): Only when Report_Flag=False and frontier is insufficiently refined; contains fields "type":2, "candidates" (array of fine-grained hypothesis nodes);
  2. Candidate node specifications (for type=2 JSON):
     - Each candidate must have "id" (format: h+3-digit number, e.g., h001), "label" (fine-grained hypothesis text), "confidence" (0-1 float), "why" (text explaining why this candidate refines the current frontier, linked to belief/graph);
     - Candidates must be more granular than the current frontier and represent the next level of reasoning (no duplicate/irrelevant content);
  3. Final answer (for type=1 JSON) must directly address the core question and align with the frontier's conclusion (if sufficiently refined);
  4. Ensure no conflicting fields (e.g., type=1 JSON must NOT contain "candidates"; type=2 JSON must NOT contain "answer"/"report").

- Execution Instructions:
  1. Parse the Report_Flag to determine if report generation is mandatory;
  2. If Report_Flag = True:
     a. Synthesize the current belief, existing graph, and frontier to form a final answer;
     b. Generate a complete reasoning report explaining how the answer is derived from belief, graph, and frontier;
     c. Output JSON with "type":1, "answer", and "report" fields;
  3. If Report_Flag = False:
     a. Evaluate the granularity of the current frontier:
        - Sufficient: Frontier has the finest possible level of detail to answer the core question (e.g., "fever caused by Staphylococcus aureus infection" – no further refinement possible);
        - Insufficient: Frontier is too general and can be broken down into more specific hypotheses (e.g., "infectious fever" – can be refined into bacterial/viral/mycoplasma infection);
     b. If frontier is sufficient: Output JSON with "type":1, "answer", and "report" fields;
     c. If frontier is insufficient: Generate fine-grained candidates that refine the frontier, output JSON with "type":2 and "candidates" field;
  4. Ensure JSON output has no syntax errors and strictly complies with the field requirements (no extra/missing fields).

- Examples:
  Example 1 (Report_Flag=True - mandatory report generation, type=1)
  Input:
  - Report_Flag: True
  - current belief: "Patient has fever (38.9°C), elevated WBC (18×10^9/L), positive blood culture for Staphylococcus aureus"
  - current frontier: "Fever caused by Staphylococcus aureus infection"
  - existing graph: "Nodes include ev001 (fever 38.9°C), ev002 (WBC 18×10^9/L), ev003 (positive blood culture for S. aureus); edges link ev001/ev002/ev003 to h001 (bacterial infection fever), h001 to h002 (S. aureus infection fever)"
  Output JSON:
  {
    "type": 1,
    "answer": "The patient's fever is caused by Staphylococcus aureus infection",
    "report": "The patient presents with a fever of 38.9°C, an elevated white blood cell count of 18×10^9/L, and a positive blood culture for Staphylococcus aureus. Clinically, fever combined with elevated white blood cells is a typical manifestation of bacterial infection, as white blood cells are the body's key immune cells that increase in number to fight off bacterial invasions. More importantly, the positive blood culture directly confirms the presence of Staphylococcus aureus in the patient's bloodstream – this is the gold standard for diagnosing bloodstream infections and identifying the pathogenic bacterium. Combining these three pieces of evidence: the patient's fever symptom, the laboratory indicator supporting bacterial infection, and the definitive detection of Staphylococcus aureus, we can conclude that the patient's fever is caused by Staphylococcus aureus infection. This conclusion is sufficiently specific and certain for addressing the core question of 'what causes the patient's fever' – further refinement (such as identifying the specific strain of Staphylococcus aureus) is not necessary for clinical diagnosis and treatment decision-making at this stage, as the key pathogenic cause has been clearly identified."
  }

  Example 2 (Report_Flag=False - frontier insufficient, generate candidates, type=2)
  Input:
  - Report_Flag: False
  - current belief: "Patient has fever (38.5°C), elevated WBC (15×10^9/L), no viral test results"
  - current frontier: "Fever caused by infectious factors"
  - existing graph: "Nodes include ev001 (fever 38.5°C), ev002 (WBC 15×10^9/L); edges link ev001/ev002 to h001 (infectious fever), no further sub-hypotheses"
  Output JSON:
  {
    "type": 2,
    "candidates": [
      {"id":"h002","label":"Fever caused by bacterial infection","confidence":0.75,"why":"Current frontier is 'infectious fever' (too general); elevated WBC (ev002 in graph) strongly supports bacterial infection as a fine-grained subcategory of infectious fever"},
      {"id":"h003","label":"Fever caused by viral infection","confidence":0.2,"why":"Current frontier is 'infectious fever' (too general); no viral test results (belief) mean viral infection is a plausible subcategory, but elevated WBC weakens this hypothesis"},
      {"id":"h004","label":"Fever caused by mycoplasma infection","confidence":0.05,"why":"Current frontier is 'infectious fever' (too general); mycoplasma infection is a rare but possible subcategory of infectious fever, with no direct evidence for/against in belief/graph"}
    ]
  }

  Example 3 (Report_Flag=False - frontier sufficient, generate report, type=1)
  Input:
  - Report_Flag: False
  - current belief: "Patient has fever (39°C), positive influenza A test, normal WBC (6×10^9/L)"
  - current frontier: "Fever caused by influenza A virus infection"
  - existing graph: "Nodes include ev001 (fever 39°C), ev002 (positive influenza A test), ev003 (normal WBC); edges link ev001/ev002 to h001 (viral infection fever), h001 to h002 (influenza A infection fever)"
  Output JSON:
  {
    "type": 1,
    "answer": "The patient's fever is caused by influenza A virus infection",
    "report": "The patient has a high fever of 39°C, and laboratory tests show a positive influenza A test result and a normal white blood cell count of 6×10^9/L. Clinically, viral infections typically do not cause elevated white blood cells – the patient's normal WBC count is consistent with a viral infection rather than a bacterial one, which rules out bacterial causes of fever. More crucially, the positive influenza A test directly detects the presence of influenza A virus in the patient's body, providing definitive evidence for viral infection. Combining the patient's fever symptom, the viral infection-compatible WBC result, and the direct detection of influenza A virus, we can logically conclude that the patient's fever is caused by influenza A virus infection. This conclusion fully addresses the core question of 'what causes the patient's fever' and is sufficiently specific – influenza A virus is a clear and identifiable pathogenic agent, and further refinement (such as distinguishing specific subtypes of influenza A) is unnecessary for the current diagnostic needs. All available evidence mutually confirms the validity of this conclusion, and there is no reasonable basis to generate additional fine-grained hypotheses."
  }
"""

    system_prompt = system_prompt + task_prompt
    user_prompt = f"""
The Report_Flag is {Report_Flag}.
Here is the current belief:
{belief}
Here is the descripion of the most possible hypothesis:
{frontier}
Here is the existing graph:
{graph_description}
"""


    return system_prompt, user_prompt