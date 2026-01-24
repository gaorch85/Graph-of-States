import os
import yaml
import pandas as pd
import csv
import time
from datetime import datetime
from Run.DiagnosisArena import Process_Diagnosis_Symptoms
from Run.FailureDiagnosis_IT import Process_Micro_Symptoms
from belief.graph import BeliefGraph
from belief.fsm import BeliefFSM
from agents.central_agent import Central_Agent
from agents.expert_agent import Expert_Agent
from utils.public_function import load_pkl, save_pkl, generate_causal_graph_html, log_to_file, print_usage_summary



def main():
    ########################################################
    #                      Prepare                         #
    ########################################################

    datasets="DiagnosisArena"

    """Load Configs"""
    Domain_config_path = os.path.join("configs", "Domain_Configs.yaml")
    with open(Domain_config_path, "r", encoding="utf-8") as f:
        cfg_domain = yaml.safe_load(f).get(datasets, {})
    Belief_config_path = os.path.join("configs", "Belief_Configs.yaml")
    with open(Belief_config_path, "r", encoding="utf-8") as f:
        cfg_belief = yaml.safe_load(f)
    Agent_config_path = os.path.join("configs", "Agent_Configs.yaml")
    with open(Agent_config_path, "r", encoding="utf-8") as f:
        cfg_agent = yaml.safe_load(f)
    
    
    if datasets == "DiagnosisArena":
        symptoms, evidence_texts, groundtruths = Process_Diagnosis_Symptoms(cfg_domain, datasets)
        csv_save_path = "Correct-Path-Here"
        os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)

        columns = ["prediction", "report", "answer"]
        if not os.path.exists(csv_save_path):
            pd.DataFrame(columns=columns).to_csv(
                csv_save_path,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                escapechar='"',
                sep=","
            )

    elif datasets == "FailureDiagnosis-IT":
        symptoms, evidence_texts, groundtruths = Process_Micro_Symptoms(cfg_domain, datasets)
        csv_save_path = "Correct-Path-Here"
        os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
        columns = ["prediction", "report", "answer"]
        if not os.path.exists(csv_save_path):
            pd.DataFrame(columns=columns).to_csv(
                csv_save_path,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                escapechar='"',
                sep=","
            )

    else:
        raise ValueError("Current scenario does not exist!")

    """Handle case by case"""
    for idx, symptom in symptoms:
        print(f"\n\n===============[RUN] Case {idx}  =================")

        # 1. Define log path
        case_log_dir = f"logs/Convince/Case_{idx}/"  
        
        # 2. Create directory
        os.makedirs(case_log_dir,exist_ok=True)
        
        # 3. create log name
        log_file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"  #format：20251203_163025.txt
        
        # 4. concate log path
        case_log_file = os.path.join(case_log_dir, log_file_name)

        ########################################################
        #                   Initialization                     #
        ########################################################
        """Get causal graph and state machine"""
        G = BeliefGraph()
        F = BeliefFSM(thresholds=cfg_belief.get("fsm", {}).get("thresholds", {}))

        """Get experts"""
        expert_names = cfg_domain.get("expert_names")
        central_name = cfg_domain.get("head_name")

        expert_agents = []
        for expert_name in expert_names:
            expert_agent = Expert_Agent(expert_name, graph=G, fsm=F, case_id = idx, dataset = datasets, log_path = case_log_file,  cfg_domain = cfg_domain, llm_args=cfg_agent.get("llm_args"), evidence_text=evidence_texts[idx], max_retrieval_steps=cfg_agent.get("llm_args")["max_retrieval_steps"])
            expert_agents.append(expert_agent)

        central_agent = Central_Agent(central_name, graph=G, fsm=F, dataset = datasets, log_path = case_log_file, case_id = idx, case_symptom = symptom, llm_args=cfg_agent.get("llm_args"), cfg_domain = cfg_domain, experts=expert_agents, evidence_text=evidence_texts[idx])

        ########################################################
        #                    Start Diagnosis                   #
        ########################################################

        """Ingest phase"""
        central_agent.ingest()


        """Run Phase"""
        answer, report = central_agent.run()
        
        ########################################################
        #                 Write results into csv               #
        ########################################################
        

        if datasets == "DiagnosisArena":
            groundtruth = groundtruths[idx]
            print(f"Answer: {answer}")
            print(f"Report: {report}")

            current_row = {
                "prediction": answer.strip(),
                "report": report.strip(),
                "answer": groundtruth.strip()
            }

            pd.DataFrame([current_row], columns=columns).to_csv(
                csv_save_path,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                escapechar='"',
                sep=",",
                mode="a",         
                header=False       
            )
        
        elif datasets == "FailureDiagnosis-IT":
            groundtruth = groundtruths[idx]
            print(f"Answer: {answer}")
            print(f"Report: {report}")

            current_row = {
                "prediction": answer.strip(),
                "report": report.strip(),
                "answer": groundtruth.strip()
            }

            pd.DataFrame([current_row], columns=columns).to_csv(
                csv_save_path,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                escapechar='"',
                sep=",",
                mode="a",
                header=False
            )
       

        """Use HTML to show the change of causal graph"""
        graph_list = central_agent.graphs
        html_path = f"/data/luoyu6/Convince/output/Micro/html/case{idx}.html"
        generate_causal_graph_html(graph_list, html_path)

    print_usage_summary()
        

if __name__ == "__main__":
    main()
