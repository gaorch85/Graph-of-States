import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _build_alarm_description(alarms: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Build a readable alarm summary and default timestamp."""
    lines: List[str] = []
    default_ts = None
    for idx, alarm in enumerate(alarms, start=1):
        labels = alarm.get("labels", {}) or {}
        alert_name = alarm.get("alertName") or labels.get("alertname") or "unknown"
        level = alarm.get("level") or labels.get("level") or "P?"
        instance = labels.get("instance") or alarm.get("instance") or "unknown"
        app = labels.get("app_name") or alarm.get("appName") or ""
        desc = (
            labels.get("description")
            or labels.get("description_en")
            or alarm.get("description")
            or alarm.get("descriptionEn")
            or ""
        )
        value = labels.get("itemvalue") or alarm.get("currentValue")
        unit = labels.get("metric_unit") or alarm.get("valueUnit") or ""
        start_at = alarm.get("startAt")
        if start_at and default_ts is None:
            try:
                default_ts = float(
                    datetime.fromisoformat(start_at.replace("Z", "+00:00")).timestamp()
                )
            except Exception:
                default_ts = None

        line = (
            f"[Alert#{idx}] {alert_name} level {level} on instance {instance}"
            f"{' (app '+app+')' if app else ''} started at {start_at or 'unknown'}"
            f"{': ' + desc if desc else ''}"
        )
        if value is not None:
            line += f" | observed value: {value}{unit}"
        metric_key = labels.get("metricKey") or labels.get("metric_key")
        if metric_key:
            line += f" | metric: {metric_key}"
        lines.append(line)

    symptom = "Incoming infrastructure alarms:\n" + "\n".join(lines)
    symptom += "\nTelemetry snapshots (metrics/logs/shell) are available for LinuxAgent to investigate."
    return symptom, default_ts


def Process_Micro_Symptoms(cfg_domain, dataset):
    """
    load data from company L
    """
    dataset_path = os.path.join("datasets", "Mirco", "data")
    case_dirs = [d for d in os.listdir(dataset_path) if d.startswith("case_")]
    case_dirs = sorted(case_dirs, key=lambda d: int(re.search(r"\d+$", d).group()))

    symptoms: List[Tuple[int, str]] = []
    evidence_texts: List[Dict[str, Any]] = []
    answers: List[str] = []

    for idx, case_name in enumerate(case_dirs):
        case_root = os.path.join(dataset_path, case_name)
        alarm_path = os.path.join(case_root, "alarm.json")
        answer_path = os.path.join(case_root, "answer.txt")
        telemetry_dir = os.path.join(case_root, "telemetry")

        with open(alarm_path, "r", encoding="utf-8") as f:
            alarms = json.load(f)
        symptom_text, default_ts = _build_alarm_description(alarms)

        with open(answer_path, "r", encoding="utf-8") as f:
            answer = f.read().strip()

        case_resources = {
            "telemetry_dir": telemetry_dir,
            "alarms": alarms,
            "default_ts": default_ts,
        }

        symptoms.append((idx, symptom_text))
        evidence_texts.append(case_resources)
        answers.append(answer)

    return symptoms, evidence_texts, answers
