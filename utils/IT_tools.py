"""
In this file, we define several tools used to solve failure diagnosis in distributed systems.
"""

import csv
import difflib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.public_function import evidence_retriever


def _parse_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts / 1000 if ts > 1e12 else ts
    if isinstance(value, str):
        try:
            ts_num = float(value)
            return ts_num / 1000 if ts_num > 1e12 else ts_num
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return None
    return None


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return str(ts)


def _earliest_alarm_ts(alarms: Any) -> Optional[float]:
    if not alarms:
        return None
    timestamps = []
    for alarm in alarms:
        start_at = alarm.get("startAt") or alarm.get("start_at")
        ts = _parse_ts(start_at)
        if ts:
            timestamps.append(ts)
    return min(timestamps) if timestamps else None


def _load_shell_records(shell_path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not os.path.isfile(shell_path):
        return records
    with open(shell_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd = (row.get("command") or "").strip()
            out = (row.get("output") or "").strip()
            if not cmd and not out:
                continue
            records.append({"command": cmd, "output": out})
    return records


def _filter_by_keywords(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    text_lower = text.lower()
    return any(k.lower() in text_lower for k in keywords)


@dataclass
class Telemetry:
    case_id: int
    telemetry_dir: str
    metric_dir: str
    log_dir: str
    shell_path: str
    metric_names: List[str] = field(default_factory=list)
    default_ts: Optional[float] = None
    shell_records: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_case_resources(cls, case_id: int, resources: Dict[str, Any]) -> "Telemetry":
        telemetry_dir = resources.get("telemetry_dir", "")
        metric_dir = os.path.join(telemetry_dir, "metrics")
        log_dir = os.path.join(telemetry_dir, "logs")
        shell_path = os.path.join(telemetry_dir, "shell", "shell.csv")
        metric_names = (
            sorted([f for f in os.listdir(metric_dir) if f.endswith(".csv")])
            if os.path.isdir(metric_dir)
            else []
        )
        shell_records = _load_shell_records(shell_path)
        default_ts = resources.get("default_ts") or _earliest_alarm_ts(resources.get("alarms"))
        return cls(
            case_id=case_id,
            telemetry_dir=telemetry_dir,
            metric_dir=metric_dir,
            log_dir=log_dir,
            shell_path=shell_path,
            metric_names=metric_names,
            default_ts=default_ts,
            shell_records=shell_records,
        )

    # ---------------------- prompt helpers ---------------------- #
    def available_metrics_text(self) -> str:
        return ", ".join(self.metric_names) if self.metric_names else "None"
    
    
    def build_tool_prompt(self) -> str:
        metric_list_text = ", ".join(self.metric_names) if self.metric_names else "None"
        default_time_text = _fmt_ts(self.default_ts)
        prompt = f"""
You can access three offline telemetry tools (priority: shell > log > metric). These tools work on provided snapshots (no real host access).
Default incident time (use if not specified): {default_time_text}.

Tool call format (always JSON string):
- Shell tool: {{"tool": "shell", "command": "<shell command to inspect>"}}
- Log tool: {{"tool": "log", "keywords": ["ERROR","kubelet"], "limit": 15}}
- Metric tool: {{"tool": "metric", "metric_names": ["cpu.used.percent","memory.used.percent"], "timestamp": "<ISO or epoch>", "window_minutes": 10}}

Constraints:
- Choose up to 3 metrics per call; recommended window is +/-10 minutes (max 60) around the interested timestamp (default +/-1h if not specified).
- Log tool auto-filters ERROR lines; provide keywords to narrow further.
- Shell tool does fuzzy matching against recorded shell.csv.

Experience:
- if you analyze metric about MySQL, first check "up" metric to see if MySQL is running. Other metrics (e.g., mysql.up, mysql.current.connections.num, mysql.data.missing) maybe unable to collect data when MySQL is down. This is not the issue of the monitoring system, but a symptom of MySQL being down.

Examples:
1) To check CPU usage around incident time:
{{"tool": "metric", "metric_names": ["cpu.used.percent"], "timestamp": "{default_time_text}", "window_minutes": 10}}  
2) To find error logs related to kubelet:
{{"tool": "log", "keywords": ["ERROR","kubelet"], "limit": 15}}  
3) To inspect running processes:
{{"tool": "shell", "command": "ps aux"}}

Only JSON format is accepted.

Available metric files:
{metric_list_text}
"""
        return prompt.strip()

    # ---------------------- tool dispatcher ---------------------- #
    def dispatch_tool(self, raw_request: Any, enable_fuzzy_shell: bool = True) -> str:
        request = self._normalize_tool_request(raw_request)
        tool = request.get("tool")
        if tool == "metric":
            return self.metric_tool(request)
        if tool == "log":
            return self.log_tool(request)
        if tool == "shell":
            return self.shell_tool(request, enable_fuzzy=enable_fuzzy_shell)
        return f"Unsupported tool in request: {raw_request}"

    def _normalize_tool_request(self, raw_request: Any) -> Dict[str, Any]:
        if isinstance(raw_request, dict):
            request = dict(raw_request)
        else:
            request = {}
            if isinstance(raw_request, str):
                try:
                    request = json.loads(raw_request)
                except Exception:
                    request["raw"] = raw_request
            else:
                request["raw"] = str(raw_request)

        raw_text = ""
        if "raw" in request:
            raw_text = str(request["raw"])
        elif isinstance(raw_request, str):
            raw_text = raw_request

        tool = request.get("tool") or request.get("tool_name")
        if not tool and raw_text:
            lower = raw_text.lower()
            if "metric" in lower:
                tool = "metric"
            elif "log" in lower:
                tool = "log"
            elif "shell" in lower or any(k in lower for k in ["top", "ps", "df", "jstack"]):
                tool = "shell"
        request["tool"] = tool or "shell"  
        return request

    # ---------------------- metric tool ---------------------- #
    def _resolve_metric_file(self, name: str) -> Optional[str]:
        if not name:
            return None
        name_lower = name.lower()
        # print("Resolving metric file for name:", name_lower)
        # print("Available metric names:", self.metric_names)
        exact = [m for m in self.metric_names if m.lower() == f"{name_lower}.csv"]
        if exact:
            return os.path.join(self.metric_dir, exact[0])

        candidates = [m for m in self.metric_names if name_lower in m.lower()]
        if not candidates:
            matches = difflib.get_close_matches(name_lower, self.metric_names, n=1, cutoff=0.0)
            candidates = matches
        if not candidates:
            return None
        return os.path.join(self.metric_dir, candidates[0])

    def _load_metric_rows(self, file_path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ts(row.get("timestamp"))
                val_raw = row.get("value")
                try:
                    val = float(val_raw) if val_raw not in (None, "") else None
                except Exception:
                    val = None
                row_clean = {
                    k: v
                    for k, v in row.items()
                    if k not in {"expression", "normal_min", "normal_max", "resourceTy", "application"}
                    and v not in (None, "")
                }
                if val is not None:
                    row_clean["value"] = val
                rows.append({"ts": ts, "value": val, "row": row_clean})
        return rows

    def metric_tool(self, request: Dict[str, Any]) -> str:
        metric_names = request.get("metric_names") or request.get("metrics") or []
        if isinstance(metric_names, str):
            metric_names = [m.strip() for m in metric_names.split(",") if m.strip()]
        metric_names = list(metric_names)[:3]
        # print("Requested metric names:", metric_names)
        if not metric_names:
            return "No metric names provided. Available metrics: " + ", ".join(self.metric_names[:20])

        center_ts = _parse_ts(
            request.get("timestamp") or request.get("ts") or request.get("time") or self.default_ts
        )
        window_min = request.get("window_minutes") or request.get("window") or 10
        try:
            window_min = int(window_min)
        except Exception:
            window_min = 10
        window_min = max(1, min(window_min, 60))
        low_ts = center_ts - window_min * 60 if center_ts else None
        high_ts = center_ts + window_min * 60 if center_ts else None

        outputs: List[str] = []
        for name in metric_names:
            # print("Processing metric name:", name)
            file_path = self._resolve_metric_file(name)
            # print("Resolved metric file path:", file_path)
            if not file_path:
                outputs.append(f"[metric] {name}: file not found in telemetry metrics")
                continue
            rows = self._load_metric_rows(file_path)
            filtered = rows
            if center_ts is not None:
                filtered = [r for r in rows if r["ts"] is None or (low_ts <= r["ts"] <= high_ts)]
            if not filtered:
                filtered = sorted(rows, key=lambda r: r["ts"] or 0)[-5:]

            values = [r["value"] for r in filtered if r["value"] is not None]
            stats = ""
            if values:
                stats = (
                    f"stats(min/avg/max): {min(values):.4f}/{sum(values)/len(values):.4f}/{max(values):.4f}; "
                    f"samples={len(values)}"
                )

            samples = sorted(
                filtered,
                key=lambda r: abs((r["ts"] or 0) - (center_ts or 0)) if center_ts else (r["ts"] or 0),
            )[:10]
            sample_lines = []
            for s in samples:
                row = s["row"]
                ts_text = _fmt_ts(s["ts"])
                val_text = row.get("value") or (f"{s['value']:.4f}" if s["value"] is not None else "n/a")
                inst = row.get("instance") or row.get("resourceTy") or ""
                sample_lines.append(f"- {ts_text} value={val_text} instance={inst}")

            outputs.append(
                f"[metric] {os.path.basename(file_path)} window=+/-{window_min}m around {_fmt_ts(center_ts)} "
                f"{stats}\n" + "\n".join(sample_lines)
            )

        return "\n".join(outputs)

    # ---------------------- log tool ---------------------- #
    def log_tool(self, request: Dict[str, Any]) -> str:
        keywords = request.get("keywords") or request.get("terms") or []
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        limit = request.get("limit") or 15
        try:
            limit = int(limit)
        except Exception:
            limit = 15
        limit = max(1, min(limit, 50))

        entries: List[Tuple[float, str]] = []
        if not os.path.isdir(self.log_dir):
            return "Log directory missing."

        for log_file in os.listdir(self.log_dir):
            if not log_file.endswith(".csv"):
                continue
            full_path = os.path.join(self.log_dir, log_file)
            with open(full_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    msg = row.get("message", "")
                    level = row.get("level", "")
                    ts = _parse_ts(row.get("timestamp"))

                    if "ERROR" not in level.upper() and "ERROR" not in msg.upper():
                        if keywords:
                            pass
                        else:
                            continue
                    if not _filter_by_keywords(msg, keywords):
                        continue
                    ts_val = ts if ts is not None else 0
                    entries.append(
                        (
                            ts_val,
                            f"[log] {log_file} { _fmt_ts(ts)} level={level} message={msg}",
                        )
                    )

        if not entries:
            return "No matching log lines (filtered by ERROR and keywords)."

        entries.sort(key=lambda x: x[0])
        picked = [e[1] for e in entries[-limit:]]
        return "\n".join(picked)

    # ---------------------- shell tool ---------------------- #
    def shell_tool(self, request: Dict[str, Any], enable_fuzzy: bool = True) -> str:
        if isinstance(request, dict):
            command = (
                request.get("command")
                or request.get("cmd")
                or request.get("query")
                or request.get("content")
            )
        else:
            command = str(request)

        if not command:
            return "Shell tool needs a 'command' field."

        command = command.strip()

        if not self.shell_records:
            return "No shell snapshots available."

        for rec in self.shell_records:
            if command == rec["command"]:
                return f"[shell snapshot]\ncommand: {rec['command']}\noutput:\n{rec['output']}"

        if not enable_fuzzy:
            return "No exact shell snapshot matched your command."

        for rec in self.shell_records:
            if command in rec["command"]:
                return f"[shell snapshot]\ncommand: {rec['command']}\noutput:\n{rec['output']}"

        evidence_texts = [
            f"command: {rec['command']}\noutput:\n{rec['output']}" for rec in self.shell_records
        ]
        match = evidence_retriever(query=command, evidence_texts=evidence_texts)
        return match or "No relevant shell snapshot matched your command."
