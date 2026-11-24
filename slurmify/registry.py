"""Run registry: every submission's predictions and (later) measurements
in one JSONL file. Predictions without calibration are guesses; measured
records let `slurmify calibrate` propose better model constants.

Append with:
    slurmify.py registry add   --json '{...}'           # manual
    slurmify.py registry scrape --epoch-log out.log --sacct-line ...
"""

from __future__ import annotations

import json
import os
import re
import time

DEFAULT_PATH = os.path.expanduser("~/.slurmify/registry.jsonl")

_STEP_RE = re.compile(
    r"step\D+(\d+)\b.*?(?:elapsed|wall)[^0-9]*(\d+):(\d+):(\d+)", re.I)
_EPOCH_TOTAL_RE = re.compile(r"total\s+elapsed\s+time[^0-9]*(\d+):(\d+):"
                             r"(\d+)", re.I)
_SDF2OPMD_RSS_RE = re.compile(r"peak RSS/rank (\d+) MiB")
_SDF2OPMD_T_RE = re.compile(r"over (\d+) file\(s\): total ([\d.]+) s")


def append(record: dict, path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def load(path: str = DEFAULT_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def scrape_epoch_log(path: str) -> dict:
    """Extract steps/s from EPOCH step lines (needs stdout_frequency>0)."""
    out: dict = {}
    seen = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if m := _STEP_RE.search(line):
                step = int(m.group(1))
                secs = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + \
                    int(m.group(4))
                seen.append((step, secs))
            if m := _EPOCH_TOTAL_RE.search(line):
                out["total_s"] = int(m.group(1)) * 3600 + int(m.group(2)) \
                    * 60 + int(m.group(3))
    if len(seen) >= 2:
        (s0, t0), (s1, t1) = seen[0], seen[-1]
        if s1 > s0 and t1 >= t0:
            out["steps_per_s"] = (s1 - s0) / (t1 - t0)
    if "total_s" in out and seen:
        out.setdefault("steps_per_s", seen[-1][0] / out["total_s"])
    return out


def scrape_sacct(line: str) -> dict:
    """One `sacct -X -p --format=JobID,Elapsed,MaxRSS,AveRSS` row
    (parses plain values; suffixes K/M/G honoured for MaxRSS)."""
    out: dict = {}
    if m := re.search(r"MaxRSS=(\d+)([KMG]?)", line):
        kb = int(m.group(1))
        mult = {"": 1.0 / 1024, "K": 1.0 / 1024, "M": 1.0, "G": 1024.0}
        out["maxrss_mib"] = kb * mult[m.group(2)]
    if m := re.search(r"Elapsed=(?:(\d+)-)?(\d+):(\d+):(\d+)", line):
        out["elapsed_s"] = (int(m.group(1) or 0) * 86400
                            + int(m.group(2)) * 3600 + int(m.group(3)) * 60
                            + int(m.group(4)))
    return out


def scrape_sdf2opmd_log(path: str) -> dict:
    out: dict = {}
    with open(path, errors="replace") as fh:
        text = fh.read()
    if m := _SDF2OPMD_RSS_RE.search(text):
        out["convert_rss_rank_mib"] = int(m.group(1))
    if m := _SDF2OPMD_T_RE.search(text):
        out["convert_files"] = int(m.group(1))
        out["convert_total_s"] = float(m.group(2))
    return out


def calibrate(records: list[dict], machine: dict) -> dict:
    """Median-derived calibration overrides from measured records that
    carry `measured` + `predict` blocks. Returns machine['calibration']
    overrides (empty if too little data)."""
    import statistics as st

    def collect(fn):
        vals = []
        for r in records:
            v = fn(r)
            if v:
                vals.append(v)
        return vals

    cal = dict(machine.get("calibration", {}))
    out: dict = {}

    # derive cell-update rate from measured steps/s
    pref = cal.get("particle_pref_per_cell", 2.0)
    rates = []
    for r in records:
        m, p = r.get("measured", {}), r.get("predict", {})
        sps = m.get("steps_per_s")
        cells, npc = p.get("cells"), p.get("npc", 0)
        P = p.get("nodes", 0) * p.get("tasks_per_node", 0)
        if sps and cells and P:
            rates.append(sps * cells * (1.0 + npc / pref) / P)
    if rates:
        out["cell_updates_per_s_per_core"] = st.median(rates)

    # derive RSS floor: measured rank RSS minus modelled payload
    floors = []
    for r in records:
        m, p = r.get("measured", {}), r.get("predict", {})
        rss = m.get("rss_rank_mib") or m.get("maxrss_mib")
        sim, P = p.get("sim_bytes"), p.get("nodes", 0) * p.get(
            "tasks_per_node", 0)
        if rss and sim and P:
            floors.append(rss - sim / (P * 1048576.0))
    if floors:
        out["rss_floor_mib"] = max(64.0, st.median(floors))

    imb = collect(lambda r: r.get("measured", {}).get("imbalance"))
    if imb:
        out["imbalance"] = st.median(imb)
    _ = cal
    return out
