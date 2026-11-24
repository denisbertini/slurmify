"""sbatch generation: reproduces the LRZ Orion / apptainer submit pattern
with the prediction table embedded as comments."""

from __future__ import annotations

from .model import Candidate, Volume, fmt_dur
from .spec import WorkloadSpec

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --tasks-per-node={tpn}
#SBATCH --no-requeue
#SBATCH --mem-per-cpu={mem_mb}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mail-type ALL
#SBATCH --mail-user {mail_user}
#SBATCH -o %j.out.log
#SBATCH -e %j.err.log
#
# ================= slurmify predictions (see README: registry to
# ================= refine; measured values beat these comments)
#   deck           : {deck}
#   grid           : {nx} x {ny} x {nz}   dt = {dt:.3e} s -> {steps} steps
#   particles      : {npcounts}
#   volume         : {total_tb:.1f} TB total, worst file {biggest_gib:.1f} GiB
#   model          : {rate:.2e} cell-upd/s/core (particles/cell = {npc:.1f}),
#                    imbalance x{imbalance}, I/O mode {io_mode}
#   predicted wall : {wall}   cost {node_hours:.1f} node-hours
#   per-rank       : {mem_mib:.0f} MiB model RSS, {cpr:.0f} cells,
#                    dump stall {stall}
#   decomposition  : nprocx={ax} nprocy={ay} nprocz={az}
#                    (add to begin:control to pin the split)
#   warnings       : {warnings}
# =====================================================================
set -euo pipefail

export OMP_NUM_THREADS=1
export CONT=${{CONT:-{sif}}}
export EPOCH_BIN=${{EPOCH_BIN:-{epoch_bin}}}
export OMPI_MCA_io=romio341
export APPTAINER_BINDPATH=/lustre/rz/dbertini,/cvmfs
export APPTAINER_SHARENS=true
export APPTAINER_CONFIGDIR=/tmp/$USER

DATADIR=${{DATADIR:-$PWD/data}}
mkdir -p "$DATADIR"
cp "{deck}" "$DATADIR/epoch.deck"

srun --export=ALL apptainer exec "$CONT" "$EPOCH_BIN" < "$DATADIR/epoch.deck" \\
    > "$DATADIR/epoch.out.log" 2>&1 &
SIMPID=$!
echo "epoch3d started (pid $SIMPID), log: $DATADIR/epoch.out.log"
wait $SIMPID
echo "simulation complete"
"""


def render(spec: WorkloadSpec, machine: dict, vol: Volume, cand: Candidate,
           *, job_name: str = "epoch", partition: str | None = None,
           cal: dict | None = None, fill: float = 0.5) -> str:
    from .model import DEFAULT_CAL, effective_rate
    cal = {**DEFAULT_CAL, **(cal or machine.get("calibration", {}))}
    sched = machine.get("scheduler", {})
    counts = ", ".join(f"{k}={int(v):,}" for k, v in vol.counts.items())
    npc = sum(vol.counts.values()) / max(spec.cells, 1.0)
    time_limit = _time_limit(cand.wall_s, sched.get("time_limit_max"))
    return TEMPLATE.format(
        job_name=job_name, nodes=cand.nodes, tpn=cand.tasks_per_node,
        mem_mb=int(cand.mem_rank_mib * 1.35) + 512,
        partition=partition or sched.get("partition_default", "long"),
        time_limit=time_limit,
        mail_user=sched.get("mail_user", ""),
        deck=spec.deck, nx=spec.nx, ny=spec.ny, nz=spec.nz,
        dt=spec.dt, steps=spec.steps, npcounts=counts,
        total_tb=vol.total_tb, biggest_gib=vol.dumps_max_stall_inputs /
        (1024.0 ** 3),
        rate=effective_rate(spec, fill, cal), npc=npc,
        imbalance=cal.get("imbalance", 1.25), io_mode=cand.io_mode,
        wall=fmt_dur(cand.wall_s), node_hours=cand.node_hours,
        mem_mib=cand.mem_rank_mib, cpr=cand.cells_per_rank,
        stall=fmt_dur(cand.stall_s_per_dump),
        ax=cand.decomp[0], ay=cand.decomp[1], az=cand.decomp[2],
        warnings="; ".join(cand.warnings) or "none",
        sif=sched.get("sif", ""), epoch_bin=sched.get("epoch_bin",
                                                      "epoch3d"))


def _time_limit(wall_s: float, max_tl: str | None) -> str:
    lim = wall_s * 1.3 + 1800
    if max_tl:
        h, m, s = (int(x) for x in max_tl.split(":"))
        lim = min(lim, h * 3600 + m * 60 + s)
    return f"{int(lim // 3600):02d}:{int(lim % 3600 // 60):02d}:00"
