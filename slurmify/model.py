"""Sizing / feasibility model.

Pure functions: (WorkloadSpec, machine profile) -> ranked, scored
submission candidates, each with its full assumption set attached.
All model constants come from the machine profile's "calibration"
section, so registry-measured values can replace them without code
changes.

Known simplifications (stated honestly, refined by calibration):
  * throughput model = cell-updates/sec, degraded by particles-per-cell
    (PIC push cost); no per-code constants beyond the profile
  * memory = grid arrays + particle structs, DLB headroom factor
  * I/O = synchronous stall per dump into Lustre at per-rank rates, or
    absorb-then-drain when the machine has the bbstage engine
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .spec import WorkloadSpec

DEFAULT_CAL = {
    "cell_updates_per_s_per_core": 1.5e6,
    "particle_pref_per_cell": 2.0,
    "imbalance": 1.25,
    "field_bytes_per_cell": 120,
    "guard_overhead": 1.08,
    "dlb_headroom": 1.5,
    "bytes_per_particle": 128,
    "output_field_bytes_per_cell": 8,
    "output_bytes_per_particle_var": 8,
    "rss_floor_mib": 400,
    "mem_fraction": 0.9,
    "min_cells_per_rank": 131072,
}
GIB = 1024.0 ** 3


@dataclass
class Volume:
    """What a run writes, independent of submission."""
    cells: float
    counts: dict[str, float]              # species -> particles (assumed)
    per_block: dict[str, dict]            # block -> {bytes_file, dumps}
    total_bytes: float
    dumps_max_stall_inputs: float         # largest single-file bytes
    sim_bytes: float = 0.0                # modelled in-simulation footprint
    rate: float = 0.0                     # cell-upd/s/core used for planning

    @property
    def total_tb(self) -> float:
        return self.total_bytes / 1e12

    def total_human(self) -> str:
        return human_bytes(self.total_bytes)


def human_bytes(b: float) -> str:
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= div:
            return f"{b / div:.1f} {unit}"
    return f"{b:.0f} B"


def volume_of(spec: WorkloadSpec, fill: float, cal: dict) -> Volume:
    counts = spec.fill_spec(fill)
    per_block: dict[str, dict] = {}
    total = 0.0
    biggest = 0.0
    for ob in spec.blocks:
        bf = ob.bytes_per_file(spec.cells, counts,
                               cal.get("output_field_bytes_per_cell", 8),
                               cal.get("output_bytes_per_particle_var", 8))
        per_block[ob.name] = {"bytes_file": bf, "dumps": ob.n_dumps}
        total += bf * ob.n_dumps
        biggest = max(biggest, bf)
    return Volume(spec.cells, counts, per_block, total, biggest)


@dataclass
class Candidate:
    nodes: int
    tasks_per_node: int
    ranks: int
    wall_s: float
    node_hours: float
    stall_s_per_dump: float
    mem_rank_mib: float
    cells_per_rank: float
    io_mode: str                       # lustre | bbstage-durable | ...
    decomp: tuple[int, int, int]
    warnings: list[str] = field(default_factory=list)
    feasible: bool = True

    @property
    def tpn(self) -> int:
        return self.tasks_per_node

    def line(self) -> str:
        return (f"nodes={self.nodes:<4} tpn={self.tasks_per_node:<3} "
                f"ranks={self.ranks:<6} wall={fmt_dur(self.wall_s):>10} "
                f"cost={self.node_hours:8.1f} node-h "
                f"RSS/rank={self.mem_rank_mib:6.0f} MiB "
                f"cells/rank={self.cells_per_rank:9.0f} "
                f"dump-stall={fmt_dur(self.stall_s_per_dump)} "
                f"[{self.io_mode}]")


def fmt_dur(s: float) -> str:
    if s != s or s in (float("inf"), -float("inf")):
        return "?"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{sec:02d}"


def effective_rate(spec: WorkloadSpec, fill: float, cal: dict) -> float:
    """cell-updates/s/core, degraded by total particles per cell."""
    npc = sum(s.count(spec.cells, fill) for s in spec.species) / max(
        spec.cells, 1.0)
    return cal["cell_updates_per_s_per_core"] / (
        1.0 + npc / cal["particle_pref_per_cell"])


def decomposition(spec: WorkloadSpec, ranks: int) -> tuple[int, int, int]:
    """nprocx*y*z: factors of `ranks` giving near-cubic subdomains."""
    dims = (spec.nx, max(spec.ny, 1), max(spec.nz, 1))
    best, best_cost = (1, 1, 1), float("inf")
    for a in _divisors_upto(ranks, 4096):
        rem = ranks // a
        for b in _divisors_upto(rem, 4096):
            c = rem // b
            sizes = [dims[i] / d for i, d in enumerate((a, b, c))]
            if any(s < 16 for s in sizes):
                continue
            g = math.prod(sizes) ** (1.0 / 3.0)
            cost = max(abs(math.log(s / g)) for s in sizes)
            if cost < best_cost:
                best, best_cost = (a, b, c), cost
    return best


def _divisors_upto(n: int, cap: int) -> list[int]:
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def plan(spec: WorkloadSpec, machine: dict, *, fill: float = 0.5,
         force: tuple[int, int] | None = None) -> tuple[
        Volume, list[Candidate]]:
    cal = {**DEFAULT_CAL, **machine.get("calibration", {})}
    vol = volume_of(spec, fill, cal)
    rate = effective_rate(spec, fill, cal)
    mem_budget_rank_mb = None  # set per candidate below

    part_bytes = sum(vol.counts.values()) * cal["bytes_per_particle"]
    field_bytes = spec.cells * cal["field_bytes_per_cell"] * cal[
        "guard_overhead"]
    sim_bytes = (field_bytes + part_bytes * (
        cal["dlb_headroom"] if spec.dlb else 1.0))

    cands: list[Candidate] = []
    ladder = machine.get("task_ladder", [32])
    cores = machine["cores_per_node"]
    node_list = [2 ** k for k in range(0, 10)]
    combos = ([(force[0], force[1])] if force else
              [(n, t) for n in node_list for t in ladder if t <= cores])
    for nodes, tpn in combos:
        P = nodes * tpn
        cand = Candidate(nodes=nodes, tasks_per_node=tpn, ranks=P,
                         wall_s=0, node_hours=0, stall_s_per_dump=0,
                         mem_rank_mib=0, cells_per_rank=0, io_mode="",
                         decomp=(1, 1, 1))
        mem_rank = cal["rss_floor_mib"] + sim_bytes / (P * 1048576.0)
        cand.mem_rank_mib = mem_rank
        budget = machine["mem_per_node_mb"] * cal["mem_fraction"] / tpn
        cand.cells_per_rank = spec.cells / P

        ok = True
        if cand.cells_per_rank < cal["min_cells_per_rank"]:
            cand.warnings.append(
                f"cells/rank {cand.cells_per_rank:.0f} < "
                f"{cal['min_cells_per_rank']} (efficiency floor)")
            ok = False
        if mem_rank > budget:
            cand.warnings.append(
                f"mem/rank {mem_rank:.0f} MiB > budget {budget:.0f} MiB")
            ok = False
        cand.decomp = decomposition(spec, P)
        cand.feasible = ok

        step_time = spec.cells / (P * rate * cal["imbalance"])
        step_wall_total = spec.steps * step_time

        # I/O analysis: worst dump stall dominates
        v_w = machine["fs"]["write_gbs_per_rank"] * GIB
        v_nv = machine.get("nvme", {}).get("write_gbs", 4.0) * GIB
        worst_stall, io_mode, io_ok = 0.0, "none", True
        for b, info in vol.per_block.items():
            bf = info["bytes_file"]
            if bf == 0 or info["dumps"] == 0:
                continue
            interval_s = info["dumps"] and (
                spec.steps / info["dumps"]) * step_time
            stall_lustre = bf / (P * v_w)
            if interval_s > 0 and stall_lustre <= 0.25 * interval_s:
                mode, stall = "lustre", stall_lustre
            elif machine.get("has_bbstage") and \
                    machine.get("nvme", {}).get("free_gb", 0) * GIB \
                    >= bf / max(nodes, 1) * 2.0:
                mode = "bbstage"
                stall = bf / (nodes * v_nv)
                if interval_s > 0 and \
                        bf / interval_s > nodes * tpn * v_w:
                    cand.warnings.append(
                        f"block '{b}': drain exceeds Lustre rate "
                        f"({bf / interval_s / GIB:.1f} GiB/s sustained)")
                    io_ok = False
            else:
                mode, stall, io_ok = "INFEASIBLE", stall_lustre, False
            if stall > worst_stall:
                worst_stall, io_mode = stall, mode
        cand.stall_s_per_dump = worst_stall
        cand.io_mode = io_mode
        cand.wall_s = step_wall_total + worst_stall * max(
            max((i["dumps"] for i in vol.per_block.values()), default=0), 1)
        cand.node_hours = cand.wall_s / 3600.0 * nodes
        if not io_ok:
            cand.feasible = False
        cands.append(cand)

    cands.sort(key=lambda c: c.wall_s)
    vol.sim_bytes = sim_bytes
    vol.rate = rate
    return vol, cands
