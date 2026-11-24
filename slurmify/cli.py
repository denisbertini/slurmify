#!/usr/bin/env python3
"""slurmify - deterministic deck -> submission advisor.

    python3 -m slurmify analyze DECK            (also: alias slurmify="python3 -m slurmify")
    python3 -m slurmify plan DECK [--machine machines/orion.json]
                         [--density-fill 0.5] [--top 8]
                         [--nodes N --tasks-per-node T]
                         [--emit run.sh] [--job-name name]
    python3 -m slurmify registry add --json '{"deck":...}'
    python3 -m slurmify registry scrape --epoch-log LOG [--sacct-line ROW]
                               [--sdf2opmd-log LOG] [--json predict.json]
    python3 -m slurmify registry report
    python3 -m slurmify calibrate [--emit-json overrides.json]

Front-ends: EPOCH .deck (WarpX/PiConGPU: contribute a parser producing
the same WorkloadSpec).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import registry
from .epoch import EpochDeck
from .model import DEFAULT_CAL, fmt_dur, plan
from .render import render

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_machine(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def parse_deck(path: str):
    # front-end dispatch could grow; EPOCH today
    return EpochDeck(path).to_spec()


def cmd_analyze(args) -> int:
    spec = parse_deck(args.deck)
    cal = DEFAULT_CAL
    print(f"deck        : {spec.deck}")
    print(f"grid        : {spec.nx} x {spec.ny} x {spec.nz} "
          f"({spec.ndims}d)")
    print(f"t_end/dt    : {spec.t_end:.3e} s / {spec.dt:.3e} s "
          f"-> {spec.steps} steps (cfl {spec.cfl})")
    print(f"moving window: {spec.moving_window}")
    for s in spec.species:
        n = s.count(spec.cells, args.density_fill)
        how = (f"{s.npart_per_cell}/cell x fill {args.density_fill}"
               if s.npart_per_cell is not None else "fixed")
        print(f"species     : {s.name:<12} {int(n):>15,} particles ({how})"
              + (" [tracer]" if s.tracer else ""))
    total = 0.0
    for ob in spec.blocks:
        bf = ob.bytes_per_file(spec.cells,
                               spec.fill_spec(args.density_fill),
                               cal["output_field_bytes_per_cell"],
                               cal["output_bytes_per_particle_var"])
        total += bf * ob.n_dumps
        print(f"output '{ob.name}': {len(ob.field_vars)} fields, "
              f"part-vars {ob.particle_vars or '-'}, "
              f"{bf / 2**30:.2f} GiB/file x {ob.n_dumps} dumps")
    print(f"total volume: {total / 1e12:.2f} TB")
    for n in spec.notes:
        print(f"note        : {n}")
    return 0


def cmd_plan(args) -> int:
    machine = load_machine(args.machine)
    spec = parse_deck(args.deck)
    force = None
    if args.nodes and args.tasks_per_node:
        force = (args.nodes, args.tasks_per_node)
    vol, cands = plan(spec, machine, fill=args.density_fill, force=force)

    cells = spec.cells
    npc = sum(vol.counts.values()) / max(cells, 1.0)
    print(f"== {os.path.basename(spec.deck)} on {machine['name']} ==")
    print(f"{int(spec.steps):,} steps, {npc:.1f} particles/cell, "
          f"{vol.total_tb:.1f} TB to write\n")
    feas = [c for c in cands if c.feasible]
    if not feas:
        print("NO feasible configuration within the searched space; "
              "closest attempts:")
        feas = sorted(cands, key=lambda c: len(c.warnings))[:3]
    shown = feas if force else feas[:args.top]
    for c in shown:
        print("  " + c.line())
        for w in c.warnings:
            print(f"      ! {w}")
    if not force and len(feas) > args.top:
        print(f"  ... {len(feas) - args.top} more")

    best = min(feas, key=lambda c: c.wall_s)
    cheap = min(feas, key=lambda c: c.node_hours)
    print(f"\nfastest   : {best.line()}")
    print(f"cheapest  : {cheap.line()}")
    print("model is a prediction, not a promise: run 20 steps at the "
          "chosen size, then `registry scrape` to calibrate.")

    if args.emit:
        script = render(spec, machine, vol, best,
                        job_name=args.job_name, fill=args.density_fill)
        with open(args.emit, "w") as fh:
            fh.write(script)
        os.chmod(args.emit, 0o755)
        # registry: prediction snapshot awaiting measurement
        registry.append({
            "kind": "predict", "deck": os.path.abspath(spec.deck),
            "machine": machine["name"],
            "predict": {"nodes": best.nodes,
                        "tasks_per_node": best.tasks_per_node,
                        "cells": cells, "npc": npc,
                        "sim_bytes": int(cells * DEFAULT_CAL[
                            "field_bytes_per_cell"]
                            + sum(vol.counts.values())
                            * DEFAULT_CAL["bytes_per_particle"]),
                        "wall_s": best.wall_s,
                        "mem_rank_mib": best.mem_rank_mib},
            "emitted": os.path.abspath(args.emit),
        })
        print(f"emitted {args.emit} (prediction recorded in registry)")
    return 0


def cmd_registry(args) -> int:
    if args.rcommand == "add":
        registry.append(json.loads(args.json))
        print("recorded")
        return 0
    if args.rcommand == "scrape":
        rec: dict = {"kind": "scrape"}
        if args.json:
            rec["predict"] = json.load(open(args.json))
        measured: dict = {}
        if args.epoch_log:
            measured.update(registry.scrape_epoch_log(args.epoch_log))
        if args.sacct_line:
            measured.update(registry.scrape_sacct(args.sacct_line))
        if args.sdf2opmd_log:
            measured.update(registry.scrape_sdf2opmd_log(args.sdf2opmd_log))
        rec["measured"] = measured
        registry.append(rec)
        print(json.dumps(rec["measured"], indent=1))
        return 0
    if args.rcommand == "report":
        recs = registry.load()
        print(f"{len(recs)} record(s) in {registry.DEFAULT_PATH}")
        for r in recs[-10:]:
            print(" ", json.dumps(r, sort_keys=True)[:150])
        return 0
    return 1


def cmd_calibrate(args) -> int:
    machine = load_machine(args.machine)
    recs = registry.load()
    overrides = registry.calibrate(recs, machine)
    if not overrides:
        print("not enough measured records yet (need scraped runs with "
              "a predict block); constants unchanged")
        return 0
    print("proposed calibration overrides (merge into machine profile "
          "'calibration'):")
    print(json.dumps(overrides, indent=1))
    if args.emit_json:
        with open(args.emit_json, "w") as fh:
            json.dump(overrides, fh, indent=1)
        print(f"wrote {args.emit_json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="slurmify")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="deck -> workload spec")
    p.add_argument("deck")
    p.add_argument("--density-fill", type=float, default=0.5,
                   help="fraction of cells actually populated for "
                        "density-dependent species (upper bound: 1.0)")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("plan", help="ranked submission candidates")
    p.add_argument("deck")
    p.add_argument("--machine",
                   default=os.path.join(HERE, "machines", "orion.json"))
    p.add_argument("--density-fill", type=float, default=0.5)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--nodes", type=int)
    p.add_argument("--tasks-per-node", type=int)
    p.add_argument("--emit", help="write sbatch script for best candidate")
    p.add_argument("--job-name", default="epoch")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("registry")
    rsub = p.add_subparsers(dest="rcommand", required=True)
    a = rsub.add_parser("add")
    a.add_argument("--json", required=True)
    s = rsub.add_parser("scrape")
    s.add_argument("--epoch-log")
    s.add_argument("--sacct-line")
    s.add_argument("--sdf2opmd-log")
    s.add_argument("--json", help="predict block json from plan --emit")
    rsub.add_parser("report")
    p.set_defaults(fn=cmd_registry)

    p = sub.add_parser("calibrate")
    p.add_argument("--machine",
                   default=os.path.join(HERE, "machines", "orion.json"))
    p.add_argument("--emit-json")
    p.set_defaults(fn=cmd_calibrate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
