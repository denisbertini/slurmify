#!/usr/bin/env python3
"""Regression tests for deck2submit (no external deps)."""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from d2s.epoch import EpochDeck                                   # noqa: E402
from d2s.model import plan                                         # noqa: E402

FAILS = []


def check(cond, what):
    if cond:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}")
        FAILS.append(what)


def near(a, b, rel, what):
    check(a is not None and abs(a - b) <= rel * max(abs(b), 1e-30),
          f"{what} ({a} ~ {b})")


def main():
    machine = json.load(open(os.path.join(ROOT, "machines", "orion.json")))

    print("EPOCH deck parsing (lwfa_bench.deck):")
    spec = EpochDeck(os.path.join(ROOT, "examples",
                                  "lwfa_bench.deck")).to_spec()
    check((spec.nx, spec.ny, spec.nz) == (1080, 225, 225),
          "grid 1080x225x225")
    near(spec.dt, 1.517e-16, 0.01, "dt 0.152 fs")
    check(1300 < spec.steps < 1340, "steps ~1318")
    check(spec.moving_window, "moving window detected")
    sp = {s.name: s for s in spec.species}
    check("electron" in sp and "electront" in sp, "both species parsed")
    check(sp["electront"].npart == 27000,
          f"tracer npart=27000 (got {sp['electront'].npart})")
    obs = {b.name: b for b in spec.blocks}
    check("normal" in obs and "testparticle" in obs, "both out blocks")
    check(sorted(obs["normal"].field_vars) == ["bz", "ex", "ey"],
          "normal fields ex/ey/bz only (never-variables excluded)")
    check(obs["normal"].particle_vars == {"electron": 8, "electront": 8},
          f"8 particle vars/species (got {obs['normal'].particle_vars})")
    check(obs["normal"].n_dumps == 5, "normal: 4 dumps + dump_final")
    check(obs["testparticle"].particle_vars == {"electront": 8},
          "subset restriction honoured (tracers only)")

    print("volume model:")
    counts = spec.fill_spec(1.0)
    bf = obs["normal"].bytes_per_file(spec.cells, counts, 8, 8)
    # 3 fields x 8 B + 8 vars x 8 B x 54.7M electrons
    near(bf, (3 * 8 * 54.675e6 + 8 * 8 * 54.702e6), 0.02,
         "normal file ~4.8 GB at fill 1.0")

    print("full deck planning (50 npc):")
    spec2 = EpochDeck(os.path.join(ROOT, "examples",
                                   "lwfa.deck")).to_spec()
    vol, cands = plan(spec2, machine, fill=0.5)
    feas = [c for c in cands if c.feasible]
    check(feas, "feasible configs exist")
    check(all(c.cells_per_rank >= 131072 for c in feas),
          "feasible obey cells/rank floor")
    check(max(c.ranks for c in feas) <= 54.675e6 / 131072 * 1.01,
          "rank ceiling from cells/rank floor")
    best = min(feas, key=lambda c: c.wall_s)
    check(best.mem_rank_mib > 500, "particle-heavy deck: RSS/rank sizable")
    check(vol.total_tb > 10, f"tens of TB predicted ({vol.total_tb:.1f})")

    print("force mode + registry round trip:")
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, HOME=td)
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "deck2submit.py"),
             "plan", os.path.join(ROOT, "examples", "lwfa_bench.deck"),
             "--density-fill", "1.0", "--nodes", "2", "--tasks-per-node",
             "32", "--emit", os.path.join(td, "run.sh")],
            capture_output=True, text=True, env=env)
        check(out.returncode == 0, "forced plan exits 0")
        script = open(os.path.join(td, "run.sh")).read()
        check("--nodes=2" in script and "--tasks-per-node=32" in script,
              "forced sizing in sbatch")
        check("deck2submit predictions" in script,
              "prediction header present")
        reg = os.path.join(td, ".deck2submit", "registry.jsonl")
        check(os.path.exists(reg), "prediction recorded")
        # fake measurement, then calibrate
        rec = json.loads(open(reg).read().splitlines()[-1])
        rec2 = {"kind": "scrape", "predict": rec["predict"],
                "measured": {"steps_per_s": 5.0, "rss_rank_mib": 500}}
        with open(reg, "a") as fh:
            fh.write(json.dumps(rec2) + "\n")
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "deck2submit.py"),
             "calibrate"], capture_output=True, text=True, env=env)
        check("cell_updates_per_s_per_core" in out.stdout,
              "calibrate proposes rate override")

    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'ALL TESTS PASS'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
