# slurmify

**Deck in, good submission out.** Parse a PIC input deck, compute what
the run *needs* (steps, memory, volume), enumerate submissions against a
machine profile, and emit an sbatch script whose comments carry the full
prediction table. No solvers, no ML, no cluster calls - deterministic
arithmetic plus honest constants, and a registry that lets measured
runs replace the guesses.

```
deck --(front-end)--> WorkloadSpec --(model x machine profile)-->
    ranked candidates --> sbatch + prediction table + registry record
```

## Quick start

```sh
python3 -m slurmify analyze examples/lwfa.deck --density-fill 0.5
python3 -m slurmify plan examples/lwfa.deck --density-fill 0.5 --emit run.sh
sbatch run.sh
# after the run: feed reality back in
python3 -m slurmify registry scrape --epoch-log data/epoch.out.log \
    --sacct-line "$(sacct -n -X -P -j $JOBID --format=Elapsed,MaxRSS)" \
    --json <(python3 -c "...predict block from plan...")   # see CLI help
python3 -m slurmify calibrate        # proposes better constants from data
./tests/run_tests.py              # regression suite
```

## What is predicted from the deck alone (no guessing for these)

- grid, cell counts, timestep (`cfl / (c sqrt(S 1/dx^2)))`, steps
- particle counts (per-cell x fill factor, fixed counts, tracers)
- bytes per dump file and total volume (EPOCH output blocks honour
  `never`, downsample modes, and subset restrictions for particle vars)

## What is *modelled* (constants in `machines/*.json:calibration`)

- throughput: cell-updates/s/core, degraded by particles/cell, x imbalance
- memory: grid arrays + particle structs + DLB headroom + RSS floor
- I/O: per-dump stall vs per-rank Lustre rates; `bbstage` engine mode
  (absorb to NVMe, drain in background) chosen when a direct write would
  stall > 25% of the dump interval and staging capacity suffices

Every number in the emitted sbatch says which model constant produced
it. Defaults are order-of-magnitude; **the registry is how they stop
being guesses**: `plan --emit` records the prediction,
`registry scrape` appends measurements (EPOCH step lines, sacct MaxRSS,
`sdf2opmd` peak-RSS lines), and `calibrate` proposes median-derived
overrides from the (predicted, measured) pairs. Five or ten logged runs
and the tool is telling you *your* machine's numbers.

## Known limitations (deliberate)

- `--density-fill`: density-shaped species are bounded, not integrated
  (deck density profiles are not evaluated on a grid)
- deck preprocessor covers numeric constants; decks requiring EPOCH's
  own symbolic evaluation may miss a derived quantity (it will say so)
- no PSATD/FFT cost term yet (calibration absorbs part of it)
- machine profile ships with LRZ Orion + one user's submit pattern;
  copy the JSON for other sites
- WarpX / PiConGPU front-ends: contribute parsers emitting
  `slurmify.spec.WorkloadSpec` (the rest of the pipeline is code-agnostic by
  construction)

## What "optimal" means here

The plan prints the feasible set with *both* objectives (wall vs
node-hours) and refuses to pretend one scalar wins: fastest and cheapest
are reported separately, and infeasible sizes carry their violation
reason (cells/rank floor, memory, dump stall, NVMe staging capacity).
Choosing between points on the frontier is physics + queue reality; the
tool's job is that the frontier and the assumptions are never hidden.
