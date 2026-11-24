"""Backend-independent workload specification for slurmify.

Every deck front-end (EPOCH, WarpX, PiConGPU, ...) must reduce an input
deck to this structure; the sizing/feasibility model then consumes only
the spec, never the deck language.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class Species:
    name: str
    # particles per cell (EPOCH npart_per_cell); None if fixed count
    npart_per_cell: float | None = None
    npart: float | None = None       # absolute count (beams, tracers)
    tracer: bool = False

    def count(self, cells: float, fill: float) -> float:
        if self.npart is not None:
            return self.npart
        if self.npart_per_cell is not None:
            return self.npart_per_cell * cells * fill
        return 0.0


@dataclass
class OutputBlock:
    """One EPOCH output block (or WarpX diagnostics series): writes its own
    file set."""
    name: str
    grid: bool = False
    field_vars: list[str] = field(default_factory=list)
    # species name -> number of stored per-particle variables
    particle_vars: dict[str, int] = field(default_factory=dict)
    n_dumps: int = 0
    dt_snapshot: float | None = None   # s between dumps (None: dump_at_times)

    def bytes_per_file(self, cells: float, counts: dict[str, float],
                       cell: float, particle: float) -> int:
        # axis grids are negligible next to fields; include a token amount
        b = (512 * 1024 if self.grid else 0)
        b += len(self.field_vars) * cells * cell
        for sp, nv in self.particle_vars.items():
            b += int(counts.get(sp, 0.0)) * nv * particle
        return b


@dataclass
class WorkloadSpec:
    code: str
    deck: str
    # grid: EPOCH-style cell counts
    nx: int = 0
    ny: int = 1
    nz: int = 1
    ndims: int = 3
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    cfl: float = 0.95
    t_end: float = 0.0            # s
    dt: float = 0.0               # estimated stable timestep, s
    steps: int = 0
    species: list[Species] = field(default_factory=list)
    blocks: list[OutputBlock] = field(default_factory=list)
    moving_window: bool = False
    dlb: bool = True              # dynamic load balancing available/on
    notes: list[str] = field(default_factory=list)

    @property
    def cells(self) -> float:
        return float(self.nx) * self.ny * self.nz

    def fill_spec(self, fill: float) -> dict[str, float]:
        return {s.name: s.count(self.cells, fill) for s in self.species}

    def to_dict(self) -> dict:
        return asdict(self)
