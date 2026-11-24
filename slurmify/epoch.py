"""EPOCH ``.deck`` front-end.

Parses the constant/control/species/output/subset/window blocks and
numerically evaluates the deck's unit-ful expressions well enough to
extract grid size, timestep, particle counts and output volume. Anything
that cannot be evaluated numerically (density profiles, laser envelopes)
is skipped - the sizing model only ever consumes scalars.

This is deliberately NOT a full EPOCH preprocessor: symbolic constants
that depend on runtime grid probing are unsupported, and dimensional
analysis is faked by evaluating pure numbers (which is exact for the
quantities we need: cell counts, times, particle numbers).
"""

from __future__ import annotations

import math
import re

from .spec import OutputBlock, Species, WorkloadSpec

UNITS = {
    # lengths
    "m": 1.0, "micron": 1e-6, "microns": 1e-6, "um": 1e-6, "nm": 1e-9,
    "mm": 1e-3, "cm": 1e-2, "km": 1e3,
    # times
    "s": 1.0, "fs": 1e-15, "femto": 1e-15, "ps": 1e-12, "pico": 1e-12,
    "ns": 1e-9, "ns": 1e-9, "micro": 1e-6, "us": 1e-6, "ms": 1e-3,
    # physics (SI)
    "c": 299792458.0, "clight": 299792458.0,
    "qe": 1.602176634e-19, "me": 9.1093837015e-31, "mp": 1.67262192369e-27,
    "epsilon0": 8.8541878128e-12, "mu0": 1.25663706212e-6,
    "hbar": 1.054571817e-34, "kb": 1.380649e-23,
    # volumes / misc as used in decks
    "cc": 1e-6, "cc3": 1e-18, "vol": 1.0,
    "pi": math.pi, "pi_2": math.pi / 2, "twopi": 2 * math.pi,
    "eV": 1.602176634e-19, "keV": 1.602176634e-16, "MeV": 1.602176634e-13,
    "GeV": 1.602176634e-10, "TeV": 1.602176634e-7,
    "W": 1.0, "GW": 1e9, "TW": 1e12,
}

_FUNCS = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10":
    math.log10, "abs": abs, "min": min, "max": max,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "floor": math.floor, "ceil": math.ceil,
    "gauss": lambda x, x0, w: math.exp(-((x - x0) / w) ** 2),
    "_if": lambda cond, a, b: a if cond else b,
}

_BLOCK_RE = re.compile(r"^begin:(\w+)\s*$", re.M)
_END_RE = re.compile(r"^end:(\w+)\s*$", re.M)


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _to_python(expr: str) -> str:
    """EPOCH expression -> python expression source (best effort)."""
    s = expr
    s = re.sub(r"\bgt\b", ">", s)
    s = re.sub(r"\blt\b", "<", s)
    s = re.sub(r"\bge\b", ">=", s)
    s = re.sub(r"\ble\b", "<=", s)
    s = re.sub(r"\beq\b", "==", s)
    s = re.sub(r"(?<![<>=!])=(?!=)", "==", s)  # only inside if(...) bodies
    s = s.replace("^", "**")
    s = re.sub(r"\bif\s*\(", "_if(", s)  # 'if' is a python keyword
    return s


def _eval(expr: str, env: dict) -> float | None:
    src = _to_python(expr).strip()
    if not src:
        return None
    scope = dict(UNITS)
    scope.update(_FUNCS)
    scope.update(env)
    try:
        val = eval(src, {"__builtins__": {}}, scope)  # noqa: S307
    except Exception:
        return None
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _blocks(text: str) -> dict[str, list[str]]:
    """Map block name -> body lines (repeated blocks accumulate, keyed
    name#k so callers can iterate all instances)."""
    out: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if m := re.match(r"^begin:(\w+)$", line):
            key = m.group(1)
            if key in out:
                i = 2
                while f"{key}#{i}" in out:
                    i += 1
                key = f"{key}#{i}"
            out[key] = []
            cur = out[key]
            continue
        if re.match(r"^end:\w+$", line):
            cur = None
            continue
        if cur is not None and line:
            cur.append(line)
    return out


# output-block variables -> per-particle variable count
PARTICLE_VARS = {
    "particles": 3, "px": 1, "py": 1, "pz": 1, "vx": 1, "vy": 1, "vz": 1,
    "particle_weight": 1, "id": 1, "charge": 1, "mass": 1,
}
FIELD_VARS = {
    "ex", "ey", "ez", "bx", "by", "bz", "jx", "jy", "jz",
    "charge_density", "number_density", "mass_density", "temperature",
    "ekbar", "ekflux", "poynt_flux", "rho", "wbar", "reg", "mask",
}
_VALUE_TOKENS = {"always", "never", "single", "all", "downsample", "first",
                 "average", "slice", "x", "y", "z", "t", "auto"}
_OUTPUT_META = {"dt_snapshot", "dump_at_times", "dump_final", "dt_output",
                "restart_at_times", "dt_restart", "nsteps"}


def _parse_output_line(line: str):
    """'px = single + testparticle' -> (var, subset_or_None, always?)."""
    if "=" not in line:
        return None
    var, _, rhs = line.partition("=")
    var = var.strip().lower()
    toks = [t.strip().lower() for t in rhs.replace("+", " ").split()]
    toks = [t for t in toks if t]
    if not toks:
        return None
    subset = next((t for t in toks if t not in _VALUE_TOKENS), None)
    return var, subset, "never" not in toks


class EpochDeck:
    """Parsed deck: numeric env + structured sizing-relevant facts."""

    def __init__(self, path: str):
        self.path = path
        raw = _strip_comments(open(path).read())
        self.bl = _blocks(raw)
        self.env: dict[str, float] = {"ndims": 3.0}
        self.warnings: list[str] = []
        self._eval_assignments(self.bl.get("constant", []))
        self._eval_assignments(self.bl.get("control", []))

    def _eval_assignments(self, lines: list[str]) -> None:
        for line in lines:
            if "=" not in line or ":" in line.split("=")[0]:
                continue
            name, _, expr = line.partition("=")
            name = name.strip()
            if not re.fullmatch(r"\w+", name):
                continue
            val = _eval(expr, self.env)
            if val is not None:
                self.env[name] = val

    def control(self, key: str) -> float | None:
        return self.env.get(key)

    def species(self) -> list[Species]:
        out = []
        for key, lines in self.bl.items():
            if not key.startswith("species"):
                continue
            sp = Species(name=f"species{len(out)}")
            for line in lines:
                p = _parse_kv(line)
                if not p:
                    continue
                k, v = p
                if k == "name":
                    sp.name = str(v)
                elif k == "npart_per_cell":
                    sp.npart_per_cell = _eval(str(v), self.env)
                elif k == "npart":
                    sp.npart = _eval(str(v), self.env)
                elif k == "tracer":
                    sp.tracer = str(v).upper().startswith("T")
            if sp.npart is None and sp.npart_per_cell is None:
                sp.npart_per_cell = 0.0
                self.warnings.append(
                    f"species '{sp.name}': no npart/npart_per_cell found")
            out.append(sp)
        return out

    def subsets(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key, lines in self.bl.items():
            if not key.startswith("subset"):
                continue
            name = None
            members: list[str] = []
            for line in lines:
                if line.lower().startswith("name"):
                    _, _, name = line.partition("=")
                    name = name.strip()
                elif line.lower().startswith("include_species"):
                    _, _, sp = line.partition(":")
                    members.append(sp.strip())
            if name:
                out[name] = members
        return out

    def output_blocks(self, species: list[Species],
                      n_dumps_default: int) -> list[OutputBlock]:
        subs = self.subsets()
        all_species = {s.name for s in species}
        out: list[OutputBlock] = []
        for key, lines in self.bl.items():
            if not re.fullmatch(r"output(?:#\d+)?", key):
                continue
            ob = OutputBlock(name="")
            for line in lines:
                if re.match(r"^name\s*=", line, re.I):
                    ob.name = line.partition("=")[2].strip()
                    continue
                p = _parse_output_line(line)
                if not p:
                    continue
                var, subset, included = p
                if var in _OUTPUT_META or not included:
                    continue
                if var == "grid":
                    ob.grid = True
                elif var in FIELD_VARS:
                    ob.field_vars.append(var)
                elif var in PARTICLE_VARS:
                    if subset and subset in subs:
                        targets = [s for s in subs[subset] if s in
                                   all_species]
                    else:
                        targets = sorted(all_species)
                    for sp in targets:
                        ob.particle_vars[sp] = (
                            ob.particle_vars.get(sp, 0) + PARTICLE_VARS[var])
                else:
                    self.warnings.append(
                        f"output '{ob.name or key}': variable '{var}' not "
                        "modeled, counted as zero bytes")
            if not ob.name:
                ob.name = key.replace("#", "")
            # dumps
            dt = None
            for line in lines:
                if line.lower().startswith("dt_snapshot"):
                    dt = _eval(line.partition("=")[2], self.env)
            dat = None
            for line in lines:
                if line.lower().startswith("dump_at_times"):
                    rhs = line.partition("=")[2]
                    dat = [v for v in (_eval(x, self.env)
                                       for x in rhs.split(",")) if v]
            ob.dt_snapshot = dt
            if dat:
                ob.n_dumps = len(dat)
            elif dt and self.env.get("t_end"):
                ob.n_dumps = int(self.env["t_end"] / dt)
            else:
                ob.n_dumps = n_dumps_default
            if any(re.match(r"^dump_final\s*=\s*T", l, re.I) for l in lines):
                ob.n_dumps += 1
            out.append(ob)
        return out

    def to_spec(self) -> WorkloadSpec:
        env = self.env
        nx, ny, nz = env.get("nx"), env.get("ny"), env.get("nz")
        if nx is None:
            raise SystemExit("slurmify: deck has no numeric nx in "
                             "begin:control - cannot size it")
        ny = ny if ny is not None else 1
        nz = nz if nz is not None else 1
        spec = WorkloadSpec(code="epoch", deck=self.path,
                            nx=int(round(nx)), ny=int(round(ny)),
                            nz=int(round(nz)),
                            t_end=env.get("t_end", 0.0),
                            cfl=env.get("cfl", 0.95))
        spec.ndims = 3 if nz > 1 else (2 if ny > 1 else 1)
        names = ("dx", "dy", "dz")
        for d, (n, lo, hi) in enumerate(
                ((spec.nx, env.get("x_min"), env.get("x_max")),
                 (spec.ny, env.get("y_min"), env.get("y_max")),
                 (spec.nz, env.get("z_min"), env.get("z_max")))):
            step = ((hi - lo) / n) if None not in (lo, hi) and n else 0.0
            setattr(spec, names[d], step)
        if all(getattr(spec, a) for a in names):
            inv = (spec.dx ** -2 + spec.dy ** -2 + spec.dz ** -2) ** 0.5
            spec.dt = spec.cfl / (UNITS["c"] * inv)
            spec.steps = int(spec.t_end / spec.dt) if spec.dt else 0
        else:
            self.warnings.append("grid extents unknown: dt/steps = 0")
        spec.species = self.species()
        spec.blocks = self.output_blocks(spec.species, n_dumps_default=1)
        spec.moving_window = any(
            re.search(r"move_window\s*=\s*T", l, re.I)
            for l in self.bl.get("window", []))
        spec.notes = self.warnings + [
            f"deck constants resolved: {len(env)} numeric symbols"]
        return spec


def _parse_kv(line: str):
    if "=" not in line:
        return None
    k, _, v = line.partition("=")
    k = k.strip().lower()
    v = v.strip()
    try:
        return k, float(v)
    except ValueError:
        return k, v
