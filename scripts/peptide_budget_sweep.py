#!/usr/bin/env python3
"""Does SNIS catch up with SBG's annealed inference when given the same energy budget?

The paper's tables compare SNIS, AIS and SMC at the same particle count (10^4), but the annealed
variants additionally run `num_annealing_steps` Langevin steps per particle, each evaluating the
target energy and its gradient. SNIS at 10^4 samples therefore spends ~100x fewer target energy
evaluations. This script sweeps the SNIS sample count on one peptide and reports every run against
its target-energy-evaluation count:

    SNIS:  evaluations = num_samples
    SMC:   evaluations = num_samples x (num_annealing_steps + 1)

so SNIS at 10^6 is budget-matched to the published SMC/AIS numbers (10^4 particles, 100 steps),
which can be read off Tables 2, 3, 9 and 10 rather than rerun. SMC is available with
`--samplers snis smc` for a within-pipeline comparison, but is not run by default.

Metrics are unchanged by the sweep size: the evaluator uses at most 10,000 conformations per metric
(PeptideEnsembleEvaluator.num_eval_samples), so more samples improve the weighted resample rather
than the metric sample size.

Each run is one `src/transferable_samplers/eval.py` invocation with hydra overrides, writing to its
own directory; metrics are read back from the CSV logger. Nothing is trained: every run reuses the
published TarFlow weights for that system.

    python scripts/peptide_budget_sweep.py --system AAA --dry-run          # print the commands
    python scripts/peptide_budget_sweep.py --system Ace-A-Nme --seeds 0 1 2
    python scripts/peptide_budget_sweep.py --system AAAAAA --collect       # re-read finished runs

Systems: Ace-A-Nme (alanine dipeptide), AAA (trialanine), Ace-AAA-Nme (alanine tetrapeptide),
AAAAAA (hexa-alanine). Results: <out>/<run tag>/ per run, and <out>/summary.csv over all of them.
Prerequisites: the system's dataset under paths.data_dir, the HuggingFace TarFlow weights, and
SCRATCH_DIR (shell or .env), i.e. whatever the stock eval configs already need.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL = REPO_ROOT / "src" / "transferable_samplers" / "eval.py"
SYSTEMS = ("Ace-A-Nme", "AAA", "Ace-AAA-Nme", "AAAAAA")     # dipeptide, tri-, tetra-, hexa-
EXPERIMENT = "single_system/eval/tarflow_{system}_{variant}"  # variant: snis, or ula for ais/smc
# The annealed arms share one config: SMC resamples when normalized ESS falls below the threshold,
# AIS is the same run with intermediate resampling disabled (negative threshold), as in the paper.
ESS_THRESHOLD = {"smc": 0.5, "ais": -1.0}
# metrics worth keeping; the evaluator prefixes them with "test/<sequence>/<sample set>/".
# The paper's T-W2 is logged as "torus-w2"; "energy-w1" is the tail-sensitive companion to energy-w2.
METRICS = ("energy-w2", "energy-w1", "torus-w2", "torus-k-jsd", "tica-w2", "tica-k-jsd",
           "effective-sample-size", "num-eval-samples")


def subprocess_env() -> dict[str, str]:
    """eval.py is run as a script, so make `src` importable whether or not the package is installed."""
    src = str(REPO_ROOT / "src")
    existing = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONPATH": src + (os.pathsep + existing if existing else "")}


def check_scratch_dir() -> None:
    """hydra resolves paths from SCRATCH_DIR, set in the shell or in the repo's .env (see .env.example)."""
    if (REPO_ROOT / ".env").exists() or os.environ.get("SCRATCH_DIR"):
        return
    sys.exit("SCRATCH_DIR is not set and no .env exists: `cp .env.example .env` and edit it, or "
             "`export SCRATCH_DIR=<dir holding transferable-samplers/many-peptides-md>`. "
             "Note eval.py loads .env with override=True, so .env wins over the shell.")


def run_tag(sampler: str, size: int, steps: int, seed: int, model_seed: int = 0) -> str:
    """Model seed 0 keeps its plain tag, so runs made before --model-seeds existed still match."""
    return (f"{sampler}_n{size}" + (f"_s{steps}" if sampler in ESS_THRESHOLD else "")
            + f"_seed{seed}" + (f"_m{model_seed}" if model_seed else ""))


def command(system: str, sampler: str, size: int, steps: int, seed: int, model_seed: int,
            out: Path, extra: list[str]) -> list[str]:
    experiment = EXPERIMENT.format(system=system, variant="snis" if sampler == "snis" else "ula")
    cmd = [sys.executable, str(EVAL), f"experiment={experiment}", "logger=csv",
           f"seed={seed}", f"callbacks.sampling_evaluation.sampler.num_samples={size}",
           f"hf_state_dict_path=single_system/tarflow_{system}_{model_seed}.pth",
           f"hydra.run.dir={out / run_tag(sampler, size, steps, seed, model_seed)}"]
    if sampler in ESS_THRESHOLD:
        cmd += [f"callbacks.sampling_evaluation.sampler.num_annealing_steps={steps}",
                f"callbacks.sampling_evaluation.sampler.ess_threshold={ESS_THRESHOLD[sampler]}"]
    return cmd + extra


def target_evals(sampler: str, size: int, steps: int) -> int:
    """Target energy evaluations: one per sample for SNIS, one per particle per step when annealing."""
    return size if sampler == "snis" else size * (steps + 1)


def read_metrics(run_dir: Path) -> list[dict[str, str]]:
    """Read the CSV logger's metrics for one run (last row per column that has a value)."""
    files = sorted(run_dir.glob("csv/**/metrics.csv"))
    if not files:
        return []
    rows: dict[str, str] = {}
    for path in files:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                for key, value in row.items():
                    if value not in ("", None):
                        rows[key] = value
    return [rows]


TAG = re.compile(r"^(?P<sampler>snis|smc|ais)_n(?P<size>\d+)(?:_s(?P<steps>\d+))?"
                 r"_seed(?P<seed>\d+)(?:_m(?P<model_seed>\d+))?$")


def found_runs(out: Path) -> list[tuple[str, int, int, int, int]]:
    """Every finished run in `out`, read off the directory names rather than the current flags."""
    runs = []
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        m = TAG.match(d.name)
        if m and read_metrics(d):
            runs.append((m["sampler"], int(m["size"]), int(m["steps"] or 0),
                         int(m["seed"]), int(m["model_seed"] or 0)))
    return runs


def minutes(run_dir: Path) -> str:
    """Wall-clock from the hydra log's first and last timestamps."""
    log = run_dir / "eval.log"
    if not log.exists():
        return ""
    stamps = [line[1:20] for line in log.read_text().splitlines() if line.startswith("[20")]
    if len(stamps) < 2:
        return ""
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        a, b = (datetime.datetime.strptime(x, fmt) for x in (stamps[0], stamps[-1]))
    except ValueError:
        return ""
    return f"{(b - a).total_seconds() / 60:.1f}"


def collect(out: Path, jobs: list[tuple[str, int, int, int, int]] | None = None) -> None:
    """Summarise every finished run in `out`; `jobs` only adds "not finished" lines for expected runs."""
    if not out.is_dir():
        print(f"no results directory {out}")
        return
    runs = found_runs(out)
    for job in jobs or []:
        if job not in runs:
            print(f"  {run_tag(job[0], job[1], job[2], job[3], job[4])}: not finished")
    summary = []
    for sampler, size, steps, seed, model_seed in runs:
        metrics = read_metrics(out / run_tag(sampler, size, steps, seed, model_seed))[0]
        row = {"sampler": sampler, "num_samples": size, "steps": steps,
               "seed": seed, "model_seed": model_seed,
               "target_energy_evals": target_evals(sampler, size, steps),
               "minutes": minutes(out / run_tag(sampler, size, steps, seed, model_seed))}
        for key, value in metrics.items():
            if any(key.endswith(m) for m in METRICS):
                row[key.replace("test/", "")] = value
        summary.append(row)
    if not summary:
        print("nothing to summarise yet")
        return
    columns = sorted({k for row in summary for k in row}, key=lambda k: (k not in
                     ("sampler", "num_samples", "steps", "seed", "model_seed",
                      "target_energy_evals", "minutes"), k))
    path = out / "summary.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in summary)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in summary:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    print(f"\nwrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000],
                   help="SNIS sample counts to sweep")
    p.add_argument("--smc-sizes", type=int, nargs="+", default=[10_000],
                   help="particle counts for ais/smc (cost is this times the step count)")
    p.add_argument("--system", choices=SYSTEMS, default="Ace-A-Nme", help="which peptide to evaluate")
    p.add_argument("--samplers", nargs="+", choices=("snis", "ais", "smc"), default=["snis"],
                   help="ais/smc rerun the annealed arms in this pipeline (same config, "
                        "resampling disabled for ais)")
    p.add_argument("--steps", type=int, default=100, help="SMC annealing steps")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="sampling seeds (Monte Carlo noise)")
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0], choices=(0, 1, 2),
                   help="trained TarFlow seeds; the paper's error bars are over these three")
    p.add_argument("--out", help="default: results/<system>_budget_sweep (results/aldp_budget_sweep "
                                 "for Ace-A-Nme, where the first runs landed)")
    p.add_argument("--collect", action="store_true", help="only read finished runs and summarise")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="further hydra overrides, passed through verbatim (put this last)")
    args = p.parse_args()
    default_out = "aldp_budget_sweep" if args.system == "Ace-A-Nme" else f"{args.system}_budget_sweep"
    out = Path(args.out) if args.out else REPO_ROOT / "results" / default_out

    jobs = [(s, n, args.steps, seed, m) for m in args.model_seeds for seed in args.seeds
            for s in args.samplers for n in (args.sizes if s == "snis" else args.smc_sizes)]
    if args.collect:
        collect(out)                       # every finished run in the folder, whatever the flags
        return
    check_scratch_dir()
    out.mkdir(parents=True, exist_ok=True)
    for sampler, size, steps, seed, model_seed in jobs:
        tag = run_tag(sampler, size, steps, seed, model_seed)
        if (out / tag / "csv" / "version_0" / "metrics.csv").exists():
            print(f"[skip] {tag} already has results")
            continue
        cmd = command(args.system, sampler, size, steps, seed, model_seed, out, args.extra)
        print(f"[run ] {args.system} {tag}  ({target_evals(sampler, size, steps):,} target energy evaluations)")
        print("       " + " ".join(cmd))
        if args.dry_run:
            continue
        code = subprocess.run(cmd, cwd=REPO_ROOT, env=subprocess_env()).returncode
        if code != 0:
            print(f"[fail] {tag} exited with {code}")
    if not args.dry_run:
        collect(out, jobs)


if __name__ == "__main__":
    main()
