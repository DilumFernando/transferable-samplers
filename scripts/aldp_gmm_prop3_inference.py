#!/usr/bin/env python3
"""Generate Ace-A-Nme samples from a trained GMM + Proposition-3 model.

This entry point does not fit a GMM and does not run an optimizer. It loads
both fitted-GMM parameters and the final component-matched EGNN drift from a
completed training run, then runs only the Proposition-3 sampler and optional
repository evaluation.

Examples:
    python scripts/aldp_gmm_prop3_inference.py \
        --trained-run-dir outputs/aldp_sbg_vs_prop3/runs/full_seed4201

    python scripts/aldp_gmm_prop3_inference.py \
        --trained-run-dir outputs/aldp_sbg_vs_prop3/runs/full_seed4201 \
        --num-particles 20000 --samples-only
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

from aldp_sbg_vs_gmm_prop3 import (
    REPO_ROOT,
    ExperimentConfig,
    ProjectedEGNNVelocity,
    Prop3Experiment,
    evaluate_and_save,
    jsonable,
    prepare_data,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse inference and sampling overrides."""
    parser = argparse.ArgumentParser(
        description="Sample ALDP with a trained full-covariance GMM + Proposition-3 drift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trained-run-dir",
        type=Path,
        required=True,
        help="Completed run containing checkpoints/gmm_parameters.pt and drift_final.pt.",
    )
    parser.add_argument("--gmm-checkpoint", type=Path, default=None)
    parser.add_argument("--drift-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "aldp_gmm_prop3_inference",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--annealing-steps", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--ess-threshold", type=float, default=None)
    parser.add_argument("--hutchinson-samples", type=int, default=None)
    parser.add_argument("--drift-batch", type=int, default=None)
    parser.add_argument("--energy-batch", type=int, default=None)
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Skip evaluator metrics/plots and save only weighted and final-resampled particles.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """Resolve and validate the requested Torch device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def load_training_config(
    trained_run_dir: Path,
    drift_checkpoint: dict[str, Any],
) -> ExperimentConfig:
    """Recover the training configuration from checkpoint or run metadata."""
    payload = drift_checkpoint.get("config")
    if payload is None:
        config_path = trained_run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                "No embedded checkpoint config or run config.json was found: "
                f"{config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    # These orchestration-only fields were added after the first Method-2
    # checkpoints and do not affect the trained GMM or velocity.
    payload = {
        "method2_only": True,
        "run_official_sbg": False,
        "run_official_ecnf": False,
        **payload,
    }
    field_names = {field.name for field in fields(ExperimentConfig)}
    missing = sorted(field_names.difference(payload))
    if missing:
        raise KeyError(f"Saved training configuration is missing fields: {missing}")
    return ExperimentConfig(**{name: payload[name] for name in field_names})


def inference_config(
    trained: ExperimentConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> ExperimentConfig:
    """Apply only inference-time overrides to the saved training config."""
    overrides: dict[str, Any] = {
        "device": str(device),
        "method2_only": True,
        "run_official_sbg": False,
        "run_official_ecnf": False,
    }
    optional = {
        "seed": args.seed,
        "num_particles": args.num_particles,
        "num_annealing_steps": args.annealing_steps,
        "epsilon": args.epsilon,
        "ess_threshold": args.ess_threshold,
        "hutchinson_samples": args.hutchinson_samples,
        "drift_batch": args.drift_batch,
        "energy_batch": args.energy_batch,
    }
    overrides.update({name: value for name, value in optional.items() if value is not None})
    config = replace(trained, **overrides)
    positive = {
        "num_particles": config.num_particles,
        "num_annealing_steps": config.num_annealing_steps,
        "hutchinson_samples": config.hutchinson_samples,
        "drift_batch": config.drift_batch,
        "energy_batch": config.energy_batch,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"These inference settings must be positive: {invalid}")
    if config.epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not 0 < config.ess_threshold <= 1:
        raise ValueError("ess_threshold must lie in (0, 1]")
    return config


def load_gmm(experiment: Prop3Experiment, checkpoint_path: Path) -> None:
    """Load the normalized full-covariance GMM without refitting it."""
    payload = torch.load(checkpoint_path, map_location=experiment.device, weights_only=False)
    required = {"weights", "means", "covariances"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"GMM checkpoint {checkpoint_path} is missing: {missing}")
    weights = payload["weights"].to(device=experiment.device, dtype=experiment.dtype)
    means = payload["means"].to(device=experiment.device, dtype=experiment.dtype)
    covariances = payload["covariances"].to(
        device=experiment.device, dtype=experiment.dtype
    )
    components = experiment.config.num_components
    if weights.shape != (components,):
        raise ValueError(
            f"GMM has {len(weights)} components, but saved config expects {components}"
        )
    if means.shape != (components, experiment.dim):
        raise ValueError(
            f"GMM mean shape {tuple(means.shape)} does not match "
            f"({components}, {experiment.dim})"
        )
    if covariances.shape != (components, experiment.dim, experiment.dim):
        raise ValueError(
            f"GMM covariance shape {tuple(covariances.shape)} is incompatible with "
            f"dimension {experiment.dim}"
        )
    tensors = (weights, means, covariances)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError(f"GMM checkpoint contains non-finite values: {checkpoint_path}")
    if bool((weights <= 0).any()):
        raise ValueError("Every saved GMM weight must be positive")
    experiment.gmm_weights = weights / weights.sum()
    experiment.gmm_means = means
    experiment.gmm_covariances = covariances
    experiment.gmm_cholesky = torch.linalg.cholesky(covariances)
    experiment.gmm_precision = torch.cholesky_inverse(experiment.gmm_cholesky)
    experiment.gmm_logdet = 2 * torch.log(
        torch.diagonal(experiment.gmm_cholesky, dim1=-2, dim2=-1)
    ).sum(-1)
    experiment.gmm_summary = payload.get("summary", {})
    experiment.component_pools = []
    print(
        "Loaded fitted GMM weights:",
        np.round(experiment.gmm_weights.cpu().numpy(), 5),
        flush=True,
    )


def load_drift(
    experiment: Prop3Experiment,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Instantiate the EGNN velocity and load its trained state."""
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Drift checkpoint has no model_state_dict: {checkpoint_path}")
    experiment.drift = ProjectedEGNNVelocity(
        experiment.num_atoms,
        experiment.spatial_dim,
        experiment.basis,
    ).to(experiment.device)
    experiment.drift.load_state_dict(checkpoint["model_state_dict"], strict=True)
    experiment.drift.eval()
    for parameter in experiment.drift.parameters():
        parameter.requires_grad_(False)
    print(
        f"Loaded drift at training step {checkpoint.get('step', 'unknown')}: "
        f"{checkpoint_path}",
        flush=True,
    )


def save_samples_only(
    run_dir: Path,
    method2: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Save the mandatory final resample and a lightweight summary."""
    generator = torch.Generator(device=method2["logw"].device).manual_seed(config.seed + 3)
    index = torch.multinomial(
        torch.softmax(method2["logw"], dim=0),
        len(method2["logw"]),
        replacement=True,
        generator=generator,
    )
    torch.save(
        {
            "samples": method2["samples"][index].cpu(),
            "target_energy": method2["target_energy"][index].cpu(),
            "pre_resampling_logw": method2["logw"].cpu(),
            "resampling_index": index.cpu(),
        },
        run_dir / "samples" / "gmm_prop3_final_resampled.pt",
    )
    summary = {
        "final_segment_ess_over_n": Prop3Experiment.normalized_ess(method2["logw"]),
        "intermediate_resampling_events": method2["num_resamples"],
        "mandatory_final_resampling": True,
        "particles": len(method2["samples"]),
        "evaluation_skipped": True,
    }
    write_json(run_dir / "metrics" / "summary.json", summary)
    return summary


def main() -> None:
    """Load a trained Method-2 model and run inference only."""
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=True)
    trained_run_dir = args.trained_run_dir.expanduser().resolve()
    if not trained_run_dir.is_dir():
        raise FileNotFoundError(f"Trained run directory does not exist: {trained_run_dir}")
    gmm_path = (
        args.gmm_checkpoint.expanduser().resolve()
        if args.gmm_checkpoint is not None
        else trained_run_dir / "checkpoints" / "gmm_parameters.pt"
    )
    drift_path = (
        args.drift_checkpoint.expanduser().resolve()
        if args.drift_checkpoint is not None
        else trained_run_dir / "checkpoints" / "drift_final.pt"
    )
    for label, path in (("GMM", gmm_path), ("drift", drift_path)):
        if not path.exists():
            raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")

    device = resolve_device(args.device)
    drift_checkpoint = torch.load(drift_path, map_location="cpu", weights_only=False)
    trained_config = load_training_config(trained_run_dir, drift_checkpoint)
    config = inference_config(trained_config, args, device)
    if args.scratch_dir is not None:
        scratch_dir = args.scratch_dir.expanduser().resolve()
        os.environ["SCRATCH_DIR"] = str(scratch_dir)
    elif "SCRATCH_DIR" in os.environ:
        scratch_dir = Path(os.environ["SCRATCH_DIR"]).expanduser().resolve()
    else:
        scratch_dir = REPO_ROOT / "scratch"
        os.environ["SCRATCH_DIR"] = str(scratch_dir)
        print(f"SCRATCH_DIR was not set; using ignored local cache: {scratch_dir}", flush=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_float32_matmul_precision("highest")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"inference_seed{config.seed}_{timestamp}"
    run_dir = args.output_dir.expanduser().resolve() / "runs" / run_name
    for directory in ("checkpoints", "metrics", "plots", "samples"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    datamodule, eval_context, reconstructed_train_y, reconstructed_basis = prepare_data(
        scratch_dir, device
    )
    saved_basis = drift_checkpoint.get("model_state_dict", {}).get("basis")
    if saved_basis is None:
        print(
            "Warning: checkpoint has no basis buffer; using the reconstructed mean-free basis.",
            flush=True,
        )
        basis = reconstructed_basis
    else:
        basis = saved_basis.to(device=device, dtype=reconstructed_basis.dtype)
        if basis.shape != reconstructed_basis.shape:
            raise ValueError(
                f"Saved basis shape {tuple(basis.shape)} does not match expected "
                f"{tuple(reconstructed_basis.shape)}"
            )
    # The saved basis is authoritative because the GMM was fitted in that basis.
    train_y = torch.empty(
        (1, basis.shape[1]),
        device=device,
        dtype=reconstructed_train_y.dtype,
    )
    experiment = Prop3Experiment(
        config, run_dir, train_y, basis, datamodule, eval_context
    )
    load_gmm(experiment, gmm_path)
    load_drift(experiment, drift_checkpoint, drift_path)

    provenance = {
        **asdict(config),
        "scratch_dir": scratch_dir,
        "run_dir": run_dir,
        "trained_run_dir": trained_run_dir,
        "gmm_checkpoint": gmm_path,
        "drift_checkpoint": drift_path,
        "drift_training_step": drift_checkpoint.get("step"),
        "samples_only": args.samples_only,
    }
    write_json(run_dir / "config.json", provenance)
    print(json.dumps(jsonable(provenance), indent=2, sort_keys=True), flush=True)

    method2 = experiment.run_prop3(resample=True)
    if args.samples_only:
        summary = save_samples_only(run_dir, method2, config)
    else:
        summary = evaluate_and_save(
            run_dir,
            method2,
            eval_context,
            datamodule,
            config,
            None,
            None,
            None,
        )
    print(
        "Inference summary:",
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        flush=True,
    )
    print(f"All inference artifacts saved under: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
