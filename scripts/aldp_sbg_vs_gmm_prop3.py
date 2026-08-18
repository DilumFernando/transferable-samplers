#!/usr/bin/env python3
"""Run the Ace-A-Nme SBG/ECNF++ versus GMM + Proposition-3 comparison.

The default invocation is a small smoke test. Pass ``--full`` for the
paper-scale particle count, annealing schedule, and drift-training budget.

Examples:
    python scripts/aldp_sbg_vs_gmm_prop3.py
    python scripts/aldp_sbg_vs_gmm_prop3.py --full
    python scripts/aldp_sbg_vs_gmm_prop3.py --full --run-official-baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from transferable_samplers.data.single_peptide_datamodule import SinglePeptideDataModule
from transferable_samplers.evaluation.evaluator import PeptideEnsembleEvaluator
from transferable_samplers.nn.egnn.egnn_dynamics_ad2_cat import EGNN_dynamics_AD2_cat
from transferable_samplers.utils.dataclasses import SamplesData
from transferable_samplers.utils.standardization import standardize_coords


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved experiment settings saved with every run."""

    smoke_test: bool
    seed: int
    device: str
    num_components: int
    gmm_reg_covar: float
    train_steps: int
    train_batch: int
    num_particles: int
    num_annealing_steps: int
    epsilon: float
    ess_threshold: float
    hutchinson_samples: int
    drift_batch: int
    energy_batch: int
    learning_rate: float
    sigma_min: float
    log_every: int
    checkpoint_every: int
    run_official_sbg: bool
    run_official_ecnf: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare official Ace-A-Nme SBG and ECNF++ with full-covariance GMM + Proposition 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--full", action="store_true", help="Use the full 100k-step, 10k-particle configuration.")
    parser.add_argument("--run-official-sbg", action="store_true", help="Run the official TarFlow + ULA-SMC baseline.")
    parser.add_argument("--run-official-ecnf", action="store_true", help="Run the official ECNF++ + SNIS baseline.")
    parser.add_argument(
        "--run-official-baselines",
        action="store_true",
        help="Run both official SBG and ECNF++ baselines.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "aldp_sbg_vs_prop3")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name below OUTPUT_DIR/runs; defaults to a timestamp.",
    )
    parser.add_argument("--scratch-dir", type=Path, default=None, help="Override SCRATCH_DIR from .env.")
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--num-components", type=int, default=6)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-4)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--train-batch", type=int, default=None)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--annealing-steps", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--hutchinson-samples", type=int, default=None)
    parser.add_argument("--drift-batch", type=int, default=None)
    parser.add_argument("--energy-batch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100, help="Print training loss every N optimizer steps.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a training checkpoint every N steps; 0 disables.",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace, device: torch.device) -> ExperimentConfig:
    """Resolve smoke/full defaults and explicit command-line overrides."""
    smoke = not args.full

    def choose(value: Any, smoke_value: Any, full_value: Any) -> Any:
        return value if value is not None else (smoke_value if smoke else full_value)

    config = ExperimentConfig(
        smoke_test=smoke,
        seed=args.seed,
        device=str(device),
        num_components=args.num_components,
        gmm_reg_covar=args.gmm_reg_covar,
        train_steps=choose(args.train_steps, 500, 100_000),
        train_batch=choose(args.train_batch, 128, 512),
        num_particles=choose(args.num_particles, 256, 10_000),
        num_annealing_steps=choose(args.annealing_steps, 20, 100),
        epsilon=args.epsilon,
        ess_threshold=args.ess_threshold,
        hutchinson_samples=choose(args.hutchinson_samples, 1, 4),
        drift_batch=choose(args.drift_batch, 32, 128),
        energy_batch=choose(args.energy_batch, 64, 256),
        learning_rate=args.learning_rate,
        sigma_min=args.sigma_min,
        log_every=args.log_every,
        checkpoint_every=choose(args.checkpoint_every, 0, 10_000),
        run_official_sbg=args.run_official_sbg or args.run_official_baselines,
        run_official_ecnf=args.run_official_ecnf or args.run_official_baselines,
    )
    positive_fields = {
        "num_components": config.num_components,
        "train_steps": config.train_steps,
        "train_batch": config.train_batch,
        "num_particles": config.num_particles,
        "num_annealing_steps": config.num_annealing_steps,
        "hutchinson_samples": config.hutchinson_samples,
        "drift_batch": config.drift_batch,
        "energy_batch": config.energy_batch,
        "log_every": config.log_every,
    }
    invalid = {name: value for name, value in positive_fields.items() if value <= 0}
    if invalid:
        raise ValueError(f"These settings must be positive: {invalid}")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if config.gmm_reg_covar <= 0 or config.learning_rate <= 0 or config.epsilon < 0:
        raise ValueError("gmm_reg_covar and learning_rate must be positive; epsilon must be non-negative")
    if not 0 < config.ess_threshold <= 1:
        raise ValueError("ess_threshold must lie in (0, 1]")
    return config


def jsonable(value: Any) -> Any:
    """Convert tensors, arrays, paths, and nested structures to JSON data."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    """Write a deterministic, human-readable JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save a figure as both PNG and PDF."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def safe_plot_name(title: str) -> str:
    """Turn an evaluator plot title into a filesystem-safe relative stem."""
    pieces = [re.sub(r"[^A-Za-z0-9_.-]+", "_", piece).strip("_") for piece in title.split("/")]
    return "/".join(piece for piece in pieces if piece) or "plot"


def samples_data_to_cpu(data: SamplesData) -> SamplesData:
    """Detach a SamplesData object and move it to CPU."""
    return SamplesData(
        samples=data.samples.detach().cpu(),
        E_target=data.E_target.detach().cpu(),
        logw=data.logw.detach().cpu() if data.logw is not None else None,
    )


def package_subprocess_env() -> dict[str, str]:
    """Return an environment in which subprocesses can import the local package."""
    environment = os.environ.copy()
    source_dir = str(REPO_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_dir + os.pathsep + existing_pythonpath if existing_pythonpath else source_dir
    )
    return environment


class ProjectedEGNNVelocity(nn.Module):
    """ECNF++ EGNN velocity projected into the mean-free coordinate basis."""

    def __init__(self, num_atoms: int, spatial_dim: int, basis: torch.Tensor) -> None:
        super().__init__()
        self.egnn = EGNN_dynamics_AD2_cat(
            num_atoms=num_atoms,
            num_dimensions=spatial_dim,
            channels=256,
            num_layers=5,
        )
        self.register_buffer("basis", basis)

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate mean-free velocity at coordinates y and time t."""
        x_flat = y @ self.basis.T
        velocity_x = self.egnn(t, x_flat)
        return velocity_x @ self.basis


class Prop3Experiment:
    """State and operations for the fitted-GMM Proposition-3 experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: Path,
        train_y: torch.Tensor,
        basis: torch.Tensor,
        datamodule: SinglePeptideDataModule,
        eval_context: Any,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.device = train_y.device
        self.dtype = train_y.dtype
        self.train_y = train_y
        self.basis = basis
        self.datamodule = datamodule
        self.eval_context = eval_context
        self.spatial_dim = 3
        self.num_atoms = basis.shape[0] // self.spatial_dim
        self.dim = train_y.shape[1]

        self.gmm_weights: torch.Tensor
        self.gmm_means: torch.Tensor
        self.gmm_covariances: torch.Tensor
        self.gmm_cholesky: torch.Tensor
        self.gmm_precision: torch.Tensor
        self.gmm_logdet: torch.Tensor
        self.component_pools: list[torch.Tensor]
        self.gmm_summary: dict[str, Any]
        self.drift: ProjectedEGNNVelocity

    def x_to_y(self, x: torch.Tensor) -> torch.Tensor:
        """Project centered Cartesian coordinates into the mean-free basis."""
        return x.reshape(len(x), -1) @ self.basis

    def y_to_x(self, y: torch.Tensor) -> torch.Tensor:
        """Map mean-free coordinates back to Cartesian point clouds."""
        return (y @ self.basis.T).reshape(len(y), self.num_atoms, self.spatial_dim)

    def fit_gmm(self) -> None:
        """Fit and save the unrestricted full-covariance GMM."""
        config = self.config
        print("Fitting full-covariance GMM...", flush=True)
        fit = GaussianMixture(
            n_components=config.num_components,
            covariance_type="full",
            reg_covar=config.gmm_reg_covar,
            n_init=5,
            max_iter=500,
            random_state=config.seed,
        ).fit(self.train_y.detach().cpu().double().numpy())

        self.gmm_weights = torch.as_tensor(fit.weights_, device=self.device, dtype=self.dtype)
        self.gmm_means = torch.as_tensor(fit.means_, device=self.device, dtype=self.dtype)
        self.gmm_covariances = torch.as_tensor(fit.covariances_, device=self.device, dtype=self.dtype)
        self.gmm_cholesky = torch.linalg.cholesky(self.gmm_covariances)
        self.gmm_precision = torch.cholesky_inverse(self.gmm_cholesky)
        self.gmm_logdet = 2 * torch.log(torch.diagonal(self.gmm_cholesky, dim1=-2, dim2=-1)).sum(-1)

        train_np = self.train_y.detach().cpu().numpy()
        train_component = torch.as_tensor(fit.predict(train_np), device=self.device)
        responsibility = torch.as_tensor(fit.predict_proba(train_np), device=self.device, dtype=self.dtype)
        self.component_pools = []
        for component in range(config.num_components):
            pool = torch.where(train_component == component)[0]
            if len(pool) == 0:
                pool = torch.topk(responsibility[:, component], k=min(64, len(self.train_y))).indices
            self.component_pools.append(pool)

        condition_numbers = torch.linalg.cond(self.gmm_covariances)
        assigned_counts = torch.bincount(train_component, minlength=config.num_components)
        self.gmm_summary = {
            "weights": self.gmm_weights,
            "assigned_counts": assigned_counts,
            "covariance_condition_numbers": condition_numbers,
            "converged": fit.converged_,
            "n_iter": fit.n_iter_,
            "lower_bound": fit.lower_bound_,
        }
        print("Fitted weights:", np.round(fit.weights_, 5), flush=True)
        print("Assigned counts:", assigned_counts.cpu().numpy(), flush=True)
        print("Covariance condition numbers:", np.round(condition_numbers.cpu().numpy(), 2), flush=True)

        torch.save(
            {
                "weights": self.gmm_weights.cpu(),
                "means": self.gmm_means.cpu(),
                "covariances": self.gmm_covariances.cpu(),
                "summary": jsonable(self.gmm_summary),
            },
            self.run_dir / "checkpoints" / "gmm_parameters.pt",
        )
        write_json(self.run_dir / "metrics" / "gmm_summary.json", self.gmm_summary)

    def sample_gmm(
        self,
        n: int,
        generator: torch.Generator | None = None,
        return_component: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample from the fitted full-covariance GMM."""
        component = torch.multinomial(self.gmm_weights, n, replacement=True, generator=generator)
        noise = torch.randn(n, self.dim, device=self.device, generator=generator)
        y = self.gmm_means[component] + torch.bmm(
            self.gmm_cholesky[component], noise.unsqueeze(-1)
        ).squeeze(-1)
        return (y, component) if return_component else y

    def gmm_energy_score(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate fitted-GMM energy and score."""
        delta = y[:, None, :] - self.gmm_means[None, :, :]
        precision_delta = torch.einsum("kij,bkj->bki", self.gmm_precision, delta)
        mahalanobis = torch.sum(delta * precision_delta, dim=-1)
        component_logp = (
            torch.log(self.gmm_weights)[None, :]
            - 0.5 * (self.dim * math.log(2 * math.pi) + self.gmm_logdet[None, :] + mahalanobis)
        )
        responsibility = torch.softmax(component_logp, dim=1)
        score = torch.sum(responsibility[..., None] * (-precision_delta), dim=1)
        return -torch.logsumexp(component_logp, dim=1), score

    def save_training_checkpoint(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
        loss_history: list[float],
        name: str | None = None,
    ) -> Path:
        """Save drift, optimizer, RNG, and loss state."""
        filename = name or f"drift_step_{step:07d}.pt"
        path = self.run_dir / "checkpoints" / filename
        torch.save(
            {
                "step": step,
                "model_state_dict": self.drift.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "loss_history": loss_history,
                "config": asdict(self.config),
            },
            path,
        )
        return path

    def train_drift(self, resume_checkpoint: Path | None = None) -> list[float]:
        """Train the component-matched ECNF++ velocity field."""
        config = self.config
        self.drift = ProjectedEGNNVelocity(self.num_atoms, self.spatial_dim, self.basis).to(self.device)
        optimizer = torch.optim.AdamW(
            self.drift.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )
        generator = torch.Generator(device=self.device).manual_seed(config.seed + 1)
        loss_history: list[float] = []
        start_step = 0
        if resume_checkpoint is not None:
            if not resume_checkpoint.exists():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=self.device, weights_only=False)
            self.drift.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            generator.set_state(checkpoint["generator_state"].cpu())
            loss_history = list(checkpoint["loss_history"])
            start_step = int(checkpoint["step"])
            if start_step > config.train_steps:
                raise ValueError(
                    f"Checkpoint step {start_step} exceeds requested train_steps={config.train_steps}"
                )
            print(f"Resuming drift training at step {start_step} from {resume_checkpoint}", flush=True)

        self.drift.train()
        print(f"Training drift for steps {start_step + 1}..{config.train_steps}...", flush=True)
        for step in range(start_step, config.train_steps):
            sampled = self.sample_gmm(config.train_batch, generator=generator, return_component=True)
            assert isinstance(sampled, tuple)
            y0, component = sampled
            y1 = torch.empty_like(y0)
            for component_index, pool in enumerate(self.component_pools):
                mask = component == component_index
                count = int(mask.sum())
                if count:
                    chosen = pool[
                        torch.randint(len(pool), (count,), device=self.device, generator=generator)
                    ]
                    y1[mask] = self.train_y[chosen]

            time = torch.rand(config.train_batch, 1, device=self.device, generator=generator)
            yt = (1 - (1 - config.sigma_min) * time) * y0 + time * y1
            target_velocity = y1 - (1 - config.sigma_min) * y0
            loss = (self.drift(yt, time[:, 0]) - target_velocity).square().mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.drift.parameters(), 10.0)
            optimizer.step()
            loss_history.append(float(loss.detach()))

            completed = step + 1
            if completed % config.log_every == 0 or completed == config.train_steps:
                print(f"train {completed}/{config.train_steps}: loss={loss_history[-1]:.6f}", flush=True)
            if config.checkpoint_every and completed % config.checkpoint_every == 0:
                checkpoint_path = self.save_training_checkpoint(
                    completed,
                    optimizer,
                    generator,
                    loss_history,
                )
                print(f"Saved checkpoint: {checkpoint_path}", flush=True)

        self.drift.eval()
        self.save_training_checkpoint(
            config.train_steps,
            optimizer,
            generator,
            loss_history,
            name="drift_final.pt",
        )
        np.save(self.run_dir / "metrics" / "loss_history.npy", np.asarray(loss_history))
        self.plot_loss(loss_history)
        return loss_history

    def plot_loss(self, loss_history: list[float]) -> None:
        """Save the flow-matching loss curve."""
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(np.arange(1, len(loss_history) + 1), loss_history)
        ax.set_yscale("log")
        ax.set(xlabel="training step", ylabel="flow-matching MSE", title="Component-matched EGNN drift")
        save_figure(fig, self.run_dir / "plots" / "training" / "flow_matching_loss")

    @torch.no_grad()
    def target_energy_score(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate OpenMM target energy and score in batches."""
        energies, scores = [], []
        for start in range(0, len(y), self.config.energy_batch):
            y_batch = y[start : start + self.config.energy_batch]
            x_batch = self.y_to_x(y_batch)
            energy, gradient_x = self.eval_context.target_energy.energy_and_grad(x_batch)
            energies.append(energy.to(self.device))
            scores.append(-(gradient_x.to(self.device).reshape(len(y_batch), -1) @ self.basis))
        return torch.cat(energies), torch.cat(scores)

    def drift_and_divergence(
        self,
        y: torch.Tensor,
        time: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate drift and its Hutchinson divergence estimate in batches."""
        velocity_parts, divergence_parts = [], []
        for start in range(0, len(y), self.config.drift_batch):
            leaf = y[start : start + self.config.drift_batch].detach().requires_grad_(True)
            with torch.enable_grad():
                velocity = self.drift(leaf, time)
                divergence = torch.zeros(len(leaf), device=self.device)
                for probe_index in range(self.config.hutchinson_samples):
                    probe = torch.empty_like(leaf).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
                    vector_jacobian = torch.autograd.grad(
                        (velocity * probe).sum(),
                        leaf,
                        retain_graph=probe_index + 1 < self.config.hutchinson_samples,
                    )[0]
                    divergence += (vector_jacobian * probe).sum(1) / self.config.hutchinson_samples
            velocity_parts.append(velocity.detach())
            divergence_parts.append(divergence.detach())
        return torch.cat(velocity_parts), torch.cat(divergence_parts)

    @staticmethod
    def normalized_ess(logw: torch.Tensor) -> float:
        """Compute ESS divided by particle count."""
        weights = torch.softmax(logw, dim=0)
        return float(1 / (len(weights) * weights.square().sum()))

    def run_prop3(self, resample: bool = True) -> dict[str, Any]:
        """Run the Proposition-3 annealed sampler."""
        config = self.config
        generator = torch.Generator(device=self.device).manual_seed(config.seed + 2)
        sampled = self.sample_gmm(config.num_particles, generator=generator, return_component=True)
        assert isinstance(sampled, tuple)
        y, ancestry = sampled
        logw = torch.zeros(config.num_particles, device=self.device)
        dt = 1.0 / config.num_annealing_steps
        diagnostics: dict[str, list[Any]] = {
            "t": [],
            "ess": [],
            "resampled": [],
            "raw_component_mass": [],
        }
        num_resamples = 0

        print("Running Proposition-3 sampler...", flush=True)
        for step in range(config.num_annealing_steps):
            time = torch.tensor((step + 0.5) * dt, device=self.device)
            U0, score0 = self.gmm_energy_score(y)
            U1, score1 = self.target_energy_score(y)
            velocity, divergence = self.drift_and_divergence(y, time, generator)
            grad_Ut = -((1 - time) * score0 + time * score1)

            logw += (divergence - (grad_Ut * velocity).sum(1) + U0 - U1) * dt
            y += (-config.epsilon * grad_Ut + velocity) * dt
            y += math.sqrt(2 * config.epsilon * dt) * torch.randn(
                y.shape,
                device=self.device,
                generator=generator,
            )

            current_ess = self.normalized_ess(logw)
            did_resample = bool(resample and current_ess < config.ess_threshold)
            if did_resample:
                index = torch.multinomial(
                    torch.softmax(logw, 0),
                    len(logw),
                    replacement=True,
                    generator=generator,
                )
                y = y[index]
                ancestry = ancestry[index]
                logw.zero_()
                num_resamples += 1

            diagnostics["t"].append(float((step + 1) * dt))
            diagnostics["ess"].append(current_ess)
            diagnostics["resampled"].append(did_resample)
            diagnostics["raw_component_mass"].append(
                (torch.bincount(ancestry, minlength=config.num_components).float() / len(ancestry)).cpu()
            )
            if not torch.isfinite(y).all() or not torch.isfinite(logw).all():
                raise FloatingPointError(f"Non-finite Proposition-3 state at step {step + 1}")
            print(
                f"prop3 {step + 1:3d}/{config.num_annealing_steps}: "
                f"ESS/N={current_ess:.4f}, resampled={did_resample}",
                flush=True,
            )

        target_energy, _ = self.target_energy_score(y)
        result = {
            "y": y.detach(),
            "samples": self.y_to_x(y).detach(),
            "target_energy": target_energy.detach(),
            "logw": logw.detach(),
            "diagnostics": diagnostics,
            "num_resamples": num_resamples,
        }
        torch.save(
            {
                "samples": result["samples"].cpu(),
                "target_energy": result["target_energy"].cpu(),
                "logw": result["logw"].cpu(),
                "diagnostics": diagnostics,
                "num_resamples": num_resamples,
            },
            self.run_dir / "samples" / "gmm_prop3.pt",
        )
        self.save_prop3_diagnostics(result)
        return result

    def save_prop3_diagnostics(self, result: dict[str, Any]) -> None:
        """Save ESS and component-mass diagnostics as JSON and CSV."""
        diagnostics = result["diagnostics"]
        write_json(self.run_dir / "metrics" / "prop3_diagnostics.json", diagnostics)
        component_mass = torch.stack(diagnostics["raw_component_mass"]).numpy()
        csv_path = self.run_dir / "metrics" / "prop3_trajectory.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["step", "t", "ess_over_n", "resampled"] + [
                f"component_{index}_mass" for index in range(self.config.num_components)
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, (time, ess, resampled) in enumerate(
                zip(diagnostics["t"], diagnostics["ess"], diagnostics["resampled"], strict=True)
            ):
                row: dict[str, Any] = {
                    "step": index + 1,
                    "t": time,
                    "ess_over_n": ess,
                    "resampled": resampled,
                }
                row.update(
                    {
                        f"component_{component}_mass": float(component_mass[index, component])
                        for component in range(self.config.num_components)
                    }
                )
                writer.writerow(row)


def prepare_data(
    scratch_dir: Path,
    device: torch.device,
) -> tuple[SinglePeptideDataModule, Any, torch.Tensor, torch.Tensor]:
    """Download Ace-A-Nme data, build OpenMM target, and construct the mean-free basis."""
    scratch_root = scratch_dir / "transferable-samplers"
    datamodule = SinglePeptideDataModule(
        data_dir=str(scratch_root / "sequential-boltzmann-generators-data"),
        sequence="Ace-A-Nme",
        temperature=300,
        num_dimensions=3,
        num_atoms=22,
        batch_size=512,
        num_workers=0,
        num_eval_samples=10_000,
    )
    datamodule.prepare_data()
    required_files = [
        datamodule.train_data_path,
        datamodule.val_data_path,
        datamodule.test_data_path,
        datamodule.pdb_path,
    ]
    missing = [path for path in required_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset download completed without the expected Ace-A-Nme_300K files:\n" + "\n".join(missing)
        )

    train_raw = torch.from_numpy(np.load(datamodule.train_data_path)).float()
    datamodule.std = (train_raw - train_raw.mean(dim=1, keepdim=True)).std()
    eval_context = datamodule.prepare_eval(sequence="Ace-A-Nme", stage="test")
    train_x = standardize_coords(train_raw, datamodule.std).to(device)

    num_atoms, spatial_dim = train_x.shape[1:]
    projector = torch.eye(num_atoms, dtype=torch.float64) - torch.ones(
        num_atoms, num_atoms, dtype=torch.float64
    ) / num_atoms
    eigenvalues, eigenvectors = torch.linalg.eigh(projector)
    basis = torch.kron(
        eigenvectors[:, eigenvalues > 0.5],
        torch.eye(spatial_dim, dtype=torch.float64),
    ).to(device=device, dtype=torch.float32)
    train_y = train_x.reshape(len(train_x), -1) @ basis
    print(
        {
            "train_samples": len(train_y),
            "mean_free_dimension": train_y.shape[1],
            "normalization_std": float(datamodule.std),
            "test_reference": len(eval_context.true_data),
        },
        flush=True,
    )
    return datamodule, eval_context, train_y, basis


def run_official_sbg(
    output_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[Path, Path]:
    """Optionally run the repository's official TarFlow + ULA-SMC baseline."""
    sample_file = output_dir / "test" / "Ace-A-Nme" / "samples_dict.pt"
    diagnostics_file = output_dir / "test" / "Ace-A-Nme" / "diagnostics.pt"
    command = [
        sys.executable,
        "-m",
        "transferable_samplers.eval",
        "experiment=single_system/eval/tarflow_Ace-A-Nme_ula",
        "trainer=gpu" if device.type == "cuda" else "trainer=cpu",
        "logger=csv",
        f"paths.output_dir={output_dir}",
        f"+callbacks.sampling_evaluation.output_dir={output_dir}",
        f"callbacks.sampling_evaluation.sampler.num_samples={config.num_particles}",
        f"callbacks.sampling_evaluation.sampler.num_annealing_steps={config.num_annealing_steps}",
    ]
    print("Official SBG command:\n" + " ".join(map(str, command)), flush=True)
    if config.run_official_sbg:
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=REPO_ROOT, env=package_subprocess_env(), check=True)
        if not sample_file.exists():
            raise FileNotFoundError(f"Official SBG run completed but did not create {sample_file}")
    elif not sample_file.exists():
        print("Official SBG artifact absent; Method 2 will be evaluated alone.", flush=True)
    return sample_file, diagnostics_file


def run_official_ecnf(
    output_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> Path:
    """Optionally run the repository's official ECNF++ + SNIS baseline."""
    sample_file = output_dir / "test" / "Ace-A-Nme" / "samples_dict.pt"
    command = [
        sys.executable,
        "-m",
        "transferable_samplers.eval",
        "experiment=single_system/eval/ecnf++_Ace-A-Nme_snis",
        "trainer=gpu" if device.type == "cuda" else "trainer=cpu",
        "logger=csv",
        f"paths.output_dir={output_dir}",
        f"+callbacks.sampling_evaluation.output_dir={output_dir}",
        f"callbacks.sampling_evaluation.sampler.num_samples={config.num_particles}",
    ]
    print("Official ECNF++ command:\n" + " ".join(map(str, command)), flush=True)
    if config.run_official_ecnf:
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=REPO_ROOT, env=package_subprocess_env(), check=True)
        if not sample_file.exists():
            raise FileNotFoundError(f"Official ECNF++ run completed but did not create {sample_file}")
    elif not sample_file.exists():
        print("Official ECNF++ artifact absent; ECNF++ will not be included.", flush=True)
    return sample_file


def rama(samples: torch.Tensor, normalization_std: torch.Tensor, topology: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the alanine phi/psi angles in degrees."""
    xyz = (samples.detach().cpu() * normalization_std.cpu()).numpy()
    trajectory = md.Trajectory(xyz, topology)
    phi = md.compute_phi(trajectory)[1].reshape(len(samples), -1)[:, 0]
    psi = md.compute_psi(trajectory)[1].reshape(len(samples), -1)[:, 0]
    return np.rad2deg(phi), np.rad2deg(psi)


def save_comparison_plots(
    run_dir: Path,
    comparison_samples: dict[str, SamplesData],
    eval_context: Any,
    datamodule: SinglePeptideDataModule,
    method2: dict[str, Any],
    sbg_diagnostics: Any,
    config: ExperimentConfig,
) -> None:
    """Save weighted Ramachandran, ESS, and GMM-component diagnostics."""
    plot_sets: list[tuple[str, torch.Tensor, torch.Tensor | None]] = [
        ("test MD", eval_context.true_data.samples, None)
    ]
    if "ecnf_snis" in comparison_samples:
        plot_sets.append(("official ECNF++ SNIS", comparison_samples["ecnf_snis"].samples, None))
    if "sbg_smc" in comparison_samples:
        plot_sets.append(("official SBG SMC", comparison_samples["sbg_smc"].samples, None))
    plot_sets.append(
        (
            "full-cov GMM + Prop. 3",
            comparison_samples["gmm_prop3"].samples,
            comparison_samples["gmm_prop3"].logw,
        )
    )

    fig, axes = plt.subplots(1, len(plot_sets), figsize=(5 * len(plot_sets), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (title, samples, logw) in zip(axes, plot_sets, strict=True):
        phi, psi = rama(samples, datamodule.std, eval_context.topology)
        if logw is None:
            ax.hexbin(phi, psi, gridsize=55, bins="log", mincnt=1)
        else:
            weights = torch.softmax(logw.detach().cpu(), 0).numpy()
            ax.hexbin(phi, psi, C=weights, reduce_C_function=np.sum, gridsize=55, mincnt=1)
        ax.set(
            xlim=(-180, 180),
            ylim=(-180, 180),
            xlabel=r"$\phi$ (deg)",
            ylabel=r"$\psi$ (deg)",
            title=title,
        )
    save_figure(fig, run_dir / "plots" / "comparison" / "ramachandran")

    diagnostics = method2["diagnostics"]
    times = np.asarray(diagnostics["t"], dtype=float)
    method2_ess = np.asarray(diagnostics["ess"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(times, method2_ess, label="Method 2: Prop. 3")
    for time, flag in zip(times, diagnostics["resampled"], strict=True):
        if flag:
            ax.axvline(time, color="C0", alpha=0.25, linestyle="--")
    if sbg_diagnostics is not None:
        sbg_diag = sbg_diagnostics["diagnostics"]
        ax.plot(np.asarray(sbg_diag["t"], float), np.asarray(sbg_diag["ess"], float), label="Official SBG SMC")
    if "ecnf_snis" in comparison_samples and comparison_samples["ecnf_snis"].logw is not None:
        ecnf_ess = Prop3Experiment.normalized_ess(comparison_samples["ecnf_snis"].logw)
        ax.scatter([1.0], [ecnf_ess], marker="D", s=55, label="Official ECNF++ SNIS ESS")
    ax.axhline(config.ess_threshold, color="black", linestyle=":", label="resampling threshold")
    ax.set(xlabel="annealing time", ylabel="ESS/N", ylim=(0, 1), title="Annealing weight efficiency")
    ax.legend()
    save_figure(fig, run_dir / "plots" / "comparison" / "ess_trajectory")

    masses = torch.stack(diagnostics["raw_component_mass"]).numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for component in range(config.num_components):
        ax.plot(times, masses[:, component], label=f"component {component}")
    ax.set(xlabel="annealing time", ylabel="raw ancestry fraction", ylim=(0, 1), title="GMM ancestry over time")
    ax.legend(ncol=2, fontsize=8)
    save_figure(fig, run_dir / "plots" / "comparison" / "component_mass_trajectory")


def evaluate_and_save(
    run_dir: Path,
    method2: dict[str, Any],
    eval_context: Any,
    datamodule: SinglePeptideDataModule,
    config: ExperimentConfig,
    sbg_sample_file: Path,
    sbg_diagnostics_file: Path,
    ecnf_sample_file: Path,
) -> dict[str, Any]:
    """Run repository metrics and save all evaluation plots and summaries."""
    comparison_samples = {
        "gmm_prop3": SamplesData(
            method2["samples"].cpu(),
            method2["target_energy"].cpu(),
            logw=method2["logw"].cpu(),
        )
    }
    sbg_diagnostics = None
    if ecnf_sample_file.exists():
        official_ecnf = torch.load(ecnf_sample_file, map_location="cpu", weights_only=False)
        if "resampled" not in official_ecnf:
            raise KeyError(f"Expected 'resampled' in official ECNF++ artifact {ecnf_sample_file}")
        comparison_samples["ecnf_snis"] = samples_data_to_cpu(official_ecnf["resampled"])
    if sbg_sample_file.exists():
        official = torch.load(sbg_sample_file, map_location="cpu", weights_only=False)
        comparison_samples["sbg_smc"] = samples_data_to_cpu(official["smc"])
        if sbg_diagnostics_file.exists():
            sbg_diagnostics = torch.load(sbg_diagnostics_file, map_location="cpu", weights_only=False)

    evaluator_plot_root = run_dir / "plots" / "evaluator"

    def save_evaluator_plot(fig: plt.Figure, title: str) -> None:
        save_figure(fig, evaluator_plot_root / safe_plot_name(title))

    evaluator = PeptideEnsembleEvaluator(
        fix_symmetry=True,
        drop_unfixable_symmetry=False,
        num_eval_samples=min(10_000, config.num_particles),
        do_plots=True,
    )
    metrics = evaluator.evaluate(
        comparison_samples,
        eval_context,
        log_image_fn=save_evaluator_plot,
        prefix="test/Ace-A-Nme",
    )
    torch.save(metrics, run_dir / "metrics" / "evaluator_metrics.pt")
    write_json(run_dir / "metrics" / "evaluator_metrics.json", metrics)

    summary = {
        "final_segment_ess_over_n": Prop3Experiment.normalized_ess(method2["logw"]),
        "resampling_events": method2["num_resamples"],
        "particles": len(method2["samples"]),
        "official_sbg_included": "sbg_smc" in comparison_samples,
        "official_ecnf_included": "ecnf_snis" in comparison_samples,
        "ecnf_snis_ess_over_n": (
            Prop3Experiment.normalized_ess(comparison_samples["ecnf_snis"].logw)
            if "ecnf_snis" in comparison_samples and comparison_samples["ecnf_snis"].logw is not None
            else None
        ),
        "evaluator_metrics": metrics,
    }
    write_json(run_dir / "metrics" / "summary.json", summary)
    save_comparison_plots(
        run_dir,
        comparison_samples,
        eval_context,
        datamodule,
        method2,
        sbg_diagnostics,
        config,
    )
    return summary


def main() -> None:
    """Run the complete experiment and persist its artifacts."""
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=True)
    if args.scratch_dir is not None:
        scratch_dir = args.scratch_dir.expanduser().resolve()
        os.environ["SCRATCH_DIR"] = str(scratch_dir)
    elif "SCRATCH_DIR" in os.environ:
        scratch_dir = Path(os.environ["SCRATCH_DIR"]).expanduser().resolve()
    else:
        scratch_dir = REPO_ROOT / "scratch"
        os.environ["SCRATCH_DIR"] = str(scratch_dir)
        print(f"SCRATCH_DIR was not set; using ignored local cache: {scratch_dir}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = resolve_config(args, device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_float32_matmul_precision("highest")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "smoke" if config.smoke_test else "full"
    run_name = args.run_name or f"{mode}_seed{config.seed}_{timestamp}"
    run_dir = args.output_dir.expanduser().resolve() / "runs" / run_name
    for directory in ("checkpoints", "metrics", "plots", "samples"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {**asdict(config), "scratch_dir": scratch_dir, "run_dir": run_dir})
    print({"run_dir": str(run_dir), **asdict(config)}, flush=True)

    sbg_variant = f"{mode}_n{config.num_particles}_steps{config.num_annealing_steps}"
    sbg_sample_file, sbg_diagnostics_file = run_official_sbg(
        args.output_dir.expanduser().resolve() / "official_sbg" / sbg_variant,
        config,
        device,
    )
    ecnf_variant = f"{mode}_n{config.num_particles}"
    ecnf_sample_file = run_official_ecnf(
        args.output_dir.expanduser().resolve() / "official_ecnf" / ecnf_variant,
        config,
        device,
    )
    datamodule, eval_context, train_y, basis = prepare_data(scratch_dir, device)
    experiment = Prop3Experiment(config, run_dir, train_y, basis, datamodule, eval_context)
    experiment.fit_gmm()
    experiment.train_drift(args.resume_checkpoint)
    method2 = experiment.run_prop3(resample=True)
    summary = evaluate_and_save(
        run_dir,
        method2,
        eval_context,
        datamodule,
        config,
        sbg_sample_file,
        sbg_diagnostics_file,
        ecnf_sample_file,
    )
    print("Final summary:", json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)
    print(f"All artifacts saved under: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
