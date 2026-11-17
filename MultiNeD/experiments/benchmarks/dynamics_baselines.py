from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pandas as pd

from .dynamics_benchmark_common import PROJECT_ROOT, project_path
from .external_methods import ExternalSnapshotMethod

from common.cd8_zdist_data import mouse_balanced_sample
from scripts.reproduce.paper_settings import drift_configs
from training.drift_training import (
    load_drift_checkpoint,
    save_drift_checkpoint,
    train_drift_model,
)
from experiments.volume_loto.model_runner import _apply_overrides


class SnapshotDynamicsMethod:
    method_id = "abstract"

    def fit(self, group_data: dict, transitions: pd.DataFrame, seed: int, output_dir: Path | None = None) -> None:
        pass

    def predict(self, source: np.ndarray, *, day0: float, day1: float, P: float, B: float) -> np.ndarray:
        raise NotImplementedError


class PersistenceMethod(SnapshotDynamicsMethod):
    method_id = "persistence"

    def predict(self, source: np.ndarray, **_: Any) -> np.ndarray:
        return np.asarray(source, dtype=np.float32).copy()


class TreatmentLinearShiftMethod(SnapshotDynamicsMethod):
    """Ridge-regressed mean velocity with treatment and time covariates.

    The method keeps the empirical source shape and predicts only its mean
    displacement.  It is a deliberately transparent lower-capacity comparator
    for sparse, unpaired snapshots.
    """

    method_id = "treatment_linear_shift"

    def __init__(self, ridge: float = 1e-3, time_scale: float = 15.0):
        self.ridge = float(ridge)
        self.time_scale = float(time_scale)
        self.coefficients: np.ndarray | None = None

    def _features(self, P: float, B: float, midpoint: float) -> np.ndarray:
        return np.asarray([1.0, P, B, P * B, midpoint / self.time_scale], dtype=float)

    def fit(self, group_data: dict, transitions: pd.DataFrame, seed: int, output_dir: Path | None = None) -> None:
        del seed, output_dir
        x_rows, y_rows, weights = [], [], []
        for row in transitions.itertuples(index=False):
            source = np.asarray(group_data[row.source_group]["Z_all"], dtype=float)
            target = np.asarray(group_data[row.target_group]["Z_all"], dtype=float)
            dt = float(row.dt)
            x_rows.append(self._features(float(row.P), float(row.B), 0.5 * (float(row.day0) + float(row.day1))))
            y_rows.append((target.mean(axis=0) - source.mean(axis=0)) / dt)
            weights.append(float(getattr(row, "loss_weight", 1.0)))
        if not x_rows:
            raise ValueError("treatment_linear_shift requires at least one training transition")
        x = np.vstack(x_rows)
        y = np.vstack(y_rows)
        w = np.sqrt(np.asarray(weights, dtype=float)).reshape(-1, 1)
        penalty = self.ridge * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve((x * w).T @ (x * w) + penalty, (x * w).T @ (y * w))

    def predict(self, source: np.ndarray, *, day0: float, day1: float, P: float, B: float) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Method has not been fitted")
        features = self._features(P, B, 0.5 * (day0 + day1))
        displacement = (day1 - day0) * (features @ self.coefficients)
        return (np.asarray(source, dtype=float) + displacement).astype(np.float32)


class EntropicOTBarycentricMethod(SnapshotDynamicsMethod):
    """Nearest-condition entropic-OT barycentric displacement transport.

    This baseline learns maps only from training snapshot pairs.  At inference,
    each new source cell receives the displacement of nearby support cells from
    the closest observed treatment/time transition; held-out target cells are
    never used.
    """

    method_id = "entropic_ot_barycentric"

    def __init__(self, epsilon: float = 0.08, n_support: int = 128, knn: int = 8):
        self.epsilon = float(epsilon)
        self.n_support = int(n_support)
        self.knn = int(knn)
        self.maps: list[dict[str, Any]] = []

    def fit(self, group_data: dict, transitions: pd.DataFrame, seed: int, output_dir: Path | None = None) -> None:
        del output_dir
        try:
            import ot
        except ImportError as exc:
            raise ImportError("POT is required for entropic_ot_barycentric") from exc
        rng = np.random.default_rng(int(seed))
        self.maps = []
        for row in transitions.itertuples(index=False):
            source, _ = mouse_balanced_sample(
                group_data, row.source_group, n_max=self.n_support,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            target, _ = mouse_balanced_sample(
                group_data, row.target_group, n_max=self.n_support,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            cost = ot.dist(source.astype(float), target.astype(float), metric="sqeuclidean")
            positive = cost[cost > 0]
            cost_scale = float(np.median(positive)) if positive.size else 1.0
            regularization = max(self.epsilon * cost_scale, 1e-6)
            a = np.full(source.shape[0], 1.0 / source.shape[0])
            b = np.full(target.shape[0], 1.0 / target.shape[0])
            plan = np.asarray(
                ot.sinkhorn(
                    a, b, cost, reg=regularization,
                    method="sinkhorn_log", numItermax=1000, stopThr=1e-7,
                ),
                dtype=float,
            )
            row_mass = plan.sum(axis=1, keepdims=True)
            barycenter = (plan @ target) / np.maximum(row_mass, 1e-12)
            self.maps.append({
                "source": source.astype(np.float32),
                "velocity": ((barycenter - source) / float(row.dt)).astype(np.float32),
                "P": float(row.P),
                "B": float(row.B),
                "midpoint": 0.5 * (float(row.day0) + float(row.day1)),
                "dt": float(row.dt),
                "transition_id": str(row.transition_id),
            })
        if not self.maps:
            raise ValueError("entropic_ot_barycentric requires at least one training transition")

    def _map_distance(self, record: dict[str, Any], P: float, B: float, midpoint: float) -> float:
        return (
            4.0 * (record["P"] - P) ** 2
            + 4.0 * (record["B"] - B) ** 2
            + ((record["midpoint"] - midpoint) / 15.0) ** 2
        )

    def predict(self, source: np.ndarray, *, day0: float, day1: float, P: float, B: float) -> np.ndarray:
        if not self.maps:
            raise RuntimeError("Method has not been fitted")
        midpoint = 0.5 * (day0 + day1)
        record = min(self.maps, key=lambda item: self._map_distance(item, P, B, midpoint))
        support = record["source"].astype(float)
        query = np.asarray(source, dtype=float)
        distances = ((query[:, None, :] - support[None, :, :]) ** 2).sum(axis=2)
        k = min(max(1, self.knn), support.shape[0])
        neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        neighbor_dist = np.take_along_axis(distances, neighbors, axis=1)
        weights = 1.0 / np.maximum(neighbor_dist, 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        velocities = record["velocity"][neighbors]
        velocity = (velocities * weights[:, :, None]).sum(axis=1)
        return (query + (day1 - day0) * velocity).astype(np.float32)


class ProjectDriftMethod(SnapshotDynamicsMethod):
    method_id = "project_balanced_sinkhorn_drift"

    def __init__(
        self, base_config: dict[str, Any], overrides: dict[str, Any] | None = None, *,
        use_paper_pretraining: bool = False,
    ):
        self.base_config = base_config
        self.overrides = copy.deepcopy(overrides or {})
        self.use_paper_pretraining = use_paper_pretraining
        self.integrator = None
        self.model = None

    def fit(self, group_data: dict, transitions: pd.DataFrame, seed: int, output_dir: Path | None = None) -> None:
        if self.use_paper_pretraining:
            model_config, train_config = drift_configs()
            values = {}
        else:
            source = project_path(self.base_config["inputs"]["drift_checkpoint"], PROJECT_ROOT)
            _, _, model_config, train_config, _ = load_drift_checkpoint(source, device="cpu")
            values = copy.deepcopy(self.base_config.get("drift", {}))
        values.update(self.overrides)
        values["seed"] = int(seed)
        train_config = _apply_overrides(train_config, values, "drift")
        self.model, self.integrator, history = train_drift_model(
            group_data, transitions, model_config=model_config, train_config=train_config
        )
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_drift_checkpoint(
                output_dir / "drift_model.pt", self.model, model_config, train_config, history_df=history
            )

    def predict(self, source: np.ndarray, *, day0: float, day1: float, P: float, B: float) -> np.ndarray:
        if self.integrator is None:
            raise RuntimeError("Method has not been fitted")
        device = next(self.integrator.parameters()).device
        values = torch.as_tensor(source, dtype=torch.float32, device=device)
        self.integrator.eval()
        with torch.no_grad():
            predicted = self.integrator(values, P=float(P), B=float(B), day0=float(day0), dt=float(day1 - day0))
        return predicted.detach().cpu().numpy().astype(np.float32)


def build_method(method_id: str, settings: dict[str, Any], base_config: dict[str, Any]):
    method_id = str(method_id)
    values = copy.deepcopy(settings or {})
    if method_id == "persistence":
        return PersistenceMethod()
    if method_id == "treatment_linear_shift":
        return TreatmentLinearShiftMethod(**values)
    if method_id == "entropic_ot_barycentric":
        return EntropicOTBarycentricMethod(**values)
    if method_id == "project_balanced_sinkhorn_drift":
        use_paper_pretraining = values.pop("use_paper_pretraining", False)
        return ProjectDriftMethod(base_config, overrides=values, use_paper_pretraining=use_paper_pretraining)
    if method_id in {"mioflow", "tigon", "official_ot_cfm"}:
        return ExternalSnapshotMethod(method_id, values)
    raise KeyError(f"Unknown dynamics method: {method_id}")


__all__ = [
    "EntropicOTBarycentricMethod",
    "PersistenceMethod",
    "ProjectDriftMethod",
    "SnapshotDynamicsMethod",
    "TreatmentLinearShiftMethod",
    "build_method",
]
