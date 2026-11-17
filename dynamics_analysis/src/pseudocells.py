"""Shared diffusion map and fixed-membership longitudinal pseudocells."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .paths import new_output

SCORE_COLS = ["P_precursor_zstd", "Activation_zstd", "Eff_Early_zstd", "Eff_Cytox_zstd", "Dysf_zstd"]
TREATMENTS = ["Control", "Anti-PD1", "Anti-41BB", "Anti-PD1+Anti-41BB"]
PRIMARY_START, PRIMARY_END = 1.0, 6.0
PSEUDOCELLS_PER_SEED = 18
PSEUDOCELL_TIME_STEP = 0.5


class DiffusionMapNystrom:
    n_neighbors: int = 64
    n_components: int = 3
    alpha: float = 1.0

    def fit(self, X: np.ndarray) -> "DiffusionMapNystrom":
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.X_ref = Z
        k = min(self.n_neighbors + 1, len(Z))
        self.nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(Z)
        distances, indices = self.nn.kneighbors(Z)
        distances, indices = distances[:, 1:], indices[:, 1:]
        self.epsilon = float(np.median(distances[:, -1] ** 2))
        self.epsilon = max(self.epsilon, 1e-8)

        rows = np.repeat(np.arange(len(Z)), distances.shape[1])
        cols = indices.reshape(-1)
        weights = np.exp(-(distances.reshape(-1) ** 2) / self.epsilon)
        W = sparse.coo_matrix((weights, (rows, cols)), shape=(len(Z), len(Z))).tocsr()
        W = W.maximum(W.T)
        self.q_ref = np.asarray(W.sum(axis=1)).ravel()
        q_inv = np.power(np.maximum(self.q_ref, 1e-12), -self.alpha)
        W_alpha = sparse.diags(q_inv) @ W @ sparse.diags(q_inv)
        d = np.asarray(W_alpha.sum(axis=1)).ravel()
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        S = sparse.diags(d_inv_sqrt) @ W_alpha @ sparse.diags(d_inv_sqrt)

        n_eigs = min(self.n_components + 1, len(Z) - 1)
        eigvals, eigvecs = eigsh(S, k=n_eigs, which="LA", tol=1e-5, maxiter=5000)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        self.eigvals = eigvals[1 : self.n_components + 1]
        self.psi_ref = d_inv_sqrt[:, None] * eigvecs[:, 1 : self.n_components + 1]

        raw_coords = self.psi_ref * self.eigvals[None, :]
        for j in range(self.n_components):
            correlations = []
            for feature_index in range(X.shape[1]):
                corr = np.corrcoef(raw_coords[:, j], X[:, feature_index])[0, 1]
                correlations.append(0.0 if np.isnan(corr) else float(corr))
            strongest = int(np.argmax(np.abs(correlations)))
            sign = 1.0 if correlations[strongest] >= 0 else -1.0
            self.psi_ref[:, j] *= sign

        raw_coords = self.psi_ref * self.eigvals[None, :]
        self.coord_scale = 1.0 / np.maximum(raw_coords.std(axis=0), 1e-12)
        self.reference_coordinates_ = raw_coords * self.coord_scale[None, :]
        return self

    def save(self, path):
        np.savez_compressed(
            path, reference_id=np.arange(len(self.X_ref)), X_ref=self.X_ref,
            scaler_mean=self.scaler.mean_, scaler_scale=self.scaler.scale_, scaler_var=self.scaler.var_,
            q_ref=self.q_ref, psi_ref=self.psi_ref, eigvals=self.eigvals, coord_scale=self.coord_scale,
            epsilon=self.epsilon, alpha=self.alpha, n_neighbors_transform=self.nn.n_neighbors,
            fitted_reference_coordinates=self.reference_coordinates_,
        )

    @classmethod
    def load(cls, path, n_jobs=-1):
        """Restore the common basis for both particle and direct-drift projections."""
        dm = cls()
        dm.scaler = StandardScaler()
        with np.load(path, allow_pickle=False) as saved:
            for attr in ("X_ref", "q_ref", "psi_ref", "eigvals", "coord_scale"):
                setattr(dm, attr, saved[attr].copy())
            dm.epsilon, dm.alpha = float(saved["epsilon"]), float(saved["alpha"])
            dm.reference_coordinates_ = saved["fitted_reference_coordinates"].copy()
            dm.scaler.mean_, dm.scaler.scale_, dm.scaler.var_ = [saved[k].copy() for k in
                ("scaler_mean", "scaler_scale", "scaler_var")]
            neighbors = int(saved["n_neighbors_transform"])
        dm.n_neighbors = neighbors - 1
        dm.scaler.n_features_in_, dm.scaler.n_samples_seen_ = dm.X_ref.shape[1], len(dm.X_ref)
        dm.nn = NearestNeighbors(n_neighbors=neighbors, n_jobs=n_jobs).fit(dm.X_ref)
        return dm

    def transform(self, X: np.ndarray, batch_size: int = 20000) -> np.ndarray:
        Z = self.scaler.transform(X)
        result = np.empty((len(Z), self.n_components), dtype=np.float32)
        for start in range(0, len(Z), batch_size):
            stop = min(start + batch_size, len(Z))
            distances, indices = self.nn.kneighbors(Z[start:stop])
            raw = np.exp(-(distances**2) / self.epsilon)
            q_x = np.maximum(raw.sum(axis=1), 1e-12)
            kernel = raw / (q_x[:, None] ** self.alpha * np.maximum(self.q_ref[indices], 1e-12) ** self.alpha)
            prob = kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-12)
            coords = np.sum(prob[:, :, None] * self.psi_ref[indices], axis=1)
            result[start:stop] = (coords * self.coord_scale[None, :]).astype(np.float32)
        return result



def project_diffusion_map(input_dir) -> pd.DataFrame:
    dm = DiffusionMapNystrom.load(input_dir / "frozen_nystrom_state.npz")

    usecols = ["treatment", "sim_seed", "time", "particle_id", "particle_weight"] + SCORE_COLS
    particles = pd.read_csv(input_dir / "particle_functional_states.csv.gz", usecols=lambda col: col in usecols)
    day16 = particles.loc[particles["time"].between(PRIMARY_START, PRIMARY_END)].copy()
    coords = dm.transform(day16[SCORE_COLS].to_numpy(float))
    for j in range(3):
        day16[f"DC{j + 1}"] = coords[:, j]
    return day16


def build_day1_6_pseudocells(day16: pd.DataFrame, output) -> None:
    """Create weighted longitudinal pseudocell centroids.

    Particle IDs modulo 18 assign each particle to a group for its lifetime.
    """

    scaled_time = (day16["time"] - PRIMARY_START) / PSEUDOCELL_TIME_STEP
    selected = day16[np.isclose(scaled_time, np.round(scaled_time), atol=1e-6)].copy()
    selected["pseudocell_id"] = selected["particle_id"].astype(int) % PSEUDOCELLS_PER_SEED

    membership = selected[["sim_seed", "particle_id", "pseudocell_id"]].drop_duplicates().sort_values(["sim_seed", "pseudocell_id", "particle_id"])
    membership.to_csv(output / "diffmap_pseudocell_membership.csv", index=False)

    value_cols = SCORE_COLS + ["DC1", "DC2", "DC3"]
    selected["particle_weight"] = selected.get("particle_weight", 1.0)
    selected[value_cols] = selected[value_cols].mul(selected["particle_weight"], axis=0)
    pc = (
        selected.groupby(["treatment", "sim_seed", "time", "pseudocell_id"], observed=True)
        .agg(n_particles=("particle_id", "size"), mass=("particle_weight", "sum"),
             **{col: (col, "sum") for col in value_cols})
        .reset_index()
    )
    pc[value_cols] = pc[value_cols].div(pc.pop("mass"), axis=0)
    pc.to_csv(output / "diffmap_sim_day1_6_pseudocells.csv.gz", index=False)

    ordered = pc.sort_values(["treatment", "sim_seed", "pseudocell_id", "time"])
    next_time = ordered.groupby(["treatment", "sim_seed", "pseudocell_id"], observed=True, sort=False)["time"].shift(-1)
    valid = np.isclose(next_time - ordered["time"], PSEUDOCELL_TIME_STEP, atol=1e-6)
    anchors = ordered.loc[valid, ["treatment", "sim_seed", "time", "pseudocell_id", "DC1", "DC2", "DC3"]]
    anchors.to_csv(output / "diffmap_pseudocell_velocity.csv", index=False)


def run(output, inputs):
    output = new_output(output)
    day16 = project_diffusion_map(inputs)
    day16.to_csv(output / "diffmap_sim_day1_6.csv.gz", index=False)
    build_day1_6_pseudocells(day16, output)
