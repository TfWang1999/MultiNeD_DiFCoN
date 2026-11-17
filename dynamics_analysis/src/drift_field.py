"""Direct projected model drift and the paper's saved-data display interpolation."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, cKDTree
from .model_access import ARMS, SCORE_COLS, TREATMENTS, load_final_reaction
from .pseudocells import DiffusionMapNystrom, TREATMENTS as LEGACY_TREATMENTS
from .paths import BASE, REPO, new_output

ALIASES = {"Anti-41BB": "Anti-4-1BB", "Anti-PD1+Anti-41BB": "Combination"}
DC = ["DC1", "DC2", "DC3"]
KEYS = ["treatment", "sim_seed", "time", "pseudocell_id"]
PROJECTION_EPSILON = 0.01
PROJECTION_BATCH_SIZE = 2048
VELOCITY_BATCH_SIZE = 512


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def run_source(cfg, output, inputs, pseudocells):
    torch.set_num_threads(int(cfg["cpu_threads"]))
    source = new_output(output)
    anchors = read_csv(pseudocells / "diffmap_pseudocell_velocity.csv")[KEYS + DC]
    membership = read_csv(pseudocells / "diffmap_pseudocell_membership.csv")
    particles = read_csv(pseudocells / "diffmap_sim_day1_6.csv.gz")
    particles = particles.loc[particles.time.isin(anchors.time.unique())]
    particles = particles.merge(membership, on=["sim_seed", "particle_id"], how="left", validate="many_to_one")
    if particles.pseudocell_id.isna().any():
        raise AssertionError("Missing original pseudocell membership")
    dm = DiffusionMapNystrom.load(inputs / "frozen_nystrom_state.npz", n_jobs=cfg["cpu_threads"])
    model, provenance = load_final_reaction(REPO, cfg)
    device = next(model.parameters()).device
    X = particles[SCORE_COLS].to_numpy(dtype=float)
    velocity = np.empty(X.shape, dtype=np.float64)
    with torch.no_grad():
        for legacy_tr, (P, B) in zip(LEGACY_TREATMENTS, ARMS):
            for time, p in particles[particles.treatment == legacy_tr].groupby("time", sort=True):
                rows = p.index.to_numpy()
                for start in range(0, len(rows), VELOCITY_BATCH_SIZE):
                    idx = rows[start:start + VELOCITY_BATCH_SIZE]
                    z = torch.tensor(X[idx], dtype=torch.float32, device=device)
                    velocity[idx] = model.drift_integrator._velocity(z, P, B, time).cpu().numpy()
            print(f"Direct MultiNeD drift on selected particles: {legacy_tr}", flush=True)
    phi = dm.transform(X, batch_size=PROJECTION_BATCH_SIZE).astype(float)
    shifted = dm.transform(X + PROJECTION_EPSILON * velocity, batch_size=PROJECTION_BATCH_SIZE)
    projected_v = (shifted.astype(float) - phi) / PROJECTION_EPSILON
    source_particles = particles[KEYS].copy()
    source_particles["mass"] = particles.get("particle_weight", 1.0)
    velocity_cols = [f"v{d}" for d in DC]
    source_particles[velocity_cols] = projected_v * source_particles["mass"].to_numpy()[:, None]
    pc_v = source_particles.groupby(KEYS, sort=True)[["mass", *velocity_cols]].sum().reset_index()
    pc_v[velocity_cols] = pc_v[velocity_cols].div(pc_v.pop("mass"), axis=0)
    pc_v = anchors.merge(pc_v, on=KEYS, how="left", validate="one_to_one")
    if pc_v.isna().any().any():
        raise AssertionError("Direct drift lacks an original velocity anchor")
    pc_v.to_csv(source / "pc_time_pseudocell_direct_vtheta.csv", index=False)
    # Reuse the frozen display parameters for the projected field.
    display = json.loads((BASE / "outputs/vtheta/source_data/pc_time_display_parameters.json").read_text(encoding="utf-8"))
    (source / "pc_time_display_parameters.json").write_text(json.dumps(display, indent=2) + "\n", encoding="utf-8")
    (source / "run_record.json").write_text(json.dumps({"model": provenance, "projection_epsilon": PROJECTION_EPSILON}, indent=2) + "\n", encoding="utf-8")

def canonical_treatments(frame):
    frame = frame.copy()
    frame["treatment"] = frame["treatment"].replace(ALIASES)
    unknown = set(frame.treatment) - set(TREATMENTS)
    if unknown:
        raise ValueError(f"Unknown treatment labels: {sorted(unknown)}")
    return frame

def dense_field(vectors, display):
    """Locally interpolate direct projected vθ; mask outside observed support."""
    coarse_bins, fine_bins = 7, 56
    max_gap_bins, smoothing_sigma = 1.2, 0.75
    bounds = display["grid"]
    xe = np.linspace(bounds["xedges"][0], bounds["xedges"][-1], coarse_bins + 1)
    ye = np.linspace(bounds["yedges"][0], bounds["yedges"][-1], coarse_bins + 1)
    xlim, ylim = display["main_limits"]
    xs, ys = np.linspace(*xlim, fine_bins), np.linspace(*ylim, fine_bins)
    xgrid, ygrid = np.meshgrid(xs, ys)
    xy = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    unit = np.array([xe[1] - xe[0], ye[1] - ye[0]])
    result = []
    for treatment in TREATMENTS:
        part = vectors.loc[vectors.treatment.eq(treatment)].copy()
        part["xbin"] = np.clip(np.searchsorted(xe, part.DC1, side="right") - 1, 0, coarse_bins - 1)
        part["ybin"] = np.clip(np.searchsorted(ye, part.DC2, side="right") - 1, 0, coarse_bins - 1)
        anchors = part.groupby(["xbin", "ybin"], sort=True).agg(
            x=("DC1", "mean"), y=("DC2", "mean"),
            u=("vDC1", "mean"), v=("vDC2", "mean")).reset_index()
        if len(anchors) < 3:
            raise ValueError(f"Too few direct velocity anchors for {treatment}")
        anchor_xy = anchors[["x", "y"]].to_numpy()
        uv = anchors[["u", "v"]].to_numpy()
        distance, nearest = cKDTree(anchor_xy / unit).query(xy / unit)
        support = ((Delaunay(part[["DC1", "DC2"]].to_numpy() / unit)
                    .find_simplex(xy / unit) >= 0) & (distance <= max_gap_bins))
        linear = np.asarray(LinearNDInterpolator(anchor_xy / unit, uv)(xy / unit))
        edge_fill = support & ~np.isfinite(linear).all(axis=1)
        linear[edge_fill] = uv[nearest[edge_fill]]
        linear[~support] = np.nan
        valid = support.reshape(xgrid.shape)
        weights = gaussian_filter(valid.astype(float), smoothing_sigma)
        fields = []
        for i in (0, 1):
            value = linear[:, i].reshape(xgrid.shape)
            smooth = np.full_like(value, np.nan)
            np.divide(gaussian_filter(np.where(valid, value, 0), smoothing_sigma),
                      weights, out=smooth, where=valid & (weights > 0))
            fields.append(smooth.ravel())
        result.append(pd.DataFrame({
            "treatment": treatment, "x": xy[:, 0], "y": xy[:, 1],
            "display_U": fields[0], "display_V": fields[1],
            "inside_display_support": support,
        }))
    return pd.concat(result, ignore_index=True)

def run_display(output, source=None):
    source = source or BASE / "outputs/vtheta/source_data"
    display = json.loads((source / "pc_time_display_parameters.json").read_text(encoding="utf-8"))
    vectors = canonical_treatments(pd.read_csv(source / "pc_time_pseudocell_direct_vtheta.csv"))
    dense = dense_field(vectors, display)
    dense.to_csv(new_output(output) / "vtheta_dense_field.csv.gz", index=False)
