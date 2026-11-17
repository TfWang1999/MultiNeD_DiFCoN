"""Generate the paper's paired trajectories from a fitted MultiNeD model."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import torch

from .model_access import ARMS, SCORE_COLS, load_final_reaction
from .paths import REPO, new_output
from .pseudocells import DiffusionMapNystrom, TREATMENTS

SEEDS = (2026, 2027, 2028)
TIMES = np.round(np.arange(1.0, 17.05, 0.1), 8)
N_PARTICLES = 256
STATES = ["Precursor-like", "Activated transition", "Early effector", "Cytotoxic effector",
          "Dysfunctional-like", "Cyto-dysf hybrid", "Low-activity"]
STATE_SLUGS = ["pre", "act", "early", "cyto", "dysf", "hybrid", "low"]
# These templates name fitted mixture components; they do not constrain the fit.
STATE_TEMPLATES = np.array([
    [0.70, 0.00, -0.70, -0.70, -0.20],
    [0.50, 0.70, 0.30, 0.20, 0.30],
    [-0.80, -0.70, 1.00, -0.30, -0.60],
    [-0.80, -0.20, 1.20, 1.00, 0.00],
    [-0.60, -0.20, -0.50, 0.30, 0.80],
    [-0.30, 0.60, 0.80, 1.00, 1.00],
    [-0.80, -1.00, -0.70, -0.90, -1.00],
])


def observed_reference(groups):
    """Sample up to 512 cells per observed group, using the original row order."""
    rng = np.random.default_rng(7123)
    frames = []
    for group_id, group in groups.items():
        z = np.asarray(group["Z_all"], dtype=float)
        indices = rng.choice(len(z), size=min(512, len(z)), replace=False)
        frame = pd.DataFrame(z[indices], columns=SCORE_COLS)
        frame.insert(0, "group_id", group_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@torch.no_grad()
def paired_trajectories(model, groups, volume0, reference):
    """Use shared day-1 Control particles and paired inflow draws across treatments."""
    from common.cd8_zdist_data import mouse_balanced_sample

    device = next(model.parameters()).device
    thresholds = reference[[SCORE_COLS[3], SCORE_COLS[4]]].quantile(0.70).to_numpy()
    summaries, particles = [], []
    for seed in SEEDS:
        initial, _ = mouse_balanced_sample(groups, "day1__Control", n_max=N_PARTICLES, random_state=seed)
        z0 = torch.as_tensor(initial, dtype=torch.float32, device=device)
        for treatment, (P, B) in zip(TREATMENTS, ARMS):
            torch.manual_seed(50000 + seed)
            records = model.simulate(z0.clone(), P=P, B=B, day0=float(TIMES[0]),
                                     time_points=TIMES, volume0=volume0)
            for record in records:
                z = record["z"]
                values = z.cpu().numpy()
                weights = record["particle_weights"].cpu().numpy()
                favorable = (values[:, 3] >= thresholds[0]) & (values[:, 4] < thresholds[1])
                n_total, mean_k = float(record["N"]), float(record["effector_activity"])
                row = {
                    "treatment": treatment, "sim_seed": seed, "time": record["time"],
                    "N": n_total, "mean_k": mean_k, "total_cd8_activity": n_total * mean_k,
                    "volume": float(record["volume"]), "favorable_fraction": float(np.average(favorable, weights=weights)),
                    "drift_P": P, "drift_B": B, "reaction_P": P, "reaction_B": B,
                }
                row.update(zip(SCORE_COLS, np.average(values, axis=0, weights=weights)))
                summaries.append(row)
                frame = pd.DataFrame(values, columns=SCORE_COLS)
                frame.insert(0, "particle_id", np.arange(len(values)))
                for col in ("time", "sim_seed", "treatment"):
                    frame.insert(0, col, row[col])
                frame["particle_weight"] = weights
                frame["q_in_index"] = record["q_in_index"].cpu().numpy()
                frame["cell_activity"] = model.cell_activity(z).cpu().numpy()
                frame["favorable"] = favorable.astype(np.int8)
                particles.append(frame)
            print(f"Paired trajectory: {treatment}, seed {seed}", flush=True)
    return pd.DataFrame(summaries), pd.concat(particles, ignore_index=True)


def functional_states(particles, reference):
    """Recreate the saved state columns and centers using the original seven-state fit."""
    observed = reference[SCORE_COLS].to_numpy(float)
    gmm = GaussianMixture(n_components=7, covariance_type="full", reg_covar=1e-4,
                          n_init=15, max_iter=700, random_state=2026).fit(observed)
    cost = ((gmm.means_[:, None] - STATE_TEMPLATES[None]) ** 2).sum(axis=2)
    components, states = linear_sum_assignment(cost)
    order = components[np.argsort(states)]
    centers = gmm.means_[order]
    center_table = pd.DataFrame(centers, columns=SCORE_COLS)
    center_table.insert(0, "state", STATES)
    center_table["reference_weight"] = gmm.weights_[order]

    values = particles[SCORE_COLS].to_numpy(float)
    probabilities = gmm.predict_proba(values)[:, order]
    pca = PCA(n_components=2, random_state=2026).fit(observed)
    coordinates = pca.transform(values)
    result = particles.copy()
    for j, slug in enumerate(STATE_SLUGS):
        result[f"p_{slug}"] = probabilities[:, j].astype(np.float32)
    result["state"] = np.asarray(STATES, dtype=object)[probabilities.argmax(axis=1)]
    result["state_confidence"] = probabilities.max(axis=1).astype(np.float32)
    result["state_entropy"] = (-np.sum(probabilities * np.log(probabilities + 1e-12), axis=1) / np.log(7)).astype(np.float32)
    result["ood_flag"] = gmm.score_samples(values) < np.quantile(gmm.score_samples(observed), 0.01)
    result[["PC1", "PC2"]] = coordinates.astype(np.float32)
    precursor, cytotoxic, dysfunctional = centers[[0, 3, 4]]
    axes = [
        ("cyto_progress", precursor, cytotoxic - precursor, 1.0),
        ("dysf_progress", precursor, dysfunctional - precursor, 1.0),
        ("cyto_dysf_balance", (cytotoxic + dysfunctional) / 2, cytotoxic - dysfunctional, 2.0),
    ]
    for column, origin, direction, scale in axes:
        result[column] = (scale * ((values - origin) @ direction) / np.dot(direction, direction)).astype(np.float32)
    return result, center_table


def generate_inputs(cfg):
    """Calculate the five input datasets in memory; do not train or write files."""
    torch.set_num_threads(int(cfg["cpu_threads"]))
    model, _ = load_final_reaction(REPO, cfg)
    from common.cd8_zdist_data import load_training_package
    from scripts.reproduce.upstream_paths import OBSERVED_VOLUME

    groups, _, _, _ = load_training_package(REPO)
    volume = pd.read_csv(REPO / OBSERVED_VOLUME)
    volume0 = float(volume.loc[volume.treatment_clean.eq("Control") & volume.day.eq(1), "tumor_volume"].mean())
    reference = observed_reference(groups)
    summary, particles = paired_trajectories(model, groups, volume0, reference)
    particles, centers = functional_states(particles, reference)
    observed = reference[SCORE_COLS].to_numpy(float)
    projection = DiffusionMapNystrom().fit(observed)
    return {
        "observed_score_reference_balanced.csv.gz": reference,
        "paired_timeseries.csv": summary,
        "particle_functional_states.csv.gz": particles,
        "state_centers.csv": centers,
    }, projection


def run(cfg, output):
    output = new_output(output)
    tables, projection = generate_inputs(cfg)
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    projection.save(output / "frozen_nystrom_state.npz")
    return output
