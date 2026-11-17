"""Names and ordering used by the retained model figures."""

TREATMENT_ORDER = ["Control", "Anti-PD1", "Anti-41BB", "Anti-PD1+Anti-41BB"]


TREATMENT_LABELS = {"Control": "Control", "Anti-PD1": "anti-PD-1", "Anti-41BB": "Anti-4-1BB", "Anti-PD1+Anti-41BB": "Combination"}


TREATMENT_COLORS = {
    "Control": "#4D4D4D",
    "Anti-PD1": "#0F4D92",
    "Anti-41BB": "#42949E",
    "Anti-PD1+Anti-41BB": "#B64342",
}


HELDOUT_DAY_ORDER = [4, 10, 14, 17]


PRIMARY_DYNAMICS_METHOD_ORDER = [
    "persistence",
    "treatment_linear_shift",
    "entropic_ot_barycentric",
    "project_balanced_sinkhorn_drift",
    "mioflow",
    "tigon",
    "official_ot_cfm",
]


METHOD_LABELS = {
    "persistence": "Persistence", "mioflow": "MIOFlow", "treatment_linear_shift": "Linear shift",
    "project_balanced_sinkhorn_drift": "MultiNeD",
    "entropic_ot_barycentric": "Entropic OT", "tigon": "TIGON", "official_ot_cfm": "OT-CFM",
}


DYNAMICS_METRICS = (
    "sinkhorn_divergence",
    "energy_distance",
    "score_mean_mae",
    "wasserstein_1",
)


COMPONENT_ABLATION_ORDER = [
    "balanced_ot_local_rollout_pretrained",
    "balanced_ot_local_only_pretrained",
    "balanced_ot_rollout_only_pretrained",
    "balanced_ot_local_rollout_no_pretrain",
]

FULL_DATA_ABLATION_ORDER = [*COMPONENT_ABLATION_ORDER, "nearest_neighbor_mse_local_rollout_pretrained"]


FULL_DATA_ABLATION_LABELS = {
    "balanced_ot_local_rollout_pretrained": "Full model", "balanced_ot_local_only_pretrained": "Local only",
    "balanced_ot_rollout_only_pretrained": "Rollout only", "balanced_ot_local_rollout_no_pretrain": "No pretraining",
    "nearest_neighbor_mse_local_rollout_pretrained": "Nearest-neighbour MSE",
}
REFERENCE_VARIANT = "balanced_ot_local_rollout_pretrained"


VARIANT_LABELS = {
    REFERENCE_VARIANT: "Balanced OT: local + rollout, pretrained",
    "balanced_ot_local_only_pretrained": "Balanced OT: local only, pretrained",
    "balanced_ot_rollout_only_pretrained": "Balanced OT: rollout only, pretrained",
    "balanced_ot_local_rollout_no_pretrain": "Balanced OT: local + rollout, no pretraining",
    "nearest_neighbor_mse_local_rollout_pretrained": "Nearest-neighbour MSE: local + rollout",
}
