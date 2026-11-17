"""Nested-CV Ridge Cox with training-fold scaling and held-out risk predictions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from statsmodels.duration.hazard_regression import PHReg
PROGRAMS = ["Precursor", "Activation", "Early effector", "Cytotoxic", "Dysfunction"]
RIDGE_PENALTIES = np.array([1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0,3.0,10.0])
OUTER_FOLDS = 5
INNER_FOLDS = 4
RANDOM_SEED = 20260919

def standardize_training(
    training: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = training.mean(axis=0)
    sd = training.std(axis=0, ddof=0)
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError(f"Invalid training-fold predictor SD: {sd}")
    return (training - mean) / sd, (validation - mean) / sd, mean, sd


def fit_ridge_cox(
    predictors: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """Fit Efron Ridge Cox using the statsmodels likelihood and L2 objective."""

    model = PHReg(time, predictors, status=event, ties="efron")
    n = len(time)

    def objective(beta: np.ndarray) -> float:
        return float(-model.loglike(beta) / n + penalty * np.dot(beta, beta) / 2.0)

    def gradient(beta: np.ndarray) -> np.ndarray:
        return -model.score(beta) / n + penalty * beta

    fit = minimize(
        objective,
        np.zeros(predictors.shape[1]),
        jac=gradient,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    if fit.success and np.isfinite(fit.x).all():
        return fit.x.astype(float)

    # At strong penalties L-BFGS-B can report an abnormal line-search stop
    # after reaching a numerically stationary point.  Retry with full BFGS,
    # then accept only a finite solution with a small analytic gradient.
    start = fit.x if np.isfinite(fit.x).all() else np.zeros(predictors.shape[1])
    retry = minimize(
        objective,
        start,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    gradient_norm = float(np.linalg.norm(gradient(retry.x), ord=np.inf))
    if np.isfinite(retry.x).all() and np.isfinite(retry.fun) and gradient_norm <= 1e-6:
        return retry.x.astype(float)
    raise RuntimeError(
        f"Ridge Cox failed at penalty={penalty:g}: "
        f"L-BFGS-B={fit.message}; BFGS={retry.message}; "
        f"gradient_inf_norm={gradient_norm:.3g}"
    )


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> dict[str, float | int]:
    """Harrell C-index with higher score indicating higher event risk.

    An event is comparable with observations at later times and with censored
    observations at the same time. Events tied in time are not ordered.
    """

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    if not (len(time) == len(event) == len(risk)) or not np.isfinite(time).all() or not np.isfinite(risk).all():
        raise ValueError("Invalid arrays for Harrell C-index")

    order = np.argsort(time, kind="mergesort")
    concordant = 0
    discordant = 0
    tied_risk = 0
    tied_time = 0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and time[order[end]] == time[order[index]]:
            end += 1
        same_time = order[index:end]
        censored_same_time = same_time[~event[same_time]]
        later = order[end:]
        for event_index in same_time[event[same_time]]:
            comparable = np.concatenate([censored_same_time, later])
            if len(censored_same_time):
                tied_time += len(censored_same_time)
            if not len(comparable):
                continue
            differences = risk[event_index] - risk[comparable]
            ties = np.isclose(differences, 0.0, rtol=0, atol=1e-8)
            tied_risk += int(ties.sum())
            concordant += int((differences > 1e-8).sum())
            discordant += int((differences < -1e-8).sum())
        index = end

    comparable_pairs = concordant + discordant + tied_risk
    if comparable_pairs == 0:
        raise ValueError("No comparable survival pairs for Harrell C-index")
    c_index = (concordant + 0.5 * tied_risk) / comparable_pairs
    return {
        "harrell_c_index": float(c_index),
        "comparable_pairs": int(comparable_pairs),
        "concordant_pairs": int(concordant),
        "discordant_pairs": int(discordant),
        "tied_risk_pairs": int(tied_risk),
        "tied_time_pairs": int(tied_time),
    }


def select_ridge_penalty(
    predictors: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cohort: str,
    outer_fold: int,
) -> tuple[float, list[dict[str, object]]]:
    random_state = RANDOM_SEED + 100 * outer_fold + (
        0 if cohort == "All tumors" else 10_000
    )
    inner = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=random_state,
    )
    standardized_folds = []
    for inner_train, inner_valid in inner.split(predictors, event):
        train_x, valid_x, _, _ = standardize_training(
            predictors[inner_train], predictors[inner_valid]
        )
        standardized_folds.append((inner_train, inner_valid, train_x, valid_x))
    rows = []
    for penalty in RIDGE_PENALTIES:
        concordant = 0
        discordant = 0
        tied_risk = 0
        tied_time = 0
        for inner_train, inner_valid, train_x, valid_x in standardized_folds:
            beta = fit_ridge_cox(train_x, time[inner_train], event[inner_train], float(penalty))
            stats = harrell_c_index(time[inner_valid], event[inner_valid], valid_x @ beta)
            concordant += int(stats["concordant_pairs"])
            discordant += int(stats["discordant_pairs"])
            tied_risk += int(stats["tied_risk_pairs"])
            tied_time += int(stats["tied_time_pairs"])
        comparable = concordant + discordant + tied_risk
        c_index = (concordant + 0.5 * tied_risk) / comparable
        rows.append(
            {
                "cohort": cohort,
                "outer_fold": outer_fold,
                "ridge_penalty_lambda": float(penalty),
                "inner_folds": INNER_FOLDS,
                "inner_fold_weighted_c_index": float(c_index),
                "comparable_pairs": comparable,
                "concordant_pairs": concordant,
                "discordant_pairs": discordant,
                "tied_risk_pairs": tied_risk,
                "tied_time_pairs": tied_time,
            }
        )
    best_c = max(row["inner_fold_weighted_c_index"] for row in rows)
    selected = max(
        row["ridge_penalty_lambda"]
        for row in rows
        if row["inner_fold_weighted_c_index"] >= best_c - 1e-12
    )
    for row in rows:
        row["selected_for_outer_fold"] = bool(np.isclose(row["ridge_penalty_lambda"], selected))
    return float(selected), rows


def ridge_oof(
    frame: pd.DataFrame,
    cohort: str,
    *,
    feature_columns: list[str] | None = None,
    outer_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Generate TCGA-SKCM OOF predictions with optional gene features and shared outer folds."""

    feature_columns = (
        PROGRAMS if feature_columns is None else list(feature_columns)
    )
    selected_columns = ["sample_id", "patient_id", "os_time_days", "os_event", *feature_columns]
    data = frame[selected_columns].dropna().copy()
    data = data[data["os_time_days"] > 0].reset_index(drop=True)
    predictors = data[feature_columns].to_numpy(dtype=float)
    time = data["os_time_days"].to_numpy(dtype=float)
    event = data["os_event"].astype(int).to_numpy()
    if outer_splits is None:
        outer_random_state = RANDOM_SEED + (0 if cohort == "All tumors" else 10_000)
        outer = StratifiedKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=outer_random_state,
        )
        split_indices = list(outer.split(predictors, event))
    else:
        split_indices = [
            (np.asarray(training, dtype=int), np.asarray(heldout, dtype=int))
            for training, heldout in outer_splits
        ]
        if len(split_indices) != OUTER_FOLDS:
            raise ValueError(f"Expected {OUTER_FOLDS} outer splits")
        for training, heldout_fold in split_indices:
            if np.intersect1d(training, heldout_fold).size:
                raise ValueError("An outer split has overlapping training and test rows")
        heldout = np.concatenate([test for _, test in split_indices])
        if sorted(heldout.tolist()) != list(range(len(data))):
            raise ValueError(
                "Provided outer_splits must hold out every row exactly once"
            )

    oof_raw = np.full(len(data), np.nan)
    oof_calibrated = np.full(len(data), np.nan)
    fold_assignment = np.full(len(data), -1, dtype=int)
    weight_rows = []
    tuning_rows = []
    fold_pair_totals = {"concordant": 0, "discordant": 0, "tied_risk": 0, "comparable": 0}

    for fold, (training_index, heldout_index) in enumerate(split_indices, start=1):
        selected_penalty, fold_tuning = select_ridge_penalty(
            predictors[training_index],
            time[training_index],
            event[training_index],
            cohort,
            fold,
        )
        tuning_rows.extend(fold_tuning)
        train_x, heldout_x, train_mean, train_sd = standardize_training(
            predictors[training_index], predictors[heldout_index]
        )
        beta = fit_ridge_cox(
            train_x,
            time[training_index],
            event[training_index],
            selected_penalty,
        )
        train_risk = train_x @ beta
        heldout_risk = heldout_x @ beta
        train_risk_mean = float(train_risk.mean())
        train_risk_sd = float(train_risk.std(ddof=0))
        if not np.isfinite(train_risk_sd) or train_risk_sd <= 0:
            raise ValueError(f"Invalid training-risk SD in {cohort}, fold {fold}")
        calibrated = (heldout_risk - train_risk_mean) / train_risk_sd
        oof_raw[heldout_index] = heldout_risk
        oof_calibrated[heldout_index] = calibrated
        fold_assignment[heldout_index] = fold

        heldout_stats = harrell_c_index(time[heldout_index], event[heldout_index], heldout_risk)
        fold_pair_totals["concordant"] += int(heldout_stats["concordant_pairs"])
        fold_pair_totals["discordant"] += int(heldout_stats["discordant_pairs"])
        fold_pair_totals["tied_risk"] += int(heldout_stats["tied_risk_pairs"])
        fold_pair_totals["comparable"] += int(heldout_stats["comparable_pairs"])

        for index, program in enumerate(feature_columns):
            weight_row = {
                "cohort": cohort,
                "outer_fold": fold,
                "program": program,
                "ridge_penalty_lambda": selected_penalty,
                "coefficient_for_training_standardized_program": beta[index],
                "coefficient_per_training_risk_sd": beta[index] / train_risk_sd,
                "training_program_mean": train_mean[index],
                "training_program_sd_ddof0": train_sd[index],
                "training_risk_mean": train_risk_mean,
                "training_risk_sd_ddof0": train_risk_sd,
                "training_n": len(training_index),
                "training_events": int(event[training_index].sum()),
                "heldout_n": len(heldout_index),
                "heldout_events": int(event[heldout_index].sum()),
                "heldout_fold_c_index": heldout_stats["harrell_c_index"],
                "ties_method": "efron",
            }
            weight_rows.append(weight_row)

    if np.isnan(oof_calibrated).any() or (fold_assignment < 1).any():
        raise AssertionError(f"Incomplete OOF predictions for {cohort}")
    data.insert(0, "cohort", cohort)
    data["outer_fold"] = fold_assignment
    data["ridge_oof_linear_predictor_raw"] = oof_raw
    data["ridge_oof_risk_score"] = oof_calibrated
    weighted_c = (
        fold_pair_totals["concordant"] + 0.5 * fold_pair_totals["tied_risk"]
    ) / fold_pair_totals["comparable"]
    fold_summary = {
        "outer_fold_weighted_c_index": float(weighted_c),
        "outer_fold_comparable_pairs": int(fold_pair_totals["comparable"]),
    }
    return data, pd.DataFrame(weight_rows), pd.DataFrame(tuning_rows), fold_summary
