"""DiFCoN paper training, ablations and evaluation. No plotting is performed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


# 1. Paper experiments and fixed seeds. Training uses the existing base.json.
ROOT = Path(__file__).resolve().parent
POLICY_STUDIES = {
    "baseline": "baseline",
    "dose-sweep": "dose_weight_sweep",
    "objective-ablation": "objective_ablation",
    "observation-history": "observation_history_ablation",
    "openloop": "openloop_comparison",
    "benchmark": "locked_test_benchmark",
    "attribution": "input_attribution",
}
SEEDS = (2031, 2032, 2033)
DOSE_WEIGHTS = (0.0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


# 2. Build commands without importing models, loading weights or creating files.
def policy_commands(task, device, destination, *, policy_root="outputs/dose_weight_sweep",
                    openloop_source=None, sac_root=None, model_inputs=None):
    """Build the paper recipes for the selected destination.

    Evaluation recipes read the explicitly selected policy runs and their saved models.
    All commands are run from DiFCoN/ with the active Python interpreter.
    """
    study = POLICY_STUDIES[task]
    result = destination.as_posix()
    commands = []
    policy_root = Path(policy_root).as_posix()

    def train(name, seed, *settings):
        command = ["train_DiFCoN.py", "--config", "configs/direct_dose/base.json",
                   "--study-name", study, "--output-dir", str(destination / name), "--device", device]
        for setting in (f"seeds.model={seed}", f"seeds.batch_order={seed}",
                        f"seeds.train_rollout={10000 + seed}", *settings):
            command.extend(["--set", setting])
        for key, value in (model_inputs or {}).items():
            command.extend(["--set", f"model.{key}={json.dumps(value)}"])
        commands.append(command)

    # 2a. Baseline training: seed 2031, full state, H=5, original 200-epoch budget.
    if study == "baseline":
        train("seed_2031", 2031)
    # 2b. Dose penalty: eight weights, three seeds, then aggregate the 24 fits.
    elif study == "dose_weight_sweep":
        for weight in DOSE_WEIGHTS:
            for seed in SEEDS:
                train(f"dose_weight_{str(weight).replace('.', 'p')}_seed_{seed}", seed,
                      f"loss.total_dose_weight={weight}")
        commands.append(["scripts/experiments/summarize_policy_sweep.py", "--study-dir", result])
    # 2c. Four objective ablations; Full is the selected seed-2031 reference.
    elif study == "objective_ablation":
        for name, field in (("no_final", "final_tumor_weight"), ("no_auc", "tumor_auc_weight"),
                            ("no_functional", "functional_activity_weight"), ("no_efficiency", "efficiency_weight")):
            train(f"{name}_seed_2031", 2031, f"loss.{field}=0.0")
    # 2d. Practical/schedule-only observations and H=3/4, each with three seeds.
    elif study == "observation_history_ablation":
        practical = '["previous_action_pd1","previous_action_b41bb","log_C_abs","log_C_rel","remaining_budget_fraction","time_fraction"]'
        schedule = '["remaining_budget_fraction","time_fraction"]'
        for seed in SEEDS:
            train(f"practical_observation_seed_{seed}", seed, f"policy.feature_names={practical}")
            train(f"schedule_only_seed_{seed}", seed, f"policy.feature_names={schedule}")
            for history in (3, 4):
                train(f"history_length_{history}_seed_{seed}", seed, f"policy.history_length={history}")
    # 2e. Validation-derived open-loop schedules versus frozen feedback policies.
    elif study == "openloop_comparison":
        commands.append(["scripts/evaluation/evaluate_openloop_feedback.py", "--run-root", policy_root,
                         "--cohort-dir", "cohorts", "--output-dir", result,
                         "--seeds", *map(str, SEEDS), "--device", device, "--n-particles", "96"])
    # 2f. Locked test benchmark, including the saved SAC policies, then aggregate.
    elif study == "locked_test_benchmark":
        runs = []
        for seed in SEEDS:
            run_id = f"seed_{seed}"
            runs.append(f"{result}/{run_id}")
            commands.append(["-m", "scripts.benchmarking.run_policy_benchmark", "--config",
                             "configs/benchmarking/locked_test_benchmark.json",
                             "--cohort", "test", "--confirm-hyperparameters-frozen", "--model-seed", str(seed),
                             "--device", device, "--run-id", run_id, "--output-root", result,
                              "--policy-root", policy_root])
            if openloop_source:
                commands[-1].extend(["--openloop-source", str(openloop_source)])
            if sac_root:
                commands[-1].extend(["--sac-root", str(sac_root)])
        commands.append(["-m", "scripts.benchmarking.aggregate_policy_benchmark", "--run-dir", *runs,
                         "--output-dir", f"{result}/combined", "--reference", "Amortized 1D-CNN"])
    # 2g. Input-feature and history-step attribution on the validation cohort.
    elif study == "input_attribution":
        commands.append(["scripts/analysis/analyze_policy_input_attribution.py", "--config",
                         "configs/analysis/input_attribution.json",
                         "--output-dir", f"{result}/full_validation", "--device", device,
                         "--policy-root", policy_root])
    else:
        raise ValueError(study)
    return commands


# 3. Record the selected Full-objective reference for the ablation tables.
def write_policy_references(destination, policy_root):
    """Record the selected Full-objective reference after the ablation finishes."""
    target = destination / "BASELINE_REFERENCES.tsv"
    policy_root = Path(policy_root).resolve()
    if policy_root.is_relative_to(ROOT):
        policy_root = policy_root.relative_to(ROOT)
    rows = ["lambda_d\texisting_200epoch_reference"]
    rows.extend(f"{w}\t{Path(policy_root).as_posix()}/dose_weight_{str(w).replace('.', 'p')}_seed_2031"
                for w in DOSE_WEIGHTS)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


# 4. Command-line entry.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=POLICY_STUDIES)
    parser.add_argument("--device", default="cuda:0")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", type=Path,
                        help="Study directory (relative to the current directory). Defaults to DiFCoN/outputs/<experiment>.")
    output.add_argument("--run-name", help="Optional suffix for a study under DiFCoN/outputs")
    parser.add_argument("--plan", action="store_true", help="Print commands without running or writing files")
    parser.add_argument("--policy-root", type=Path, default=ROOT / "outputs/dose_weight_sweep",
                        help="Dose-sweep study used for evaluation and Full-ablation references")
    parser.add_argument("--openloop-source", type=Path, help="Open-loop result directory used by benchmark")
    parser.add_argument("--sac-root", type=Path, help="Frozen SAC result directory containing the per-seed actor checkpoints")
    parser.add_argument("--drift-source", help="Pretrained drift checkpoint or run/result directory for training")
    parser.add_argument("--reaction-source", help="Joint checkpoint or run/result directory for training")
    args = parser.parse_args()
    args.policy_root = args.policy_root.resolve()
    for name in ("openloop_source", "sac_root"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    model_inputs = {}
    if args.drift_source or args.reaction_source:
        if args.experiment not in ("baseline", "dose-sweep", "objective-ablation", "observation-history"):
            parser.error("Evaluation restores model inputs from each policy run's resolved_config.json")
        sys.path.insert(0, str(ROOT.parent / "MultiNeD/scripts/reproduce"))
        from upstream_paths import checkpoint_path, DRIFT_NAME, REACTION_NAME
        for value, filename, key in ((args.drift_source, DRIFT_NAME, "drift_checkpoint"),
                                     (args.reaction_source, REACTION_NAME, "reaction_checkpoint")):
            if value:
                path = checkpoint_path(value, filename, ROOT.parent)
                # Repository-relative paths keep an in-repository run portable after cloning.
                model_inputs[key] = path.relative_to(ROOT.parent).as_posix() if path.is_relative_to(ROOT.parent) else str(path)
    study = POLICY_STUDIES[args.experiment]
    if args.run_name is not None:
        tag = args.run_name
        if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
            parser.error("--run-name must contain only letters, numbers, underscores or hyphens")
        study = f"{study}_{tag}"
    destination = args.output_dir.resolve() if args.output_dir else ROOT / "outputs" / study
    commands = policy_commands(args.experiment, args.device, destination, policy_root=args.policy_root,
                               openloop_source=args.openloop_source, sac_root=args.sac_root, model_inputs=model_inputs)
    if args.plan:
        print(json.dumps({"experiment": args.experiment, "cwd": str(ROOT), "output": str(destination),
                          "commands": commands, "executed": False}, indent=2))
        return
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for command in commands:
        subprocess.run([sys.executable, "-B", "-u", *command], cwd=ROOT, env=env, check=True)
    if args.experiment == "objective-ablation":
        write_policy_references(destination, args.policy_root)


if __name__ == "__main__":
    main()
