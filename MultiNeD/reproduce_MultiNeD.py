"""MultiNeD training and paper experiments.

drift/joint: cellular pretraining and multiscale joint calibration.
volume-loto: held-out tumour-volume predictions.
cell-loto/cell-reconstruction: cellular-distribution method comparisons.
loss-ablation: five training-objective/pretraining variants.
compare-pretrained: fixed pretrained drift versus complete MultiNeD; no training.

Use --plan to print the selected experiment without execution.
Plotting remains in the independent paper notebooks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# 1. Repository paths and shared model-input helpers.
CODE = Path(__file__).resolve().parent
REPO = CODE.parent
sys.path.insert(0, str(CODE))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONUTF8"] = "1"
sys.dont_write_bytecode = True

from scripts.reproduce.upstream_paths import (
    DEFAULT_DRIFT_CHECKPOINT, DEFAULT_REACTION_CHECKPOINT,
    DRIFT_NAME, PARAMETER_NAME, QIN_NAME, REACTION_NAME, absolute_path, checkpoint_path, require_file,
)

# 2. Paper training, validation and cellular-distribution benchmarks.
TASKS = {
    "drift": (DEFAULT_DRIFT_CHECKPOINT.parent.name, None, None, None),
    "joint": (DEFAULT_REACTION_CHECKPOINT.parent.name, None, None, None),
    "volume-loto": ("volume_loto", "experiments.volume_loto.run_volume_loto",
                    "configs/validation/volume_loto_common_days.json", 48),
    "cell-loto": ("cell_loto_benchmark", "experiments.benchmarks.run_cell_state_benchmark",
                  "configs/benchmarking/cell_loto_benchmark.json", 42),
    "cell-reconstruction": ("cell_reconstruction_benchmark", "experiments.benchmarks.run_cell_state_benchmark",
                            "configs/benchmarking/cell_reconstruction_benchmark.json", 6),
    "loss-ablation": ("training_loss_ablation", "experiments.benchmarks.run_full_data_loss_ablation",
                      "configs/benchmarking/training_loss_ablation.json", 5),
    "compare-pretrained": ("pretrained_vs_MultiNeD", None, None, None),
}
CELL_BENCHMARK_TASKS = ("cell-loto", "cell-reconstruction")
RUN_PREFIXES = {
    "cell-loto": "b1_shared_loto_all7",
    "cell-reconstruction": "b1_full_data",
    "loss-ablation": "full_data_loss",
}


# 3. Resolve each experiment's saved configuration, seeds and training budget.
def jobs_for(task, work, input_overrides):
    """Prepare the job configs without writing to the paper configuration files."""
    _, module, relative, expected = TASKS[task]
    if module is None:
        return []
    template = json.loads((CODE / relative).read_text(encoding="utf-8"))
    if task == "volume-loto":
        specs = [("common4", template)]
        result_root = work / "results"
    else:
        prefix = RUN_PREFIXES[task]
        specs = [(f"{prefix}_seed{seed}", {**template, "seeds": [seed]})
                 for seed in template["seeds"]]
        result_root = work
    config_dir = work / "configs"
    jobs = []
    for label, config in specs:
        if task == "volume-loto":
            config["inputs"].update(input_overrides)
            config["output"]["root"] = str(result_root)
            base = None
        else:
            base_path = REPO / config["base_validation_config"]
            base = json.loads(base_path.read_text(encoding="utf-8"))
            base["inputs"].update(input_overrides)
            config["base_validation_config"] = str(config_dir / f"{label}_base.json")
            config["output_root"] = str(result_root)
        jobs.append({"label": label, "module": module, "config": config,
                     "config_path": config_dir / f"{label}.json", "base": base,
                     "result": result_root / label, "expected": expected})
    return jobs


# 4. External comparison methods: MIOFlow, TIGON and OT-CFM.
def external_environment(methods):
    """Use installed benchmark environments or explicit environment overrides."""
    external_root = Path(os.environ.get("EXTERNAL_BASELINE_ROOT", REPO.parent / "external_baselines"))
    conda_base = Path(os.environ.get("CONDA_EXE", sys.executable)).resolve()
    # Current env Python: <conda>/envs/<name>/python.exe or bin/python.
    candidates = [p for p in conda_base.parents if (p / "envs").is_dir()]
    conda_base = candidates[0] if candidates else Path(sys.prefix).parent
    definitions = {"mioflow": ("MIOFLOW", "MIOFlow", "multined_mioflow"),
                   "tigon": ("TIGON", "TIGON", "multined_tigon"),
                   "official_ot_cfm": ("OTCFM", "conditional-flow-matching", "multined_otcfm")}
    for method in methods:
        if method not in definitions:
            continue
        prefix, repo_name, env_name = definitions[method]
        os.environ.setdefault(prefix + "_REPO", str(external_root / repo_name))
        executable = conda_base / "envs" / env_name / ("python.exe" if os.name == "nt" else "bin/python")
        os.environ.setdefault(prefix + "_PYTHON", str(executable))


def command(module, args):
    cmd = [sys.executable, "-B", "-u", "-m", module, *map(str, args)]
    print("RUN", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=CODE, check=True)


# 5. Aggregate the chosen experiment's saved predictions and metrics.
def aggregate(task, jobs, work):
    roots = [str(j["result"]) for j in jobs]
    if task in CELL_BENCHMARK_TASKS:
        combined = work / f"{RUN_PREFIXES[task]}_combined"
        command("experiments.benchmarks.aggregate_dynamics_benchmark", ["--run-dir", *roots, "--output-dir", combined])
    elif task == "loss-ablation":
        combined = work / "full_data_loss_all_seeds_combined"
        command("experiments.benchmarks.aggregate_full_data_loss_ablations", ["--run-dir", *roots, "--output-dir", combined])
    elif task == "volume-loto":
        combined = work / "combined"
        import pandas as pd
        # Merge the 48 newly generated common-day folds.
        tables = [pd.read_csv(p) for root in map(Path, roots) for p in root.glob("m1/seed_*/*/heldout_volume_metrics.csv")]
        frame = pd.concat(tables, ignore_index=True)
        combined.mkdir(parents=True, exist_ok=True)
        frame.to_csv(combined / "all_heldout_volume_metrics.csv", index=False)


# 6. Command-line entry. All tasks use the formal paper settings.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--plan", action="store_true", help="Print settings without running or writing files")
    parser.add_argument("--device", help="Torch device; use the paper configuration by default")
    parser.add_argument("--output-dir", type=Path,
                        help="Result directory (relative to the repository root). Defaults to MultiNeD/outputs/<experiment>.")
    parser.add_argument("--drift-source", help="Drift checkpoint or exact run/result directory")
    parser.add_argument("--reaction-source", help="Joint checkpoint or exact run/result directory")
    args = parser.parse_args()
    task = args.task
    output_name, _, _, _ = TASKS[task]
    if args.drift_source and task == "drift":
        parser.error("drift pretraining has no upstream drift input")
    if args.reaction_source and task not in ("volume-loto", "loss-ablation", "compare-pretrained"):
        parser.error("--reaction-source applies to volume-loto, loss-ablation and compare-pretrained")
    inputs = {}
    if args.drift_source:
        inputs["drift_checkpoint"] = str(checkpoint_path(args.drift_source, DRIFT_NAME, REPO))
    if args.reaction_source:
        checkpoint = checkpoint_path(args.reaction_source, REACTION_NAME, REPO)
        inputs["reaction_checkpoint"] = str(checkpoint)
        if task in ("volume-loto", "loss-ablation"):
            inputs["reaction_parameter_settings"] = require_file(checkpoint.parent / PARAMETER_NAME)
        if task == "loss-ablation":
            inputs["q_in_npy"] = require_file(checkpoint.parent / QIN_NAME)
    work = absolute_path(args.output_dir, REPO) if args.output_dir else CODE / "outputs" / output_name
    jobs = jobs_for(task, work, inputs)
    if args.device and task == "volume-loto":
        config = jobs[0]["config"]
        config["drift"]["device"] = args.device
        config["reaction"]["device"] = args.device
    if args.plan:
        print(json.dumps({"task": output_name, "device": args.device, "output": str(work),
                          "planned_fits": [j["expected"] for j in jobs],
                          "configs": [j["config"] for j in jobs],
                          "base_configs": [j["base"] for j in jobs if j["base"] is not None],
                          "input_overrides": inputs, "executed": False}, indent=2))
        return
    print(f"TASK={output_name} OUTPUT={work}", flush=True)
    if task == "compare-pretrained":
        from scripts.compare_pretrained import run
        run(work, device=args.device or "auto",
            drift_checkpoint=inputs.get("drift_checkpoint"),
            reaction_checkpoint=inputs.get("reaction_checkpoint"))
    elif not jobs:
        # Drift: 800 epochs. Joint cell/volume calibration: 600 epochs.
        from scripts.reproduce.native_training import run
        run(output_name, work, args.device, drift_checkpoint=inputs.get("drift_checkpoint"))
    else:
        if task in CELL_BENCHMARK_TASKS:
            external_environment(jobs[0]["config"]["methods"])
        (work / "configs").mkdir(parents=True, exist_ok=True)
        for job in jobs:
            config_path = job["config_path"]
            config = job["config"]
            files = [(config_path, config)]
            if job["base"] is not None:
                files.append((Path(config["base_validation_config"]), job["base"]))
            for path, value in files:
                path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            cli = ["--config", config_path, "--run-id", job["label"]]
            if args.device and task != "volume-loto":
                cli += ["--device", args.device]
            command(job["module"], cli)
        aggregate(task, jobs, work)
    print(f"COMPLETE {output_name}; results: {work}", flush=True)


if __name__ == "__main__":
    main()
