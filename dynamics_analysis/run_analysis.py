"""Generate model trajectories and run the paper dynamics analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from src.paths import BASE, INPUTS, REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "trajectories": "Generate the five baseline input datasets from fitted MultiNeD weights",
        "pseudocells": "Rebuild shared projections and pseudocell tables from saved inputs",
        "drift-source": "Evaluate frozen model drift at saved particle states (model computation)",
        "drift-field": "Prepare the display grid from saved direct-drift vectors",
        "perturb": "Run paired program perturbations (full simulation)",
        "importance": "Summarize alpha from saved perturbation metrics only",
    }
    for name, help_text in descriptions.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--output", type=Path, required=True, help="Directory for the new results")
        if name in {"trajectories", "drift-source", "perturb"}:
            command.add_argument("--config", type=Path, default=BASE / "configs/paper_analysis.json")
            command.add_argument("--device", default=None)
            command.add_argument("--reaction-source", type=Path, help="Full MultiNeD checkpoint or its exact run directory")
            command.add_argument("--drift-source", type=Path, help="Matching drift checkpoint for architecture/integrator settings")
        if name in {"pseudocells", "drift-source"}:
            command.add_argument("--inputs", type=Path, default=INPUTS, help="Trajectory input directory")
        if name == "perturb":
            command.add_argument("--inputs", type=Path, required=True, help="Weighted inputs from trajectories")
        if name == "drift-source":
            command.add_argument("--pseudocells", type=Path, default=BASE / "outputs/pseudocell/tables",
                                 help="Pseudocell table directory built from the selected inputs")
        if name == "drift-field":
            command.add_argument("--source", type=Path, default=None)
        if name == "perturb":
            command.add_argument("--deltas", type=float, nargs="+", default=(0.5, 0.25, 1.0),
                                 help="Positive SD magnitudes (default: 0.5 0.25 1.0)")
        if name == "importance":
            command.add_argument("--metrics", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "pseudocells":
        from src.pseudocells import run
        run(args.output, args.inputs)
    elif args.command == "drift-field":
        from src.drift_field import run_display
        run_display(args.output, args.source)
    elif args.command == "importance":
        from src.importance import run
        run(args.output, args.metrics)
    else:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        if args.device:
            cfg["device"] = args.device
        if args.reaction_source or args.drift_source:
            sys.path.insert(0, str(REPO))
            from MultiNeD.scripts.reproduce.upstream_paths import checkpoint_path, DRIFT_NAME, REACTION_NAME
            if args.reaction_source:
                cfg["reaction_checkpoint"] = str(checkpoint_path(args.reaction_source, REACTION_NAME, REPO))
            drift_source = args.drift_source or REPO / "MultiNeD/outputs/notebook_drift_run"
            cfg["drift_checkpoint"] = str(checkpoint_path(drift_source, DRIFT_NAME, REPO))
        if args.command == "trajectories":
            from src.trajectories import run
            run(cfg, args.output)
        elif args.command == "drift-source":
            from src.drift_field import run_source
            run_source(cfg, args.output, args.inputs.resolve(), args.pseudocells.resolve())
        else:
            from src.perturbations import run
            run(cfg, args.output, args.inputs.resolve(), args.deltas)
    print(f"Saved {args.command} results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
