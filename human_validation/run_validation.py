"""Run human validation with explicit upstream results and a shared output workspace."""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from common.paths import HUMAN_ROOT, HumanInputs

COMMANDS = {
    "tcga": ("tcga", "TCGA_SKCM"),
    "liu": ("liu", "Liu2019"),
    "longitudinal": ("longitudinal", "GSE272993_longitudinal/results/cd8score_longitudinal_validation"),
    "five-program-ridge": ("five_program_ridge", "TCGA_SKCM/five_program_cox"),
    "gene-ridge": ("gene_ridge", "TCGA_SKCM/gene_level_cd8score/gene_ridge_cox"),
    "fixed-topk": ("fixed_topk", "TCGA_SKCM/gene_level_cd8score/fixed_topk"),
    "decompose": ("decompose", "TCGA_SKCM/gene_level_cd8score"),
    "top50": ("top50", "TCGA_SKCM/gene_level_cd8score/top_gene_enrichment"),
    "enrichment-all": ("enrichment_all", "TCGA_SKCM/gene_level_cd8score/all_cd8score_enrichment"),
    "enrichment-sign": ("enrichment_sign", "TCGA_SKCM/gene_level_cd8score/all_cd8score_enrichment"),
    "table-s3-cox": ("table_s3_cox", "TCGA_SKCM/gene_level_cd8score/gene_ridge_cox"),
}


def main():
    defaults = HumanInputs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--input-root", type=Path, default=defaults.results,
                        help="Upstream human results and longitudinal expression export")
    parser.add_argument("--dynamics-dir", type=Path, default=defaults.dynamics)
    parser.add_argument("--mouse-trajectories", type=Path, default=defaults.mouse_trajectories)
    parser.add_argument("--output", type=Path, required=True,
                        help="human_validation for the standard layout, or a separate workspace under human_validation/runs")
    args = parser.parse_args()
    workspace = args.output.resolve()
    runs = HUMAN_ROOT / "runs"
    if workspace != HUMAN_ROOT and (workspace == runs or not workspace.is_relative_to(runs)):
        raise ValueError(f"--output must be {HUMAN_ROOT} or a subdirectory of {runs}")
    module_name, relative = COMMANDS[args.command]
    output = workspace / relative
    output.mkdir(parents=True, exist_ok=True)
    inputs = HumanInputs(args.input_root.resolve(), args.dynamics_dir.resolve(),
                         args.mouse_trajectories.resolve())
    importlib.import_module("analyses." + module_name).run(inputs, output)


if __name__ == "__main__":
    main()
