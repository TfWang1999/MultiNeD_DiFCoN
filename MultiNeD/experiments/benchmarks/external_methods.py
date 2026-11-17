from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[2]


class ExternalSnapshotMethod:
    """Run a published baseline in its own Python environment.

    Training groups and transitions are passed with separate prediction tasks
    for cellular LOTO or full-data reconstruction. Each fit trains one shared
    treatment-conditioned model.
    """

    requires_tasks = True

    def __init__(self, method_id: str, settings: dict[str, Any]):
        self.method_id = str(method_id)
        self.settings = dict(settings)
        self.predictions_by_transition: dict[str, np.ndarray] = {}

    def _required_environment(self, name: str) -> str:
        value = self.settings.get(name)
        if value:
            return str(value)
        env_name = str(self.settings[f"{name}_env"])
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(
                f"{self.method_id} requires {env_name} to be set; see docs/REPRODUCIBILITY.md#external-benchmark-methods."
            )
        return value


    @staticmethod
    def _pack_ragged(blocks, *, label: str) -> tuple[np.ndarray, np.ndarray]:
        arrays = []
        offsets = [0]
        state_dim = None
        for index, block in enumerate(blocks):
            values = np.asarray(block, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] == 0:
                raise ValueError(f"Invalid {label} block {index}: {values.shape}")
            if state_dim is None:
                state_dim = int(values.shape[1])
            elif int(values.shape[1]) != state_dim:
                raise ValueError(f"All {label} blocks must share the same state dimension")
            arrays.append(values)
            offsets.append(offsets[-1] + int(values.shape[0]))
        if not arrays:
            raise ValueError(f"Cannot serialize empty {label} blocks")
        return np.concatenate(arrays, axis=0), np.asarray(offsets, dtype=np.int64)

    @staticmethod
    def _unpack_ragged(
        values,
        offsets,
        *,
        expected_count: int,
        label: str,
    ) -> list[np.ndarray]:
        values = np.asarray(values, dtype=np.float32)
        offsets = np.asarray(offsets, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError(f"Invalid {label} values: {values.shape}")
        if offsets.ndim != 1 or offsets.shape[0] != expected_count + 1:
            raise ValueError(f"{label} offsets do not match transition IDs")
        if offsets[0] != 0 or offsets[-1] != values.shape[0] or np.any(np.diff(offsets) <= 0):
            raise ValueError(f"Invalid {label} offsets")
        return [
            values[int(offsets[index]):int(offsets[index + 1])]
            for index in range(expected_count)
        ]

    @staticmethod
    def _write_input(
        output_dir: Path,
        group_data: dict,
        transitions,
        tasks: list[dict[str, Any]],
    ) -> Path:
        if not tasks:
            raise ValueError("Conditional external fit requires prediction tasks")
        required_columns = {
            "transition_id", "source_group", "target_group", "day0", "day1", "P", "B"
        }
        missing_columns = sorted(required_columns - set(transitions.columns))
        if missing_columns:
            raise ValueError(f"External transition table is missing: {missing_columns}")

        group_ids = sorted(map(str, group_data))
        group_index = {group_id: index for index, group_id in enumerate(group_ids)}
        blocks = []
        offsets = [0]
        z_dim = None
        for group_id in group_ids:
            values = np.asarray(group_data[group_id]["Z_all"], dtype=np.float32)
            if values.ndim != 2 or values.shape[0] == 0:
                raise ValueError(f"Invalid group values for {group_id}: {values.shape}")
            if z_dim is None:
                z_dim = int(values.shape[1])
            elif int(values.shape[1]) != z_dim:
                raise ValueError("All external training groups must share the same state dimension")
            blocks.append(values)
            offsets.append(offsets[-1] + int(values.shape[0]))

        transition_ids = transitions["transition_id"].astype(str).tolist()
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("External training transition IDs must be unique")
        task_ids = [str(task["transition_id"]) for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Conditional prediction task IDs must be unique")
        task_manifest_separated = all(
            bool(task.get("training_task_manifest_separated", False)) for task in tasks
        )
        if any(
            bool(task.get("training_task_manifest_separated", False))
            for task in tasks
        ) and not task_manifest_separated:
            raise ValueError("External tasks cannot mix manifest relationship modes")
        if not task_manifest_separated and set(task_ids) != set(transition_ids):
            raise ValueError(
                "Full-data external prediction tasks must match the complete transition manifest"
            )
        missing_groups = sorted(
            (
                set(transitions["source_group"].astype(str))
                | set(transitions["target_group"].astype(str))
            )
            - set(group_ids)
        )
        if missing_groups:
            raise ValueError(f"External transitions reference unknown groups: {missing_groups}")
        missing_task_sources = sorted(
            {str(task["source_group"]) for task in tasks} - set(group_ids)
        )
        if missing_task_sources:
            raise ValueError(
                f"External prediction tasks reference unavailable source groups: {missing_task_sources}"
            )

        task_source_values, task_source_offsets = ExternalSnapshotMethod._pack_ragged(
            (task["source_z"] for task in tasks), label="external task source"
        )
        input_path = output_dir / "distribution_inputs.npz"
        np.savez_compressed(
            input_path,
            training_group_values=np.concatenate(blocks, axis=0),
            training_group_offsets=np.asarray(offsets, dtype=np.int64),
            training_group_ids=np.asarray(group_ids),
            training_group_days=np.asarray(
                [float(group_data[group_id]["day"]) for group_id in group_ids], dtype=np.float32
            ),
            transition_ids=np.asarray(transition_ids),
            transition_source_group_indices=np.asarray(
                [group_index[str(value)] for value in transitions["source_group"]], dtype=np.int64
            ),
            transition_target_group_indices=np.asarray(
                [group_index[str(value)] for value in transitions["target_group"]], dtype=np.int64
            ),
            transition_source_days=transitions["day0"].astype(float).to_numpy(dtype=np.float32),
            transition_target_days=transitions["day1"].astype(float).to_numpy(dtype=np.float32),
            transition_P=transitions["P"].astype(float).to_numpy(dtype=np.float32),
            transition_B=transitions["B"].astype(float).to_numpy(dtype=np.float32),
            task_transition_ids=np.asarray(task_ids),
            task_source_group_ids=np.asarray([str(task["source_group"]) for task in tasks]),
            task_target_group_ids=np.asarray([str(task["target_group"]) for task in tasks]),
            task_source_values=task_source_values,
            task_source_offsets=task_source_offsets,
            task_source_days=np.asarray([float(task["source_day"]) for task in tasks], dtype=np.float32),
            task_target_days=np.asarray([float(task["target_day"]) for task in tasks], dtype=np.float32),
            task_P=np.asarray([float(task["P"]) for task in tasks], dtype=np.float32),
            task_B=np.asarray([float(task["B"]) for task in tasks], dtype=np.float32),
            evaluation_task_manifest_separated=np.asarray(
                [int(task_manifest_separated)], dtype=np.int64
            ),
            allow_disconnected_training_trajectories=np.asarray(
                [int(task_manifest_separated)], dtype=np.int64
            ),
        )
        return input_path

    def fit(self, group_data, transitions, seed: int, output_dir: Path, *, tasks) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_by_transition = {}
        input_path = self._write_input(
            output_dir, group_data, transitions, tasks
        )
        prediction_path = output_dir / "distribution_predictions.npz"
        adapter_settings = {
            key: value
            for key, value in self.settings.items()
            if key not in {"python", "python_env", "external_repo", "external_repo_env", "timeout_seconds"}
        }
        settings_path = output_dir / "external_settings.json"
        settings_path.write_text(
            json.dumps(adapter_settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        python_executable = self._required_environment("python")
        external_repo = Path(self._required_environment("external_repo")).resolve()
        adapter = CODE_DIR / "experiments" / "benchmarks" / "external" / f"{self.method_id}_adapter.py"
        metadata_path = output_dir / "external_metadata.json"
        command = [
            python_executable,
            str(adapter),
            "--input", str(input_path),
            "--output", str(prediction_path),
            "--metadata", str(metadata_path),
            "--settings", str(settings_path),
            "--external-repo", str(external_repo),
            "--seed", str(int(seed)),
        ]
        subprocess.run(
            command,
            cwd=external_repo,
            timeout=float(self.settings.get("timeout_seconds", 86400)),
            check=True,
        )
        with np.load(prediction_path) as predictions:
            prediction_ids = [str(value) for value in predictions["transition_ids"]]
            prediction_blocks = self._unpack_ragged(
                predictions["prediction_values"],
                predictions["prediction_offsets"],
                expected_count=len(prediction_ids),
                label="external prediction",
            )
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("External prediction IDs must be unique")
        self.predictions_by_transition = {
            transition_id: prediction_blocks[index]
            for index, transition_id in enumerate(prediction_ids)
        }
        tasks_by_id = {str(task["transition_id"]): task for task in tasks}
        if set(self.predictions_by_transition) != set(tasks_by_id):
            raise ValueError("External predictions do not match requested transition IDs")
        for transition_id, prediction in self.predictions_by_transition.items():
            expected_shape = np.asarray(tasks_by_id[transition_id]["source_z"]).shape
            if prediction.shape != expected_shape:
                raise ValueError(
                    f"External prediction shape for {transition_id} is {prediction.shape}; "
                    f"expected {expected_shape}"
                )

    def predict(self, source, **kwargs):
        return self.predictions_by_transition[str(kwargs["transition_id"])].copy()


__all__ = ["ExternalSnapshotMethod"]
