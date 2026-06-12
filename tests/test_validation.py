from __future__ import annotations

import pytest

from experiment_console.validation import ConfigValidationError, validate_experiment_config


def test_formal_sweep_requires_grid(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("method: random\nparameters: {}\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(path, "sweep")


def test_expected_run_count(tmp_path):
    path = tmp_path / "ok.yaml"
    path.write_text(
        "method: grid\nprogram: train.py\nparameters:\n  dataset:\n    values: [Cora]\n  seed:\n    values: [0, 1, 2, 3, 4]\n  lr:\n    values: [0.1, 0.01]\n",
        encoding="utf-8",
    )
    result = validate_experiment_config(path, "sweep")
    assert result["expected_run_count"] == 10
