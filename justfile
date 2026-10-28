set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

prepare:
    python scripts/prepare_dataset.py

outcomes:
    python scripts/summarize_outcomes.py

cv:
    python scripts/run_random_cv.py

ablation:
    python scripts/run_feature_ablation.py

scaffold:
    python scripts/run_scaffold_split.py

mechanism:
    python scripts/run_mechanism_holdout.py

rolling:
    python scripts/run_rolling.py

importance:
    python scripts/run_group_importance.py

subgroup:
    python scripts/run_subgroup_holdout.py

calibration:
    python scripts/run_calibration.py

paper: outcomes cv ablation scaffold mechanism rolling importance subgroup
