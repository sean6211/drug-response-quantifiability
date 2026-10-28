# drug-response-quantifiability

EvE Bio quantifiability analysis. The release files are not in this repo.

```bash
pip install -e .
python scripts/prepare_dataset.py --evebio /path/to/evebio_release
python scripts/summarize_outcomes.py --detailed /path/to/evebio_release/detailed_result_table.csv
python scripts/run_random_cv.py
python scripts/run_feature_ablation.py
python scripts/run_scaffold_split.py
python scripts/run_mechanism_holdout.py
python scripts/run_rolling.py
python scripts/run_group_importance.py
python scripts/run_subgroup_holdout.py
python scripts/run_calibration.py
python scripts/run_external_amplitude.py --table matched.csv
```

`prepare_dataset.py` writes `data/prepared/`. Later scripts read that directory. `summarize_outcomes.py` reads `detailed_result_table.csv` directly.

A pair is quantified when `Final_Category` is `Active - Quantified (11pt)`. Random-split profiles use the full screening matrix. `run_rolling.py` recomputes the nine profile columns from earlier releases only.

Full model: random forest, 500 trees, `class_weight='balanced'`, `random_state=42`. Ablation, scaffold split, mechanism holdout, and the two-stage score use 300 trees. Logistic regression is standardized inside each training fold.

`run_external_amplitude.py` expects columns `endpoint`, `screen_max`, and `quantified`.
