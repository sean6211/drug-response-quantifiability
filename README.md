# drug-response-quantifiability

EvE Bio quantifiability analysis. The release files are not in this repo.

```bash
pip install -e .
python scripts/prepare_dataset.py --evebio /path/to/evebio_release
python scripts/summarize_outcomes.py --detailed /path/to/evebio_release/detailed_result_table.csv
python scripts/run_random_cv.py
python scripts/run_bootstrap.py
python scripts/run_calibration.py
python scripts/run_feature_ablation.py
python scripts/run_scaffold_split.py
python scripts/run_mechanism_holdout.py
python scripts/run_rolling.py
python scripts/run_group_importance.py
python scripts/run_subgroup_holdout.py
python scripts/run_descriptor_tests.py
python scripts/run_external_amplitude.py --table matched.csv
```

`prepare_dataset.py` writes `data/prepared/`. Later scripts read that directory. `summarize_outcomes.py` reads `detailed_result_table.csv` directly. `run_random_cv.py` also writes `outputs/random_cv_oof.csv.gz`, which `run_bootstrap.py` and `run_calibration.py` read.

A pair is quantified when `Final_Category` is `Active - Quantified (11pt)`. Random-split profiles use the full screening matrix. `run_rolling.py` recomputes the nine profile columns from earlier releases only and ranks each release with the full forest against the reported screening maximum.

Full model: random forest, 500 trees, `class_weight='balanced'`, `random_state=42`. `run_random_cv.py` also fits it without the screening-category and promotion-reason indicators. Ablation, scaffold split, mechanism holdout, and the two-stage score use 300 trees. Isolated and nested ablations share the same five folds. Logistic regression is standardized inside each training fold.

Bootstrap intervals resample compounds with replacement (2,000 resamples). `run_descriptor_tests.py` reports Welch tests and Cohen's d for six RDKit descriptors across pairs and across compounds.

`run_external_amplitude.py` expects columns `endpoint`, `screen_max`, and `quantified`.
