# Quantifiability

Code for predicting whether a compound–assay pair promoted from a three-point screen will yield a reportable dose–response potency (pXC<sub>50</sub>).

The comparisons in the main text are here: the quantifiability label, the 16 three-point features, the rest of the 1,074-column screening-stage matrix, the ranking metrics, and the evaluations those tables use. Figure rendering and the side analyses stay in the laboratory notebook.

## What this release contains

| Script | What it reproduces |
| --- | --- |
| `scripts/summarize_outcomes.py` | Promoted-pair counts, screen and profile amplitude, mechanism-family failure among active pairs |
| `scripts/run_random_cv.py` | Five-fold table: screening heuristics, logistic regression, fingerprint forest, three-point forest, full forest, and the two-stage ranking |
| `scripts/run_feature_ablation.py` | Isolated blocks (five-fold) and nested addition of fingerprints, profiles, and metadata (three-fold) |
| `scripts/run_scaffold_split.py` | Random versus Bemis–Murcko scaffold-disjoint forests |
| `scripts/run_mechanism_holdout.py` | Leave-one-mechanism-family-out transfer |
| `scripts/run_rolling.py` | Leakage-safe rolling policy. Direct-forest AUROC, and recall and 90% retention from the two-stage ranking |
| `scripts/run_group_importance.py` | Summed Gini importance by feature group |
| `scripts/run_subgroup_holdout.py` | Held-out target-class and screening-category table, and quantifiable pairs missed at a 50% budget |
| `scripts/run_calibration.py` | Expected calibration error and Brier score of the two-stage out-of-fold scores |
| `scripts/run_external_amplitude.py` | Screening-maximum ranking on an already matched external table |
| `examples/score_screen.py` | Turn a new three-point screen into the 16 features and rank it |
| `examples/walkthrough.py` | The same matrix and a small forest on synthetic pairs |

## What stays out

These were run while writing the paper and are omitted here, in the same way a methods paper ships the model and the main comparisons rather than every notebook cell:

- Plots
- Plate-control features and physicochemical property tests
- The concentration-design simulation, substructure enrichment, and the hierarchical variance model
- ChEMBL re-runs
- The invitroDB build. The external result that the paper can support with one concentration is the amplitude ranking, and that script reads a matched table
- EvE Bio and ToxCast source files. EvE Bio releases are CC BY-NC-SA; this repository does not redistribute them

Steps that only call an existing library are called, not reimplemented. Random forests, logistic regression, and Morgan fingerprints come from scikit-learn and RDKit.

## Layout

```
src/quantifiability/     feature definitions, metrics, models, cohort matrix
scripts/                 one script per main-text comparison
examples/                a new screen, and a synthetic walkthrough
data/prepared/           local matrix written by prepare_dataset.py
outputs/                 tables written by the scripts
```

## Setup

```bash
pip install -e .
python examples/walkthrough.py
python examples/score_screen.py examples/example_screen.csv
python -m unittest discover -s tests
```

`just` is optional. `just --list` shows the same commands.

## Paper comparisons

Place EvE Bio data releases 1–11 in a directory with `detailed_result_table.csv`, `summary_result_table.csv`, `observed_points_table.csv`, `compound_table.csv`, and `assay_table.csv`. From this directory:

```bash
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
```

`prepare_dataset.py` reads the three-point measurements once and writes `data/prepared/`. The later scripts read that matrix. The default `--evebio` path is `../evebio_datasets` relative to this directory.

For the external amplitude check, build a matched table elsewhere and pass it:

```bash
python scripts/run_external_amplitude.py --table matched_toxcast.csv
```

Required columns are `endpoint`, `screen_max`, and `quantified`. `examples/external_example.csv` is only a format example.

## Model protocol

| Analysis | Estimator |
| --- | --- |
| Principal forest | 500 trees, class-balanced, full training fold |
| Logistic regression | Class-balanced, standardized inside the training fold |
| Isolated blocks, ablation, scaffold split, mechanism holdout | 300 trees, class-balanced, full training fold |
| Rolling discrimination | 500-tree forest on the quantifiability label |
| Recall and budget along a release | Two-stage forest, 300 trees per stage. Score is P(active) times P(quantifiable given active) |
| Subgroup table | One gradient-boosted tree on the first stratified fold |
| Random cross-validation and scaffold random split | Five-fold, stratified, `random_state=42` |
| Nested ablation | Three-fold, stratified, `random_state=42` |
| Fold standard deviation | NumPy `std` (ddof=0), which is how the tables were summarized |

Profiles in a random split are computed from the whole screening matrix. In the rolling evaluation those nine columns are replaced using only earlier releases. Compounds or assays with no earlier history get the historical mean.

The quantifiability label is 1 when `Final_Category` is `Active - Quantified (11pt)`. Active pairs are those whose final category starts with `Active`. The maximum-activity baseline is EvE Bio's reported screening maximum (`Scr_Observed_Max`), which is the maximum of the per-concentration means.

## Code availability

Analysis code for the quantifiability label, the three-point screening features, and the main-text comparisons is available at [github.com/sean6211/drug-response-quantifiability](https://github.com/sean6211/drug-response-quantifiability). EvE Bio and invitroDB files are obtained from their public releases and are not included.
