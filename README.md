# Kaggle Topper Agent

Autonomous, rule-aware Kaggle competition agent focused first on tabular competitions.

## What it does

- Scouts current Kaggle competitions and prioritizes low-participation candidates.
- Reads the Rules and Evaluation pages before training.
- Downloads and extracts competition data.
- Detects ordinary train/test/sample-submission layouts.
- Infers classification vs regression.
- Runs HistGradientBoosting, ExtraTrees, CatBoost, LightGBM and XGBoost.
- Uses cross-validation and builds an ensemble.
- Persists experiment state between scheduled runs.
- Research-only loop keeps improving models and saves the best candidate CSV without submitting.
- Writes out-of-fold confusion-matrix and classification reports for classification models.
- Can submit to Kaggle only through a separately enabled, reviewed submission workflow.
- Includes GitHub Actions for daily scouting and 12-hour competition cycles.

## Required GitHub secret

Create this repository secret:

`KAGGLE_API_TOKEN`

Do not commit the token into the repository.

## Repository variables

After we choose a competition, add:

- `KAGGLE_COMPETITION` — competition slug
- `KAGGLE_ALLOW_AUTO_SUBMIT` — keep `false` until rules are reviewed
- `KAGGLE_APPROVED_COMPETITIONS` — approved competition slugs, comma-separated

## Useful commands

```powershell
python -m pip install -e ".[boost]"
kaggle-agent doctor
kaggle-agent scout --limit 25
kaggle-agent prepare <competition-slug>
kaggle-agent train <competition-slug>
kaggle-agent auto <competition-slug>
kaggle-agent research <competition-slug> --target-score 0.90 --max-hours 5
kaggle-agent monitor <competition-slug>
```

## Automation

`.github/workflows/scout.yml` runs daily and refreshes the competition shortlist.

`.github/workflows/ev-purchases.yml` runs every six hours on GitHub-hosted runners. Each run has a five-hour research budget and continues experimenting until the configured validation target is reached or the run budget expires. It saves the best candidate to `reports/<competition>/best_submission.csv` and commits the best-model metadata and validation report so later runs can resume.

`.github/workflows/compete.yml` is manual-only and also runs research mode; automatic submissions are disabled in both workflows.

## Current scope

The generic engine targets normal tabular CSV competitions. Image, NLP, code competitions, unusual metrics, grouped/time-aware validation, and multi-target competitions need competition-specific adapters.
\n## Validation vs leaderboard score\n\nThe confusion matrix is calculated from out-of-fold predictions on labeled training data at a 0.5 threshold for binary classification (argmax for multiclass). It is a diagnostic, not the Kaggle leaderboard score. The `0.90` research target refers to cross-validation ROC-AUC for the EV binary-classification setup. Test/sample-submission CSV files do not contain ground-truth test labels, so they cannot produce a genuine confusion matrix or verify public leaderboard performance without submitting or obtaining labeled outcomes. Research workflows do not submit to Kaggle.\n