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
- Can submit to Kaggle only after the competition is explicitly approved.
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
kaggle-agent monitor <competition-slug>
```

## Automation

`.github/workflows/scout.yml` runs daily and refreshes the competition shortlist.

`.github/workflows/compete.yml` runs every 12 hours after a competition is selected. It tests the next model variant, compares cross-validation with the previous best, and submits only when the validation gate passes and auto-submit has explicitly been enabled.

## Current scope

The generic engine targets normal tabular CSV competitions. Image, NLP, code competitions, unusual metrics, grouped/time-aware validation, and multi-target competitions need competition-specific adapters.
