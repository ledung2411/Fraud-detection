# Data

Nothing in this folder is committed (see `.gitignore`).

## IEEE-CIS Fraud Detection (real data)

Kaggle competition data from Vesta Corporation; you must accept the competition rules
on <https://www.kaggle.com/c/ieee-fraud-detection/rules> before downloading.

```bash
pip install kaggle            # needs ~/.kaggle/kaggle.json (Kaggle > Account > API token)
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-fraud-detection
unzip data/raw/ieee-fraud-detection/ieee-fraud-detection.zip -d data/raw/ieee-fraud-detection
make train-ieee
```

Only `train_transaction.csv` and `train_identity.csv` are used: the Kaggle test files
have no labels. The first run caches the merged frame to `data/processed/ieee.parquet`.

Peak memory is about 7 GB: a full run on a synthetic stand-in of the same size
(584k rows x 425 columns) peaked at 6.7 GB RSS. V/C/D columns are read as float32.

## Synthetic data

`make train` generates `data/synthetic/` on first use with
`src/fraud_detection/synthetic.py`: same file names, columns and conventions as the
Kaggle files, so the identical code path runs without a download (also in CI).
