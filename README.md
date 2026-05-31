# ETAN-SCA

ETAN-SCA is a side-channel analysis training project for evaluating
whether neural models can classify leakage patterns in Reference-PPM traces.

#Currently working on expnading the ETAN-SCA to a full scale attack model that is capable of pirtial/full key recovery

## Current Scope

The current loader labels each trace as the Hamming weight of a selected nonce
byte. That makes the project a leakage-classification benchmark, not a full
key-recovery pipeline.

The training script records this explicitly:

- `target_semantics`: currently `nonce_byte_hamming_weight`
- `target_is_secret_dependent`: currently `false`
- `leakage_proof_report.json`: states whether the model cleared the leakage
  classifier gate and whether partial-key extraction is ready

Partial-key extraction should only be claimed after the target is replaced with
a secret-dependent intermediate and key-candidate ranking metrics are added.

## Run

```bash
source .venv/bin/activate
python -m src.train_all_models
tensorboard --logdir runs/ETAN_SCA --host 127.0.0.1 --port 6006
```

Outputs are written to `results/run_<timestamp>/` and TensorBoard logs to
`runs/ETAN_SCA/<timestamp>/`.

## A100 Profile

`src/train_all_models.py` is currently configured for a full A100 run:

- all matched Reference-PPM set A files (`MAX_FILES = None`)
- full CNN/RNN/LSTM/ensemble suite (`LEARNABILITY_MODE = False`)
- batch size 256, 50 max epochs, patience 12
- CUDA TF32 enabled and bfloat16 autocast enabled
- 50,000-sample window downsampled by 5 (`T = 10000`)
- wider A100 model profile: CNN base 64, CNN `d_model` 128, 8 attention
  heads, RNN/LSTM `d_model` 64

Useful outputs include:

- `model_comparison.csv`
- `leakage_proof_report.json`
- `<model>_metrics.csv`
- `<model>_best.pt`
- `<model>_confusion_matrix.png`
- `<model>_confusion_matrix.npy`

## Notes

Set `LEARNABILITY_MODE = True` in `src/train_all_models.py` only when you want a
short CNN-only proof run on middle Hamming-weight classes.
