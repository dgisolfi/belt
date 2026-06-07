# belt

A composable machine learning toolbelt

## Pipelines

ML Tasks are broken into pipelines allowing models to be used with multiple pipelines and datasets.

| Pipeline | Model | Dataset |
|---|---|---|
| `classifier` | SoftmaxRegressor (NumPy) | Iris |
| `translation` | Seq2Seq RNN/LSTM | multi30k, opus_books |

## Usage

```bash
# Softmax classifier
python -m belt classifier
# Seq2Seq text translation
python -m belt translation
```

Configs live in `configs/`. Each pipeline has a YAML controlling data, model, training, and output settings.

## Extending

**New model**: register with the decorator, then set `model_name` in your config:

```python
from belt.supervised.models import supervised_model_registry

@supervised_model_registry.register("NewModel")
class NewModel(nn.Module):
    pass
```

**New dataset**: add a loader to `belt/data/` and register it in `belt/registry.py`.

## Data cache

Downloaded datasets and tokenizers are stored under `data/` at the repo root and reused on subsequent runs. Pass `--no-cache` to force a fresh download.

## Install

```bash
python -m pip install -e .

# DirectML backend (Windows GPU):
python -m pip install -e .[directml]
```
