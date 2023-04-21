The following is classification over movie scripts


To run code, download each data from its kaggle link (see data/raw/*/README.md) and paste under their specific path data/raw/*/?

## Jupyer notebook for tests

the final model scratch is under `jupyter_for_experiments/freezed_bert.ipynb`.

## Data Preprocessing

You have to use 
```yaml
dvc repro starwars
```
## Installation

Check `scripts/init.bash` for more details.

## Training

Train moviescripts Bert model on the star wars dataset
```yaml
poetry run train
```
## Testing
```yaml
poetry run test
```

