# adapter-transfer
Experiments about adapter transfer.

## Environment

Use miniconda:
```bash
conda create --name {env_name} -f environment.yml
```

## Run Experiment

1. Modify `config.yaml`
2. `python run_transfer.py`

### Config.yaml

* `trainer`: Configuration of `AdapterTrainer`
  * `ckpt_path`: Path to store checkpoints.
  * logging_dir: Path to store trainer's output log.
* `model`: Model configuration.
* `tasks`: list of tasks you want to run transfer.
