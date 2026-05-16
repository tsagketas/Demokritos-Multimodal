import yaml
import os
import shutil
from pathlib import Path
from datetime import datetime


def load_config(experiment_yaml: str) -> dict:
    base = Path(experiment_yaml).parent

    with open(experiment_yaml) as f:
        exp = yaml.safe_load(f)

    cfg = {}
    for key, filename in exp.get("configs", {}).items():
        with open(base / filename) as f:
            cfg[key] = yaml.safe_load(f)

    cfg["experiment"] = exp["experiment"]

    for dotted_key, value in exp.get("overrides", {}).items():
        _set_nested(cfg, dotted_key.split("."), value)

    return cfg


def _set_nested(d: dict, keys: list, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def setup_run_dir(cfg: dict) -> Path:
    name = cfg["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("/workspace/outputs/experiments") / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # snapshot των configs που χρησιμοποιήθηκαν
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return run_dir
