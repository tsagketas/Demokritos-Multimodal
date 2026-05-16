import sys
sys.path.insert(0, "/workspace/src")

from utils.config import load_config, setup_run_dir
from utils.results import save_metrics

cfg = load_config("/workspace/configs/experiment.yaml")
run_dir = setup_run_dir(cfg)

print(f"[run] experiment : {cfg['experiment']['name']}")
print(f"[run] run_dir    : {run_dir}")
print(f"[run] lr         : {cfg['train']['training']['learning_rate']}")
print(f"[run] epochs     : {cfg['train']['training']['epochs']}")
print(f"[run] audio model: wav2vec2={cfg['features']['audio']['wav2vec2']['enabled']} | hubert={cfg['features']['audio']['hubert']['enabled']}")
print(f"[run] fusion     : late={cfg['fusion']['fusion']['late']['enabled']}, strategy={cfg['fusion']['fusion']['late']['strategy']}")

dummy_metrics = {
    "accuracy": 0.9230,
    "auc":      0.9610,
    "eer":      0.0780,
    "f1":       0.9180,
    "precision":0.9250,
    "recall":   0.9110,
    "per_category": {
        "RealVideo-FakeAudio": {"accuracy": 0.9400, "auc": 0.9700},
        "FakeVideo-RealAudio": {"accuracy": 0.8900, "auc": 0.9300},
        "FakeVideo-FakeAudio": {"accuracy": 0.8600, "auc": 0.9100},
    }
}

save_metrics(dummy_metrics, run_dir, tag="eval")
print("[run] done.")
