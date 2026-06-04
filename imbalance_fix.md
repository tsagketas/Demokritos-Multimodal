# Class Imbalance Fix — Instructions

## Context

The FakeAVCeleb v1.2 dataset has a severe class imbalance:

| Split | Real (label=0) | Fake (label=1) | Ratio |
|-------|---------------|----------------|-------|
| train | 350           | 14 746         | 42:1  |
| val   | 75            | 3 160          | 42:1  |
| test  | 75            | 3 160          | 42:1  |

- label=0 → `RealVideo-RealAudio` only
- label=1 → `RealVideo-FakeAudio` + `FakeVideo-RealAudio` + `FakeVideo-FakeAudio`

A model that predicts "fake" for everything achieves 97.7% accuracy.
The current training pipeline has no imbalance handling at all.

## Decision

Use **`WeightedRandomSampler` with target ratio 1:5 (real:fake)** on the train DataLoader only.

- Do NOT change `pos_weight` in `BCEWithLogitsLoss` — in this dataset fake=label=1 is the
  majority, so `pos_weight > 1` would make things worse, and choosing the right `pos_weight < 1`
  adds fragility. The sampler alone is sufficient.
- Do NOT touch val/test loaders — evaluation must reflect the real distribution.
- target_ratio=5 is a deliberate choice: goes from 1:42 → 1:5, reduces imbalance without
  forcing 50-50 which would cause overfitting on the 350 real samples.

## Files to change

### 1. `src/data/feature_dataset.py`

Add a `labels` property to `FeatureDataset` that returns all labels as a numpy array.
This is needed by the training script to compute per-sample weights.

Add after `__len__`:

```python
@property
def labels(self) -> np.ndarray:
    return np.array([s["label"] for s in self.samples])
```

Also add `import numpy as np` at the top if not already present (it already is).

### 2. `src/train.py`

#### 2a. New helper function `make_weighted_train_loader`

Add this function near the other helpers at the top of the file (after `collate_fn`):

```python
def make_weighted_train_loader(
    dataset: FeatureDataset,
    batch_size: int,
    target_ratio: int = 5,
) -> DataLoader:
    """
    Returns a train DataLoader that uses WeightedRandomSampler so that
    each batch contains approximately 1 real sample per `target_ratio` fake samples.

    target_ratio=5 means batches are ~1:5 (real:fake) instead of the natural 1:42.

    Only use for the TRAIN split. Val/test loaders must stay unweighted.
    """
    labels  = dataset.labels                      # np.ndarray of 0s and 1s
    n_real  = int((labels == 0).sum())
    n_fake  = int((labels == 1).sum())

    # Each real sample gets weight  n_fake / (n_real * target_ratio)
    # Each fake sample gets weight  1.0
    # This produces P(real in batch) = 1 / (1 + target_ratio)
    w_real = n_fake / (n_real * target_ratio)
    w_fake = 1.0

    sample_weights = np.where(labels == 0, w_real, w_fake).astype(np.float64)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(dataset),
        replacement = True,
    )

    print(
        f"[sampler] WeightedRandomSampler  "
        f"n_real={n_real}  n_fake={n_fake}  "
        f"target_ratio=1:{target_ratio}  "
        f"w_real={w_real:.2f}  w_fake={w_fake:.2f}"
    )

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,       # sampler and shuffle are mutually exclusive
        collate_fn  = collate_fn,
        num_workers = 0,
    )
```

Note: when a `sampler` is passed to `DataLoader`, do NOT also pass `shuffle=True` — they are
mutually exclusive. The sampler handles the shuffling implicitly.

#### 2b. Replace train_loader creation in `main()`

Find the block in `main()` that creates the three DataLoaders:

```python
bs = train_cfg["batch_size"]
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)
```

Replace ONLY the `train_loader` line:

```python
bs = train_cfg["batch_size"]
train_loader = make_weighted_train_loader(train_ds, batch_size=bs, target_ratio=5)
val_loader   = DataLoader(val_ds,  batch_size=bs, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)
```

Val and test loaders are unchanged.

## What does NOT change

- `BCEWithLogitsLoss` stays with default `pos_weight=None`
- All model architectures (`EarlyFusionMLP`, `UnimodalClassifier`, `LateFusionCombiner`)
- All evaluation logic and metrics
- Val/test DataLoaders
- Config files
- `test_train.py` smoke tests — synthetic data is balanced so `w_real ≈ w_fake ≈ 1.0`,
  WeightedRandomSampler will work but produce nearly uniform weights (correct behaviour)

## Verification after the change

Run the smoke tests to confirm nothing broke:

```bash
python test_train.py
```

Expected output: all four modes (audio_only, video_only, early_fusion, late_fusion) show ✅.

To visually verify the sampler is working during a real training run, check the print line:

```
[sampler] WeightedRandomSampler  n_real=350  n_fake=14746  target_ratio=1:5  w_real=8.43  w_fake=1.00
```

## Optional future improvement

If results still show the model collapsing to "predict fake always" (val AUC ≈ 0.5, EER ≈ 0.5),
consider reducing `target_ratio` from 5 to 3 (closer to 50-50) as a next step.
Do NOT set `target_ratio=1` (50-50) as this risks overfitting on the 350 real samples.
