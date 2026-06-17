import pickle
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

# (audio_method, visual_method) -> n_components to reduce visual to.
# None = no PCA needed (dims already balanced).
_PCA_TABLE: dict[tuple[str, str], int | None] = {
    ("handcrafted", "landmarks"): 536,
    ("handcrafted", "xception"):  536,
    ("wav2vec2",    "landmarks"): None,
    ("wav2vec2",    "xception"):  768,
    ("hubert",      "landmarks"): None,
    ("hubert",      "xception"):  768,
}


def get_n_components(audio_method: str, visual_method: str) -> int | None:
    """Return the PCA target dim for the given feature combination, or None."""
    key = (audio_method.lower(), visual_method.lower())
    if key not in _PCA_TABLE:
        raise ValueError(
            f"Unknown audio/visual combination: {key}. "
            f"Valid options: {list(_PCA_TABLE.keys())}"
        )
    return _PCA_TABLE[key]


class VisualPCAReducer:
    """
    Reduces visual feature vectors via PCA.

    Fit ONLY on train features to avoid data leakage.
    Persist with save()/load() so val/test use the same transform.

    Usage:
        reducer = VisualPCAReducer(n_components=768)
        train_visual = reducer.fit_transform(train_visual_array)  # fit on train
        val_visual   = reducer.transform(val_visual_array)        # apply to val/test
        reducer.save(run_dir / "pca.pkl")
    """

    def __init__(self, n_components: int | None):
        self.n_components = n_components
        self._pca: PCA | None = None

    @property
    def needed(self) -> bool:
        return self.n_components is not None

    def fit(self, X: np.ndarray) -> "VisualPCAReducer":
        if self.needed:
            n = min(self.n_components, X.shape[0], X.shape[1])
            self._pca = PCA(n_components=n, random_state=42)
            self._pca.fit(X)
        return self

    @property
    def output_dim(self) -> int | None:
        """Actual output dimensionality after fit (may be < n_components if data-constrained)."""
        if not self.needed:
            return None
        if self._pca is not None:
            return int(self._pca.n_components_)
        return self.n_components

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.needed:
            return X
        if self._pca is None:
            raise RuntimeError("Call fit() before transform().")
        return self._pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "VisualPCAReducer":
        with open(path, "rb") as f:
            return pickle.load(f)
