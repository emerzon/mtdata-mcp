from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Dict, Optional, Tuple

import numpy as np


class _SkModelMixin:
    """Mixin providing fit/transform helpers for sklearn-like models on self._model.

    Expects subclasses to set `self._model` in __init__ and may override
    supports_transform() when the underlying model cannot transform new samples.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):  # type: ignore[override]
        self._model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self._model.transform(X), dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self._model.fit_transform(X), dtype=np.float32)

@lru_cache(maxsize=None)
def _optional_dependency(module_name: str, attribute: Optional[str] = None) -> Any:
    """Import an optional backend on first use and cache its resolved object."""
    try:
        module = import_module(module_name)
        return getattr(module, attribute) if attribute else module
    except Exception:
        return None


@lru_cache(maxsize=None)
def _dependency_available(module_name: str) -> bool:
    """Probe package metadata without importing an optional runtime."""
    try:
        return find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _require_dependency(
    module_name: str,
    attribute: Optional[str],
    unavailable_message: str,
) -> Any:
    dependency = _optional_dependency(module_name, attribute)
    if dependency is None:
        raise RuntimeError(unavailable_message)
    return dependency


class DimReducer:
    """Abstract interface for dimensionality reducers.

    Implementations should support fit, transform, and fit_transform. For some
    algorithms (e.g., t-SNE), transform on new samples is not supported; in such
    cases `supports_transform` should return False.
    """

    name: str = "none"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DimReducer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def supports_transform(self) -> bool:
        return True

    def info(self) -> Dict[str, Any]:
        return {"method": self.name}


class NoneReducer(DimReducer):
    name = "none"

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


class PCAReducer(_SkModelMixin, DimReducer):
    name = "pca"

    def __init__(self, n_components: int) -> None:
        model_cls = _require_dependency(
            "sklearn.decomposition",
            "PCA",
            "scikit-learn not available; cannot use PCA",
        )
        self.n_components = int(max(1, n_components))
        self._model = model_cls(
            n_components=self.n_components,
            svd_solver="auto",
            whiten=False,
        )

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components)}


class SVDReducer(_SkModelMixin, DimReducer):
    name = "svd"

    def __init__(self, n_components: int) -> None:
        model_cls = _require_dependency(
            "sklearn.decomposition",
            "TruncatedSVD",
            "scikit-learn not available; cannot use TruncatedSVD",
        )
        self.n_components = int(max(1, n_components))
        self._model = model_cls(n_components=self.n_components)

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components)}


class SparsePCAReducer(_SkModelMixin, DimReducer):
    name = "spca"

    def __init__(self, n_components: int = 2, alpha: float = 1.0) -> None:
        model_cls = _require_dependency(
            "sklearn.decomposition",
            "SparsePCA",
            "scikit-learn not available; cannot use SparsePCA",
        )
        self.n_components = int(max(1, n_components))
        self.alpha = float(alpha)
        self._model = model_cls(n_components=self.n_components, alpha=self.alpha)

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components), "alpha": float(self.alpha)}


class KPCAReducer(_SkModelMixin, DimReducer):
    name = "kpca"

    def __init__(self, n_components: int = 2, kernel: str = "rbf", gamma: Optional[float] = None, degree: int = 3, coef0: float = 1.0) -> None:
        model_cls = _require_dependency(
            "sklearn.decomposition",
            "KernelPCA",
            "scikit-learn not available; cannot use KernelPCA",
        )
        self.n_components = int(max(1, n_components))
        self.kernel = str(kernel)
        self.gamma = None if gamma is None else float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self._model = model_cls(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0, fit_inverse_transform=False)

    def info(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_components": int(self.n_components),
            "kernel": str(self.kernel),
            "gamma": None if self.gamma is None else float(self.gamma),
            "degree": int(self.degree),
            "coef0": float(self.coef0),
        }


class LaplacianReducer(DimReducer):
    name = "laplacian"

    def __init__(self, n_components: int = 2, n_neighbors: int = 10) -> None:
        model_cls = _require_dependency(
            "sklearn.manifold",
            "SpectralEmbedding",
            "scikit-learn not available; cannot use SpectralEmbedding",
        )
        self.n_components = int(max(1, n_components))
        self.n_neighbors = int(max(1, n_neighbors))
        self._model = model_cls(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
        )

    def supports_transform(self) -> bool:
        return False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LaplacianReducer":
        self._model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("SpectralEmbedding does not support transforming new samples; use 'pca' or 'umap'")

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(self._model.fit_transform(X), dtype=np.float32)

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components), "n_neighbors": int(self.n_neighbors), "supports_transform": False}


class IsomapReducer(_SkModelMixin, DimReducer):
    name = "isomap"

    def __init__(self, n_components: int = 2, n_neighbors: int = 5) -> None:
        model_cls = _require_dependency(
            "sklearn.manifold",
            "Isomap",
            "scikit-learn not available; cannot use Isomap",
        )
        self.n_components = int(max(1, n_components))
        self.n_neighbors = int(max(1, n_neighbors))
        self._model = model_cls(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
        )

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components), "n_neighbors": int(self.n_neighbors)}


class UMAPReducer(_SkModelMixin, DimReducer):
    name = "umap"

    def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> None:
        model_cls = _require_dependency(
            "umap",
            "UMAP",
            "umap-learn not available; `pip install umap-learn`",
        )
        self.n_components = int(max(1, n_components))
        self.n_neighbors = int(max(1, n_neighbors))
        self.min_dist = float(min_dist)
        self._model = model_cls(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
        )

    def info(self) -> Dict[str, Any]:
        return {"method": self.name, "n_components": int(self.n_components), "n_neighbors": int(self.n_neighbors), "min_dist": float(self.min_dist)}


class TSNEReducer(DimReducer):
    name = "tsne"

    def __init__(self, n_components: int = 2, perplexity: float = 30.0, learning_rate: float = 200.0, n_iter: int = 1000) -> None:
        model_cls = _require_dependency(
            "sklearn.manifold",
            "TSNE",
            "scikit-learn not available; cannot use TSNE",
        )
        self.n_components = int(max(1, n_components))
        self.perplexity = float(perplexity)
        self.learning_rate = float(learning_rate)
        self.n_iter = int(max(250, n_iter))
        import inspect
        _tsne_params = inspect.signature(model_cls).parameters
        iter_kwarg = "max_iter" if "max_iter" in _tsne_params else "n_iter"
        self._model = model_cls(n_components=self.n_components, perplexity=self.perplexity, learning_rate=self.learning_rate, init="pca", **{iter_kwarg: self.n_iter})

    def supports_transform(self) -> bool:
        # sklearn TSNE does not support transforming new samples
        return False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TSNEReducer":
        # Fit returns self; TSNE computes embedding in fit_transform
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("TSNE does not support transforming new samples; use 'pca' or 'umap' instead")

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(self._model.fit_transform(X), dtype=np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_components": int(self.n_components),
            "perplexity": float(self.perplexity),
            "learning_rate": float(self.learning_rate),
            "n_iter": int(self.n_iter),
            "supports_transform": False,
        }


def create_reducer(method: Optional[str], params: Optional[Dict[str, Any]] = None) -> Tuple[DimReducer, Dict[str, Any]]:
    """Factory: create a dimensionality reducer from a method string and params.

    method: one of None/'none', 'pca', 'svd', 'umap', 'isomap', 'tsne'.
    params: dict of keyword args relevant to the method.

    Returns: (reducer_instance, effective_params)
    """
    if not method or str(method).lower() in ("none", "null", "false"):
        return NoneReducer(), {"method": "none"}
    m = str(method).lower().strip()
    p = dict(params or {})
    if m == "pca":
        n = int(p.get("n_components", p.get("components", 0) or 0))
        if n <= 0:
            raise ValueError("PCA requires a positive n_components")
        r = PCAReducer(n)
        return r, r.info()
    if m == "svd":
        n = int(p.get("n_components", p.get("components", 0) or 0))
        if n <= 0:
            raise ValueError("SVD requires a positive n_components")
        r = SVDReducer(n)
        return r, r.info()
    if m == "spca":
        n = int(p.get("n_components", 2))
        alpha = float(p.get("alpha", 1.0))
        r = SparsePCAReducer(n_components=n, alpha=alpha)
        return r, r.info()
    if m == "kpca":
        n = int(p.get("n_components", 2))
        kernel = str(p.get("kernel", "rbf"))
        gamma = p.get("gamma", None)
        gamma = None if gamma in (None, "none", "null", "") else float(gamma)
        degree = int(p.get("degree", 3))
        coef0 = float(p.get("coef0", 1.0))
        r = KPCAReducer(n_components=n, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        return r, r.info()
    if m == "isomap":
        n = int(p.get("n_components", 2))
        k = int(p.get("n_neighbors", 5))
        r = IsomapReducer(n_components=n, n_neighbors=k)
        return r, r.info()
    if m == "laplacian":
        n = int(p.get("n_components", 2))
        k = int(p.get("n_neighbors", 10))
        r = LaplacianReducer(n_components=n, n_neighbors=k)
        return r, r.info()
    if m == "umap":
        n = int(p.get("n_components", 2))
        k = int(p.get("n_neighbors", 15))
        md = float(p.get("min_dist", 0.1))
        r = UMAPReducer(n_components=n, n_neighbors=k, min_dist=md)
        return r, r.info()
    if m == "tsne":
        n = int(p.get("n_components", 2))
        perplexity = float(p.get("perplexity", 30.0))
        lr = float(p.get("learning_rate", 200.0))
        iters = int(p.get("n_iter", 1000))
        r = TSNEReducer(n_components=n, perplexity=perplexity, learning_rate=lr, n_iter=iters)
        return r, r.info()
    raise ValueError(f"Unknown dimensionality reduction method: {method}")


def list_dimred_methods() -> Dict[str, Dict[str, Any]]:
    """Return available dimension reduction methods and availability flags."""
    sklearn_available = _dependency_available("sklearn")
    return {
        "none": {"available": True, "description": "No reduction; pass-through."},
        "pca": {"available": sklearn_available, "description": "Principal Component Analysis (sklearn)."},
        "svd": {"available": sklearn_available, "description": "Truncated SVD (sklearn)."},
        "spca": {"available": sklearn_available, "description": "Sparse PCA (sklearn)."},
        "kpca": {"available": sklearn_available, "description": "Kernel PCA (sklearn)."},
        "isomap": {"available": sklearn_available, "description": "Isomap manifold learning (sklearn)."},
        "laplacian": {"available": sklearn_available, "description": "Laplacian Eigenmaps / Spectral Embedding (sklearn)."},
        "umap": {"available": _dependency_available("umap"), "description": "UMAP dimensionality reduction (umap-learn)."},
        "tsne": {"available": sklearn_available, "description": "t-SNE (sklearn); no transform for new samples."},
    }
