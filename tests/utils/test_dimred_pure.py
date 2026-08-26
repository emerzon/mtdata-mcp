"""Tests for utils/dimred.py — DimReducer classes and create_reducer factory."""
import numpy as np
import pytest

from mtdata.utils.dimred import (
    IsomapReducer,
    KPCAReducer,
    LaplacianReducer,
    NoneReducer,
    PCAReducer,
    SparsePCAReducer,
    SVDReducer,
    TSNEReducer,
    create_reducer,
    list_dimred_methods,
)


def _synthetic_data(n=50, d=5, seed=42):
    return np.random.RandomState(seed).randn(n, d).astype(np.float32)


class TestNoneReducer:
    def test_passthrough(self):
        X = _synthetic_data()
        r = NoneReducer()
        out = r.fit_transform(X)
        np.testing.assert_allclose(out, X, atol=1e-6)
        assert r.supports_transform()
        assert r.info() == {"method": "none"}


@pytest.mark.parametrize(
    ("factory", "kwargs", "method"),
    [
        (PCAReducer, {"n_components": 2}, "pca"),
        (SVDReducer, {"n_components": 2}, "svd"),
        (SparsePCAReducer, {"n_components": 2, "alpha": 0.5}, "spca"),
        (KPCAReducer, {"n_components": 2, "kernel": "rbf"}, "kpca"),
        (LaplacianReducer, {"n_components": 2, "n_neighbors": 5}, "laplacian"),
        (IsomapReducer, {"n_components": 2, "n_neighbors": 5}, "isomap"),
        (TSNEReducer, {"n_components": 2, "perplexity": 10.0, "n_iter": 300}, "tsne"),
    ],
)
def test_reducer_fit_transform_shape_and_method(factory, kwargs, method):
    X = _synthetic_data()
    r = factory(**kwargs)
    out = r.fit_transform(X)
    assert out.shape == (50, 2)
    assert r.info()["method"] == method


class TestTransformSupport:
    def test_pca_transform_after_fit(self):
        X = _synthetic_data()
        r = PCAReducer(n_components=2)
        r.fit(X)
        assert r.transform(X).shape == (50, 2)

    @pytest.mark.parametrize(
        "factory_kwargs",
        [
            (LaplacianReducer, {"n_components": 2, "n_neighbors": 5}),
            (TSNEReducer, {"n_components": 2, "perplexity": 10.0, "n_iter": 300}),
        ],
    )
    def test_no_transform_reducers_raise(self, factory_kwargs):
        factory, kwargs = factory_kwargs
        r = factory(**kwargs)
        assert not r.supports_transform()
        if factory is LaplacianReducer:
            r.fit(_synthetic_data())
        with pytest.raises(RuntimeError):
            r.transform(_synthetic_data(10, 5) if factory is LaplacianReducer else _synthetic_data())


class TestCreateReducer:
    @pytest.mark.parametrize(
        ("method", "params", "expected_type"),
        [
            (None, None, NoneReducer),
            ("none", None, NoneReducer),
            ("pca", {"n_components": 3}, PCAReducer),
            ("svd", {"n_components": 2}, SVDReducer),
            ("spca", {"n_components": 2, "alpha": 0.5}, SparsePCAReducer),
            ("kpca", {"n_components": 2, "kernel": "poly", "gamma": 0.1, "degree": 2}, KPCAReducer),
            ("isomap", {"n_components": 2, "n_neighbors": 7}, IsomapReducer),
            ("laplacian", {"n_components": 2, "n_neighbors": 8}, LaplacianReducer),
            ("tsne", {"n_components": 2, "perplexity": 10, "n_iter": 300}, TSNEReducer),
        ],
    )
    def test_factory_builds_known_methods(self, method, params, expected_type):
        r, p = create_reducer(method, params or {})
        assert isinstance(r, expected_type)
        if method == "pca":
            assert p["n_components"] == 3
        if method == "kpca":
            assert p["kernel"] == "poly"

    @pytest.mark.parametrize("method", ["pca", "svd"])
    def test_missing_components_raises(self, method):
        with pytest.raises(ValueError):
            create_reducer(method, {})

    def test_kpca_gamma_none(self):
        _, p = create_reducer("kpca", {"n_components": 2, "gamma": "none"})
        assert p["gamma"] is None

    @pytest.mark.parametrize(
        ("method", "exc", "match"),
        [
            ("lda", ValueError, "Unknown"),
            ("invalid_method", ValueError, "Unknown"),
            ("deep_diffusion_maps", ValueError, "Unknown"),
            ("dreams", ValueError, "Unknown"),
            ("pcc", ValueError, "Unknown"),
        ],
    )
    def test_unsupported_methods_raise(self, method, exc, match):
        with pytest.raises(exc, match=match):
            create_reducer(method)


class TestListDimredMethods:
    def test_catalog_shape(self):
        methods = list_dimred_methods()
        assert isinstance(methods, dict)
        assert "pca" in methods
        assert methods["none"]["available"] is True
        for info in methods.values():
            assert "description" in info
            assert "available" in info
            assert isinstance(info.get("params"), list)

    def test_catalog_matches_factory_contract(self):
        methods = list_dimred_methods()
        pca_names = {item["name"] for item in methods["pca"]["params"]}
        assert pca_names == {"n_components"}
        assert methods["pca"]["params"][0].get("required") is True
        assert "default" not in methods["pca"]["params"][0]

        spca_params = {item["name"]: item for item in methods["spca"]["params"]}
        assert spca_params["n_components"]["default"] == 2
        assert "alpha" in spca_params

        isomap_params = {item["name"]: item for item in methods["isomap"]["params"]}
        assert isomap_params["n_neighbors"]["default"] == 5

        kpca_names = {item["name"] for item in methods["kpca"]["params"]}
        assert {"degree", "coef0"} <= kpca_names
