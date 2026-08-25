"""Extended coverage tests for utils/dimred.py targeting uncovered lines."""
from unittest.mock import MagicMock, patch

import pytest

from mtdata.utils.dimred import (
    DimReducer,
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

# ===== UMAPReducer (lines 269-277) =========================================

class TestUMAPReducerInit:
    def test_missing_umap_raises(self):
        with patch("mtdata.utils.dimred._optional_dependency", return_value=None):
            from mtdata.utils.dimred import UMAPReducer
            with pytest.raises(RuntimeError, match="umap-learn"):
                UMAPReducer(n_components=2)

    def test_info(self):
        mock_umap = MagicMock()
        with patch("mtdata.utils.dimred._optional_dependency", return_value=mock_umap):
            from mtdata.utils.dimred import UMAPReducer
            r = UMAPReducer(n_components=3, n_neighbors=10, min_dist=0.2)
            info = r.info()
            assert info["method"] == "umap"
            assert info["n_components"] == 3
            assert info["n_neighbors"] == 10
            assert info["min_dist"] == 0.2


# ===== TSNEReducer missing-dependency path (happy path covered in pure) =====

class TestTSNEReducerExtended:
    def test_info_fields(self):
        """Lines 306-314."""
        mock_tsne_cls = MagicMock()
        with patch("mtdata.utils.dimred._optional_dependency", return_value=mock_tsne_cls):
            from mtdata.utils.dimred import TSNEReducer
            r = TSNEReducer(n_components=2, perplexity=15.0, learning_rate=100.0, n_iter=500)
            info = r.info()
            assert info["method"] == "tsne"
            assert info["perplexity"] == 15.0
            assert info["learning_rate"] == 100.0
            assert info["n_iter"] == 500
            assert info["supports_transform"] is False

    def test_missing_sklearn_raises(self):
        """Line 285."""
        with patch("mtdata.utils.dimred._optional_dependency", return_value=None):
            from mtdata.utils.dimred import TSNEReducer
            with pytest.raises(RuntimeError, match="scikit-learn"):
                TSNEReducer(n_components=2)


# ===== create_reducer factory (lines 480-553) ==============================

class TestCreateReducerExtended:
    def test_umap(self):
        """Lines 480-484."""
        mock_umap = MagicMock()
        with patch("mtdata.utils.dimred._optional_dependency", return_value=mock_umap):
            r, p = create_reducer("umap", {"n_components": 3, "n_neighbors": 10, "min_dist": 0.05})
            assert p["method"] == "umap"
            assert p["n_components"] == 3

    def test_tsne_factory(self):
        """Lines 542-548."""
        mock_tsne = MagicMock()
        with patch("mtdata.utils.dimred._optional_dependency", return_value=mock_tsne):
            r, p = create_reducer("tsne", {"n_components": 3, "perplexity": 20.0, "learning_rate": 150.0, "n_iter": 800})
            assert p["method"] == "tsne"
            assert p["n_components"] == 3

    def test_null_false_string_returns_none(self):
        """Line 439: 'null' and 'false' strings → NoneReducer."""
        for s in ("null", "false", "False", "NULL"):
            r, p = create_reducer(s)
            assert isinstance(r, NoneReducer)


# ===== list_dimred_methods (lines 565-585) =================================

class TestListDimredMethodsExtended:
    def test_all_methods_present(self):
        methods = list_dimred_methods()
        expected = {
            "none",
            "pca",
            "svd",
            "spca",
            "kpca",
            "isomap",
            "laplacian",
            "umap",
            "tsne",
        }
        assert set(methods.keys()) == expected

    def test_catalog_excludes_research_stubs(self):
        methods = list_dimred_methods()
        assert "lda" not in methods
        assert "diffusion" not in methods
        assert "dreams_cne" not in methods
        assert "deep_diffusion_maps" not in methods
        assert "dreams" not in methods
        assert "pcc" not in methods
