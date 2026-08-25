"""Tests for src/mtdata/forecast/forecast_validation.py"""

import mtdata.forecast.forecast_validation as fv


class TestCanonicalizeForecastMethods:
    def test_case_insensitive_canonical_names_and_duplicate_rejection(self):
        canonical, error = fv.canonicalize_forecast_methods(
            ["NAIVE", "Drift"],
            valid_methods=["naive", "drift", "theta"],
        )
        assert error is None
        assert canonical == ["naive", "drift"]

        _canonical, duplicate = fv.canonicalize_forecast_methods(
            ["naive", "NAIVE"],
            valid_methods=["naive", "drift"],
        )
        assert duplicate is not None
        assert duplicate["error_code"] == "duplicate_method"
        assert "naive" in duplicate["error"]

    def test_unknown_methods_can_be_lowercased_without_registry(self):
        canonical, error = fv.canonicalize_forecast_methods(
            ["EWMA", "Parkinson"],
            require_known=False,
        )
        assert error is None
        assert canonical == ["ewma", "parkinson"]


class TestSuggestForecastMethods:
    def test_no_spurious_cross_family_suggestion(self):
        valid = ['theta', 'drift', 'seasonal_naive', 'analog', 'sf_constantmodel', 'sf_autoarima']
        assert fv.suggest_forecast_methods('nonexistent_method', valid) == []

    def test_unrelated_invalid_method_does_not_suggest_nan_model(self):
        valid = ["theta", "seasonal_naive", "sf_nanmodel"]

        assert fv.suggest_forecast_methods("not_a_model", valid) == []

    def test_close_typo_is_suggested(self):
        valid = ['theta', 'drift', 'seasonal_naive', 'analog']
        assert 'drift' in fv.suggest_forecast_methods('drft', valid)
        assert 'analog' in fv.suggest_forecast_methods('analg', valid)
