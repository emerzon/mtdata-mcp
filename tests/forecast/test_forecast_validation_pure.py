"""Tests for src/mtdata/forecast/forecast_validation.py"""

import mtdata.forecast.forecast_validation as fv


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
