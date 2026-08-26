
import importlib.util
import unittest

import numpy as np
import pandas as pd

from mtdata.forecast.forecast import execute_forecast
from mtdata.forecast.forecast_engine import forecast_engine
from mtdata.forecast.forecast_registry import ForecastRegistry
from mtdata.forecast.interface import ForecastResult

# Ensure engine is imported to register methods

class TestUnifiedForecast(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        self.series = pd.Series(np.random.randn(100) + 100, index=dates)
        # Add epoch for engine
        self.df = pd.DataFrame({'close': self.series})
        self.df['__epoch'] = self.series.index.astype(np.int64) // 10**9
        self.df['time'] = self.df['__epoch'].astype(float)
        
        # Mock fetch_history to return our dummy data
        # We'll patch it in the test methods or just test the registry/methods directly first
        
    def test_registry_registration(self):
        """Test that methods are registered correctly."""
        methods = ForecastRegistry.get_all_method_names()
        print(f"Registered methods: {methods}")
        self.assertIn('naive', methods)
        self.assertIn('theta', methods)
        self.assertIn('sf_autoarima', methods)
        self.assertIn('mlf_rf', methods)
        self.assertIn('sktime', methods)
        self.assertIn('skt_naive', methods)
        self.assertIn('mlforecast', methods)
        
    def test_classical_method_direct(self):
        """Test calling a classical method directly via registry."""
        forecaster = ForecastRegistry.get('naive')
        res = forecaster.forecast(self.series, horizon=10, seasonality=1, params={})
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)
        
    def test_statsforecast_method_direct(self):
        """Test calling a statsforecast method directly."""
        if importlib.util.find_spec("statsforecast") is None:
            self.skipTest("statsforecast not installed")
            
        forecaster = ForecastRegistry.get('sf_seasonalnaive')
        res = forecaster.forecast(self.series, horizon=10, seasonality=24, params={})
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)

    def test_statsforecast_ci(self):
        """Test StatsForecast CI extraction."""
        if importlib.util.find_spec("statsforecast") is None:
            self.skipTest("statsforecast not installed")
            
        forecaster = ForecastRegistry.get('sf_seasonalnaive')
        # Pass ci_alpha via kwargs (simulating engine)
        res = forecaster.forecast(self.series, horizon=10, seasonality=24, params={}, ci_alpha=0.05)
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)
        self.assertIsNotNone(res.ci_values)
        self.assertEqual(len(res.ci_values), 2)
        self.assertEqual(len(res.ci_values[0]), 10)
        self.assertEqual(len(res.ci_values[1]), 10)

    def test_generic_statsforecast(self):
        """Test generic StatsForecast method."""
        if importlib.util.find_spec("statsforecast") is None:
            self.skipTest("statsforecast not installed")
            
        # Test using the generic 'statsforecast' method with model_name param
        forecaster = ForecastRegistry.get('statsforecast')
        res = forecaster.forecast(self.series, horizon=10, seasonality=24, params={'model_name': 'AutoARIMA'})
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)
        
        # Test using a dynamically registered alias (e.g. sf_adida)
        forecaster = ForecastRegistry.get('sf_adida')
        res = forecaster.forecast(self.series, horizon=10, seasonality=24, params={})
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)

    def test_sktime_method(self):
        """Test Sktime method."""
        if importlib.util.find_spec("sktime") is None:
            self.skipTest("sktime not installed")
        try:
            import sktime  # noqa: F401
        except Exception:
            self.skipTest("sktime is installed but fails to import")
            
        # Test using generic 'sktime' method
        forecaster = ForecastRegistry.get('sktime')
        # Use NaiveForecaster as it's simple
        res = forecaster.forecast(self.series, horizon=10, seasonality=1, 
                                 params={'estimator': 'sktime.forecasting.naive.NaiveForecaster', 'strategy': 'last'})
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)
        
        # Test using registered alias 'skt_naive'
        forecaster = ForecastRegistry.get('skt_naive')
        res = forecaster.forecast(self.series, horizon=10, seasonality=1, params={})
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)

    def test_mlforecast_method_direct(self):
        """Test calling an mlforecast method directly."""
        if importlib.util.find_spec("mlforecast") is None:
            self.skipTest("mlforecast not installed")
            
        forecaster = ForecastRegistry.get('mlf_rf')
        res = forecaster.forecast(self.series, horizon=10, seasonality=1, params={'lags': [1, 2]})
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)

    def test_generic_mlforecast(self):
        """Test generic MLForecast method."""
        if importlib.util.find_spec("mlforecast") is None or importlib.util.find_spec("sklearn") is None:
            self.skipTest("mlforecast or sklearn not installed")
            
        # Test using generic 'mlforecast' method with LinearRegression
        forecaster = ForecastRegistry.get('mlforecast')
        res = forecaster.forecast(self.series, horizon=10, seasonality=1, 
                                 params={'model': 'sklearn.linear_model.LinearRegression', 'lags': [1, 2]})
        self.assertIsInstance(res, ForecastResult)
        self.assertEqual(len(res.forecast), 10)
        
    def test_engine_dispatch(self):
        """Test that forecast_engine dispatches correctly."""
        # We need to mock _fetch_history since engine calls it
        from unittest.mock import patch
        
        with patch('mtdata.forecast.forecast_engine._fetch_history') as mock_fetch:
            mock_fetch.return_value = self.df
            
            # Test naive
            res = forecast_engine('dummy', method='naive', horizon=5)
            self.assertTrue(res['success'])
            self.assertEqual(res['method'], 'naive')
            self.assertEqual(len(res['forecast_price']), 5)
            
            # Test theta
            res = forecast_engine('dummy', method='theta', horizon=5)
            self.assertTrue(res['success'])
            self.assertEqual(res['method'], 'theta')
            
            # Test ensemble
            res = forecast_engine('dummy', method='ensemble', horizon=5, params={'methods': 'naive,theta'})
            self.assertTrue(res['success'])
            self.assertIn('ensemble', res)
            self.assertEqual(len(res['forecast_price']), 5)

    def test_engine_returns_quantity_outputs_return_and_price(self):
        """Return-mode forecasts expose forecast_return and reconstructed prices."""
        from unittest.mock import patch
        df = pd.DataFrame({
            'time': np.arange(50, dtype=float),
            'close': np.linspace(100.0, 105.0, 50),
        })
        with patch('mtdata.forecast.forecast_engine._fetch_history', return_value=df):
            res = forecast_engine('dummy', method='naive', horizon=3, quantity='return')
        self.assertTrue(res['success'])
        self.assertIn('forecast_return', res)
        self.assertEqual(len(res['forecast_return']), 3)
        self.assertIn('forecast_price', res)
        self.assertEqual(len(res['forecast_price']), 3)

    def test_wrapper_dimred_no_nameerror(self):
        """Wrapper dimred builder should not raise when dimred_method is set."""
        from unittest.mock import patch
        df = pd.DataFrame({
            'time': np.arange(30, dtype=float),
            'open': np.linspace(100, 101, 30),
            'high': np.linspace(101, 102, 30),
            'low': np.linspace(99, 100, 30),
            'close': np.linspace(100, 101, 30),
            'volume': np.ones(30) * 1000,
        })
        with patch('mtdata.forecast.forecast_engine._fetch_history', return_value=df):
            res = execute_forecast(
                symbol='X',
                timeframe='H1',
                method='naive',
                horizon=3,
                features={
                    'include': 'ohlcv',
                    'observed_future_policy': 'carry_forward',
                },
                dimred_method='pca',
            )
        self.assertTrue(res['success'])

if __name__ == '__main__':
    unittest.main()
