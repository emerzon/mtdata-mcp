from types import SimpleNamespace
from unittest.mock import patch

from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
from mtdata.utils.barriers import get_pip_size

from ._helpers import _BARRIER_OPT_ROOT, _BarrierTestBase


class TestPipResolution(_BarrierTestBase):
    def test_broker_path_identified_fx_uses_same_pip_for_barriers_and_costs(self):
        self._set_flat_history(price=1.1)
        info = SimpleNamespace(point=0.00001, digits=5, path="Forex/Majors")
        with (
            patch(f"{_BARRIER_OPT_ROOT}._get_pip_size", side_effect=get_pip_size),
            patch("mtdata.utils.barriers.get_symbol_info_cached", return_value=info),
        ):
            result = forecast_barrier_optimize(
                symbol="BrokerPair", timeframe="H1", horizon=3,
                method="mc_gbm", direction="long", mode="pips",
                tp_min=10, tp_max=10, tp_steps=1,
                sl_min=10, sl_max=10, sl_steps=1,
                params={"spread_pips": 1.0, "use_live_price": False},
                viable_only=False,
            )
        assert "error" not in result, result
        assert result["trading_costs"]["spread_pips"] == 1.0
        assert result["trading_costs"]["cost_per_trade"] == 1.0
