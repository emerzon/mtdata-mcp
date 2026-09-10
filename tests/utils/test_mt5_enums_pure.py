import gc
import weakref
from types import ModuleType

from mtdata.utils import mt5_enums


def test_constants_by_prefix_caches_dir_scan() -> None:
    mt5_enums._PREFIX_CONSTANTS_CACHE.clear()
    module = ModuleType("fake_mt5_dir_scan")
    module.ORDER_TYPE_BUY = 0
    module.ORDER_TYPE_SELL = 1
    module.OTHER = 99
    first = mt5_enums._constants_by_prefix(module, "ORDER_TYPE_")
    module.ORDER_TYPE_BUY_LIMIT = 2
    second = mt5_enums._constants_by_prefix(module, "ORDER_TYPE_")

    assert first == {0: "ORDER_TYPE_BUY", 1: "ORDER_TYPE_SELL"}
    assert second is first
    assert 2 not in second


def test_constants_cache_does_not_leak_across_module_instances() -> None:
    mt5_enums._PREFIX_CONSTANTS_CACHE.clear()
    first_module = ModuleType("fake_mt5_cache_probe")
    first_module.DEAL_REASON_TP = 5
    probe = weakref.ref(first_module)

    assert mt5_enums._constants_by_prefix(first_module, "DEAL_REASON_") == {
        5: "DEAL_REASON_TP"
    }

    del first_module
    gc.collect()
    assert probe() is None
    assert len(mt5_enums._PREFIX_CONSTANTS_CACHE) == 0

    second_module = ModuleType("fake_mt5_cache_probe")
    assert mt5_enums._constants_by_prefix(second_module, "DEAL_REASON_") == {}
