from types import SimpleNamespace

from mtdata.utils import mt5_enums


def test_constants_by_prefix_caches_dir_scan() -> None:
    mt5_enums._PREFIX_CONSTANTS_CACHE.clear()
    module = SimpleNamespace(ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, OTHER=99)
    first = mt5_enums._constants_by_prefix(module, "ORDER_TYPE_")
    module.ORDER_TYPE_BUY_LIMIT = 2
    second = mt5_enums._constants_by_prefix(module, "ORDER_TYPE_")

    assert first == {0: "ORDER_TYPE_BUY", 1: "ORDER_TYPE_SELL"}
    assert second is first
    assert 2 not in second
