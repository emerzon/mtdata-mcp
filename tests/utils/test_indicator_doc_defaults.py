import pytest

from mtdata.utils.indicators import (
    _DEFAULT_MISSING,
    _parse_doc_default_value,
    infer_defaults_from_doc,
)


@pytest.mark.parametrize("token, expected", [
    ("1e-5", 0.00001), ("-2.5E+3", -2500.0), (".5", 0.5), ("+42", 42),
    ("True", True), ("false", False), ("None", None), ("null", None),
    ("'001'", "001"), ('"true"', "true"), ("sma", "sma"),
])
def test_doc_defaults_keep_scalar_types(token, expected):
    value = _parse_doc_default_value(token)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize("token", ["1 / 2", "[]", "{}", "1e999", ""])
def test_doc_defaults_do_not_infer_expressions_or_nonfinite_numbers(token):
    assert _parse_doc_default_value(token) is _DEFAULT_MISSING


@pytest.mark.parametrize("doc", [
    "demo(close, epsilon=1e-5)",
    "epsilon (float): Default: 1e-5.",
])
def test_documented_scientific_default_is_not_truncated_or_lost(doc):
    params = [{"name": "epsilon"}]
    infer_defaults_from_doc("demo", doc, params)
    assert params == [{"name": "epsilon", "default": 0.00001}]
