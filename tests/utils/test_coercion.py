from mtdata.utils.coercion import (
    UNPARSED_BOOL,
    coerce_finite_float,
    coerce_optional_bool,
    is_explicit_false,
    parse_bool_like,
    round_finite,
    safe_float,
    split_top_level_csv,
)


def test_coerce_optional_bool_accepts_only_finite_bool_and_numeric_flags() -> None:
    assert coerce_optional_bool(True) is True
    assert coerce_optional_bool(0) is False
    assert coerce_optional_bool(1.0) is True
    assert coerce_optional_bool(float("nan")) is None
    assert coerce_optional_bool("false") is None


def test_parse_bool_like_accepts_canonical_boolean_values():
    for value in (True, 1, "true", " YES ", "y", "on"):
        assert parse_bool_like(value) is True
    for value in (False, 0, "false", " NO ", "n", "off"):
        assert parse_bool_like(value) is False


def test_parse_bool_like_distinguishes_null_and_unrecognized_values():
    assert parse_bool_like(None) is UNPARSED_BOOL
    assert parse_bool_like("null") is UNPARSED_BOOL
    assert parse_bool_like([], allow_none=True) is UNPARSED_BOOL
    assert parse_bool_like(None, allow_none=True) is None
    assert parse_bool_like("null", allow_none=True) is None


def test_split_top_level_csv_preserves_nested_and_quoted_commas() -> None:
    assert split_top_level_csv(
        'rsi(14),macd(12,26,9),label="fast,slow",bands[1,2]'
    ) == [
        "rsi(14)",
        "macd(12,26,9)",
        'label="fast,slow"',
        "bands[1,2]",
    ]
    assert split_top_level_csv(" a, ,b ") == ["a", "b"]
    assert split_top_level_csv("") == []


def test_is_explicit_false_distinguishes_missing_from_falsey_values():
    assert is_explicit_false(None) is False
    assert is_explicit_false(False) is True
    assert is_explicit_false(0) is True
    assert is_explicit_false("") is True
    assert is_explicit_false([]) is True
    assert is_explicit_false(True) is False
    assert is_explicit_false(1) is False


def test_round_finite_rounds_and_rejects_invalid_values():
    assert round_finite(1.23456, 2) == 1.23
    assert round_finite("1.239", 2) == 1.24
    assert round_finite(None, 2) is None
    assert round_finite(True, 2) is None
    assert round_finite(float("nan"), 2) is None
    assert round_finite("x", 2, on_invalid="passthrough") == "x"
    assert round_finite(1.23456, -3) == 1.0


def test_coerce_finite_float_accepts_numeric_and_string_values() -> None:
    class StringFloat:
        def __float__(self) -> float:
            raise TypeError

        def __str__(self) -> str:
            return "2.75"

    assert coerce_finite_float(3) == 3.0
    assert coerce_finite_float(" 1.25 ") == 1.25
    assert coerce_finite_float(StringFloat()) == 2.75


def test_coerce_finite_float_rejects_missing_invalid_and_non_finite_values() -> None:
    for value in (None, "not-a-number", float("nan"), float("inf"), "-inf"):
        assert coerce_finite_float(value) is None


def test_safe_float_uses_default_only_when_coercion_fails() -> None:
    assert safe_float("4.5", default=9.0) == 4.5
    assert safe_float("invalid", default=9.0) == 9.0
    assert safe_float(float("nan")) is None
