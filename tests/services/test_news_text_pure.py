from mtdata.services.news_text import normalize_news_text


def test_normalize_news_text_repairs_oem_smart_quote_mojibake() -> None:
    assert normalize_news_text("HPE\u00d4\u00c7\u00d6s stock") == "HPE\u2019s stock"


def test_normalize_news_text_repairs_double_encoded_oem_mojibake() -> None:
    garbled = "HPE\u00c3\u201d\u00c3\u2021\u00c3\u2013s stock"

    assert normalize_news_text(garbled) == "HPE\u2019s stock"


def test_normalize_news_text_strips_controls_and_compacts_whitespace() -> None:
    assert normalize_news_text("  Fed\x00  holds\r\nrates  ") == "Fed holds rates"


def test_normalize_news_text_repairs_cp1252_curly_apostrophe() -> None:
    garbled = "Here\u00e2\u20ac\u2122s everything you need to know."
    fed = "Fed\u00e2\u20ac\u2122s Lisa Cook denies committing mortgage fraud"

    assert normalize_news_text(garbled) == "Here\u2019s everything you need to know."
    assert normalize_news_text(fed) == (
        "Fed\u2019s Lisa Cook denies committing mortgage fraud"
    )


def test_normalize_news_text_repairs_apostrophe_when_whole_string_cannot_roundtrip() -> None:
    mixed = "Here\u00e2\u20ac\u2122s a title with \u2605 leftover unicode"

    assert normalize_news_text(mixed) == "Here\u2019s a title with \u2605 leftover unicode"
