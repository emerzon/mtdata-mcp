from mtdata.bootstrap.env import get_bool_env, get_csv_env, get_float_env, get_int_env


def test_get_bool_env_uses_default_when_unset_or_invalid(monkeypatch, caplog) -> None:
    monkeypatch.delenv("MTDATA_TEST_BOOL", raising=False)
    assert get_bool_env("MTDATA_TEST_BOOL", default=True) is True

    monkeypatch.setenv("MTDATA_TEST_BOOL", "invalid")
    assert get_bool_env("MTDATA_TEST_BOOL", default=True) is True
    assert "Invalid boolean MTDATA_TEST_BOOL='invalid'" in caplog.text


def test_get_bool_env_accepts_project_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "YES", "y", "on"):
        monkeypatch.setenv("MTDATA_TEST_BOOL", value)
        assert get_bool_env("MTDATA_TEST_BOOL") is True


def test_get_bool_env_accepts_project_false_values(monkeypatch) -> None:
    for value in ("0", "false", "NO", "n", "off"):
        monkeypatch.setenv("MTDATA_TEST_BOOL", value)
        assert get_bool_env("MTDATA_TEST_BOOL", default=True) is False


def test_get_int_env_warns_and_falls_back(monkeypatch, caplog) -> None:
    monkeypatch.delenv("MTDATA_TEST_INT", raising=False)
    assert get_int_env("MTDATA_TEST_INT", 8000) == 8000

    monkeypatch.setenv("MTDATA_TEST_INT", " 42 ")
    assert get_int_env("MTDATA_TEST_INT", 8000) == 42

    monkeypatch.setenv("MTDATA_TEST_INT", "abc")
    assert get_int_env("MTDATA_TEST_INT", 8000) == 8000
    assert "Invalid MTDATA_TEST_INT='abc'" in caplog.text


def test_get_float_env_warns_and_falls_back(monkeypatch, caplog) -> None:
    monkeypatch.setenv("MTDATA_TEST_FLOAT", "")
    assert get_float_env("MTDATA_TEST_FLOAT", 1.5) == 1.5
    assert "MTDATA_TEST_FLOAT is blank" in caplog.text


def test_get_csv_env_falls_back_when_empty(monkeypatch) -> None:
    monkeypatch.delenv("MTDATA_TEST_CSV", raising=False)
    assert get_csv_env("MTDATA_TEST_CSV", ("a", "b")) == ("a", "b")

    monkeypatch.setenv("MTDATA_TEST_CSV", "  x, y , ")
    assert get_csv_env("MTDATA_TEST_CSV", ("a", "b")) == ("x", "y")

    monkeypatch.setenv("MTDATA_TEST_CSV", "   ")
    assert get_csv_env("MTDATA_TEST_CSV", ("a", "b")) == ("a", "b")
