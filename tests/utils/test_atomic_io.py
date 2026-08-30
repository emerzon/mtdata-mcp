from __future__ import annotations

import os

import pytest

from mtdata.utils import atomic_io


def test_atomic_write_bytes_retries_short_writes(tmp_path, monkeypatch):
    target = tmp_path / "artifact.bin"
    expected = b"complete model artifact"
    real_write = os.write
    write_sizes: list[int] = []

    def short_write(fd: int, data: bytes | memoryview) -> int:
        chunk = data[: min(3, len(data))]
        written = real_write(fd, chunk)
        write_sizes.append(written)
        return written

    monkeypatch.setattr(atomic_io.os, "write", short_write)

    atomic_io.atomic_write_bytes(target, expected)

    assert target.read_bytes() == expected
    assert len(write_sizes) > 1


def test_atomic_write_bytes_rejects_zero_progress(tmp_path, monkeypatch):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"previous generation")

    monkeypatch.setattr(atomic_io.os, "write", lambda _fd, _data: 0)

    with pytest.raises(OSError, match="made no progress"):
        atomic_io.atomic_write_bytes(target, b"replacement generation")

    assert target.read_bytes() == b"previous generation"
    assert list(tmp_path.glob("*.tmp")) == []
