from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_symbol_scan_import_does_not_load_optional_denoise_stack() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(source_root), environment.get("PYTHONPATH")),
        )
    )
    program = """
import json
import sys

import mtdata.core.symbols.scan

print(json.dumps(sorted(sys.modules)))
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    modules = set(json.loads(completed.stdout.strip().splitlines()[-1]))

    assert "mtdata.services.data_service.candles" not in modules
    assert not any(
        name == "PyEMD"
        or name.startswith("PyEMD.")
        or name == "matplotlib"
        or name.startswith("matplotlib.")
        for name in modules
    )
