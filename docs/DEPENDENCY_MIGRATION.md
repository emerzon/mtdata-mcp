# Dependency Migration Status

**Audience:** Contributor

**Audited:** 2026-08-21
**Runtime:** Windows x86-64, CPython 3.14.3, pip 26.x

This is the compatibility snapshot for direct project dependencies. Lower bounds in
`pyproject.toml` identify the versions exercised by this migration; upper bounds are
kept only where an upstream package currently makes the newer major/minor line
unresolvable or where a dependent framework has not migrated yet.

## Completed

| Area | Previous | Migrated to | Verification |
|------|----------|-------------|--------------|
| MCP SDK | 1.28.1 | 1.29.0 (`<2`) | Server and Web API import smoke; MCP 2 dropped `mcp.server.fastmcp` |
| Unused direct pins | Prefect FastMCP, PyYAML, h5py | Removed | No remaining imports; Prefect FastMCP is distinct from MCP SDK FastMCP |
| Over-declared pins | matplotlib, TA-Lib, torchvision, extra hmmlearn | Demoted | matplotlib unused; TA-Lib is pandas-ta's optional oracle; torchvision unused; hmmlearn already core |
| Patch floors | dotenv 1.2.2, holidays 0.102, SciPy 1.18.0, Uvicorn 0.52.1, Transformers 5.15.0, Hub 1.27, numba 0.66 | 1.2.3 / 0.103 / 1.18.1 / 0.52.4 / 5.15.1 / 1.28 / 0.67 | PyPI latest within existing upper bounds; numba 0.67 has a Windows cp314 wheel |
| StatsForecast | 1.7.6 | 2.1.1 | Windows cp314 wheel resolution plus a real `Naive` forecast |
| sktime | 1.0.1 | 1.1.0 | Resolver check plus a real `NaiveForecaster` forecast |
| TimesFM | Git commit / 2.0.0 / PyPI 2.0.2 | PyPI 3.0.1 | Extra still exposes TimesFM 2.5; new `timesfm3` method uses TimesFM 3.0 (non-commercial weights) |
| ycnbc | Git commit `d5e5029` | PyPI 1.0.12 | Same 1.0.12 release; `[news-ycnbc]` / `[all-git]` no longer use a Direct URL (PyPI rejects git extras) |
| Foundation stack | Torch 2.11 / Transformers 5.12 floor | Torch 2.13 / Transformers 5.15.1 floor | Windows cp314 wheel resolution |
| Scientific/runtime packages | Earlier compatible floors | Latest compatible releases | Full `[all]` resolver pass |
| HNSW search | Opt-in helper only | `[pattern-search-hnsw]` and `[all]` | Fresh local cp314 wheel built from `hnswlib` 0.8.0 source; CI installs the extra |
| Web UI | React 18, Lightweight Charts 4, Tailwind 3, TypeScript 5 | React 19.2, Lightweight Charts 5.2.1, Vite 8.2.2, Tailwind 4.3, TypeScript 7.0 | Frontend tests, type-check, production build, and visual smoke |

The Python floor updates also include current compatible releases of MetaTrader5,
SciPy, QuantLib, LightGBM, FastAPI, Uvicorn, Sentence Transformers, Hugging Face
Hub, and the timezone/date libraries. See `pyproject.toml` for the canonical
ranges.

## Still blocked or intentionally deferred

| Library | Latest checked | Blocker / decision | Revisit when |
|---------|----------------|--------------------|--------------|
| pandas | 3.0.5 | StatsForecast 2.1.1, sktime 1.1.0, and MLForecast 1.1.0 require `pandas<3` | All three upstream constraints permit pandas 3 and adapter tests pass |
| NumPy | 2.5.2 | sktime 1.1.0 still requires `numpy<2.5` (numba 0.67 allows `numpy<2.6`) | sktime raises its ceiling |
| scikit-learn | 1.9.0 | sktime 1.1.0 requires `scikit-learn<1.8` | sktime raises its ceiling |
| ruptures | 1.1.10 | Stable release declares `Requires-Python <3.14` and has no cp314 wheel | A stable release accepts real Python 3.14 patch versions |
| NeuralForecast | 3.2.1 | Requires Ray; Ray 2.57 has cp314 wheels for Linux/macOS but not Windows | Ray publishes a Windows cp314 wheel or NeuralForecast makes Ray optional |
| MCP SDK | 2.0.0 | MCP 2 renamed `FastMCP` to `MCPServer` and removed `mcp.server.fastmcp` | Server import paths are migrated |
| hmmlearn | 0.3.3 | No upstream cp314 wheel, but the MSVC source build succeeds and remains in core | Prefer an upstream wheel to remove the compiler prerequisite |
| hnswlib | 0.8.0 | No upstream wheels at all; the validated MSVC build is now supported | Prefer an upstream Windows cp314 wheel |
| GluonTS / Lag-Llama | GluonTS 0.17.0 | GluonTS now resolves, but mtdata has no GluonTS or Lag-Llama adapter to enable | An adapter and model contract are implemented and tested |
| stock-pattern | repository `main` | Repository still has no `pyproject.toml` or `setup.py` | Upstream becomes pip-installable or mtdata vendors a maintained adapter |

## Reproduction checks

```powershell
# Resolve the complete supported package-index stack.
python -m pip install --dry-run --ignore-installed -e ".[all]"

# Verify source builds that have no upstream cp314 wheels.
python -m pip wheel --no-cache-dir --no-deps "hmmlearn==0.3.3"
python -m pip wheel --no-cache-dir --no-deps "hnswlib==0.8.0"

# Frontend verification.
cd webui
npm test
npm run typecheck
npm run build
```

Upstream package metadata used for the compatibility boundaries:
[StatsForecast](https://pypi.org/project/statsforecast/),
[sktime](https://pypi.org/project/sktime/),
[TimesFM](https://pypi.org/project/timesfm/),
[numba](https://pypi.org/project/numba/),
[ruptures](https://pypi.org/project/ruptures/),
[NeuralForecast](https://pypi.org/project/neuralforecast/),
[Ray](https://pypi.org/project/ray/), and the
[MCP Python SDK v1→v2 migration guide](https://py.sdk.modelcontextprotocol.io/migration/).
