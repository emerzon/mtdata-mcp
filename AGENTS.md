# Repository Guidelines

## Project Structure & Module Organization

Python code lives in `src/mtdata/`. Runtime setup is in `bootstrap/`; CLI, MCP, Web API, reporting, regimes, and trading tools are in `core/`; market access belongs in `services/`; forecasting and pattern logic live in their named packages; shared schemas and utilities are in `shared/` and `utils/`. The React/Vite client is under `webui/src/`. Keep tests in `tests/`, grouped by the corresponding domain (`tests/core/`, `tests/services/`, and so on). Documentation belongs in `docs/`, maintenance utilities in `scripts/`, and reusable agent prompts in `prompts/`.

## Build, Test, and Development Commands

- `pip install -e ".[web,forecast-classical,pattern-search-hnsw]" pytest ruff` installs the backend and CI test tools for Python 3.14.
- `ruff check src tests` runs Python lint and import-order checks.
- `python scripts/schema_evaluate.py` validates public tool schemas.
- `python -m pytest tests` runs the backend suite; pass a test path or `-k expression` for focused work.
- `mtdata-webapi` serves the API and built UI locally.
- From `webui/`, run `npm ci`, `npm run dev`, `npm test`, `npm run typecheck`, and `npm run build` to install, develop, verify, and package the frontend.

## Coding Style & Naming Conventions

Use four spaces in Python, type hints on public interfaces, `snake_case` for modules/functions, and `PascalCase` for classes. Ruff rules in `pyproject.toml` are authoritative. Frontend code uses strict TypeScript, two-space indentation, `PascalCase` React components, and `camelCase` hooks/helpers. Preserve established single-quote, semicolon-free TS style.

## Testing Guidelines

Pytest files and functions use the `test_*.py` / `test_*` pattern. Frontend tests use `*.test.ts` with Vitest. Add focused regression tests beside the affected domain and cover error paths and public contracts. No numeric coverage threshold is configured; all lint, schema, backend, frontend test, type-check, and build jobs must pass.

## Commit & Pull Request Guidelines

Follow the history's concise imperative format, preferably Conventional Commits such as `fix(cli): clarify output contract` or `refactor: extract shared helpers`. Keep each commit scoped. Pull requests should summarize behavior changes, list verification commands, link relevant issues, and include screenshots for UI changes. Call out new dependencies, configuration changes, and any MT5 or broker-facing risk.

## Documentation

User-facing docs live in `docs/` and the root `README.md`. Follow
[docs/STYLE.md](docs/STYLE.md): helpful, friendly, jargon-aware, progressive
disclosure, calm safety. Tag pages **User**, **Operator**, or **Contributor**.
New tools need a User-page link, not only a CLI table row. Keep Web API trading
claims aligned with the Tools invoke path and [docs/TRADING_SAFETY.md](docs/TRADING_SAFETY.md).

## Security & Configuration

Copy settings from `.env.example`; never commit credentials, API tokens, or account details. Use an MT5 demo account for development and keep trading operations in `--dry-run true` unless live execution is explicitly intended.
