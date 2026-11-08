# Codebase untangling decisions

**Audience:** Contributor

This log records the structural audit and refactor started on 2026-11-08.
The user selected both comment cleanup and test cleanup. Tests remain the
regression safety net until structural changes are in place. See [the docs
index](README.md) for user documentation.

## Initial evidence

- Baseline: 311 Python modules and 1,591 internal module import edges,
  including `from package import module` edges.
- The AST inventory found 203 exception handlers containing `pass` and 21
  sleep calls. These are review indicators, not a reason to remove error
  handling or external retry policies mechanically.
- Baseline Ruff and tool schema checks pass (83 tools, no warnings).
- The initial backend run used the repository's mocked MT5 adapter. Because
  it overlapped file moves, its import failures are not baseline evidence.
  The two independent clock-sensitive Finviz failures were fixed in fixtures.
- Recent Git history already contains substantial cleanup. Preserve those
  public contracts and prefer demonstrable dependency improvements.

## Fixes and the problems they solve

1. **Give cross-domain contracts stable shared ownership.**
   - Forecast and services import error, execution logging, runtime metadata,
     and output contracts from `core`, which owns transport entry points.
   - Request context must move with logging/error handling to keep context
     identity shared; retaining duplicate context variables would break tracing.
   - Analytics request models live in `core/analytics_requests.py`, forcing
     all analytics engines to import the tool layer for their own inputs.
   - Pure symbol matching/ranking in `core/symbols/classify.py` imports
     `utils/symbol.py`, which imports the classifier back for suggestions.
2. **Separate transport and presentation policy from tool execution.**
   - `web_api_tools.py` imports mutation-policy constants from CLI runtime.
   - Web tool/radar error handling imports heavy history/forecast handlers.
   - `utils/minimal_output.py` imports trading positions and market-status
     tool modules just to format already-produced data.
   - `output_serialization.py` recursively passes a `compact_numbers` option
     whose float handler ignores it; finite JSON numbers must stay numeric.
   - The wait-event package exports private helpers solely for tests; expose
     the actual package entry points and test helpers at their owner modules.
3. **Make ensemble dispatch and metadata follow one registry contract.**
   - Ensemble adapters inspect callback identity/signatures and pass errors
     through mutable `_last_error`; these paths support test-only callbacks.
   - Registry fallback metadata duplicates the ensemble class metadata.
   - Engine and ensemble duplicate available-method lookup.
   - `patterns/common.py` imports bar completion through the candle service
     even though `utils/bar_completion.py` owns it.
4. **Make browser data dependencies explicit.**
   - `api/client.ts` changes a timeframe global as a side effect; workspace
     polling silently depends on the toolbar query having run.
   - Geometry, price-line helpers, and workspace depend on a chart component
     for a shared type. Put the contract in `types.ts`.
   - The date-only `lib/utils.ts` duplicates the ownership of `lib/time.ts`.
   - Denoise zero-phase capabilities are copied from Python into a frontend
     hardcoded set; use fetched metadata to migrate saved specifications.
   - Auth-memory tests inspect source text despite having behavioral seams.
5. **Clean comments and tests after the structural fixes.**
   - Remove comments that restate implementation in audited/touched areas.
   - Replace callback/type-spelling/source-string tests with observable
     behavior; keep tests that protect actual public and numerical contracts.
   - Test configuration snapshots manually enumerate model fields and can
     miss future settings; use the models' complete serialization contract.

## Requirements to retain

- Broker execution guards, quote freshness, trading dry-run behavior, and
  symbol visibility restoration represent external requirements.
- MT5 history polling, HTTP backoff, Windows atomic-file replacement retries,
  and asynchronous job polling wait for external resources.
- Optional numerical providers and lazy registration avoid requiring every
  forecasting backend for ordinary market-data use.
- Exchange holidays, FX weekends, crypto sessions, and provider schemas vary
  in reality and need explicit policy branches.
- Bootstrap registration repair serves reloads; lazy tool-prefix discovery
  keeps startup lightweight. Thin executable CLI entry points are intentional.
- UI forecast, backtest, and idea workflows have different lifecycle needs;
  a generic hook is not justified solely because their control flow resembles.
- Cross-domain analytics use of trading validation and forecasting engines
  represents real domain collaboration; relocating it wholesale would merely
  rename the dependency.

## Verification and follow-up

Implementation, independent review findings, and final checks are recorded
below as they complete. No source changes preceded this initial decision log.

### Completed implementation

- Moved shared contracts to `shared/` and analytics inputs to
  `analytics/requests.py`; all consumers and patch targets use the new owners.
- Colocated pure classification, matching, and ranking in
  `shared/symbol_catalog.py`; `utils/symbol.py` now handles gateway operations.
- Extracted shared mutation policy and pure market/trade presentation; HTTP
  error handling now belongs to `core/web_api_errors.py`.
- Removed unused serialization configuration, private output-shaping alias,
  test-only ensemble callback protocols, mutable callback error state, and
  duplicate ensemble metadata. Registry adapters return forecast/error tuples.
- Reduced wait-event exports and combined branches with identical requirements.
- Browser timeframe and denoise capabilities now arrive through shared query
  data. Legacy denoise settings wait for metadata; explicit settings remain
  intact. Chart types and date formatting live outside UI components.
- Replaced source-text and type-existence checks with behavior assertions;
  ensemble tests exercise registry dispatch, cross-validation, and fitting.
- Replaced the permanently skipped genetic-search smoke class with active,
  deterministic coverage of the real generational loop and hint limits.
- Fixed symbol ranking fixtures that left stale-history refresh unmocked and
  incurred broker retry sleeps. All 109 symbol/scan tests now pass alone in
  about two seconds; refresh-specific tests retain their own responses.
- Configuration fixtures snapshot all fields of plain config objects and the
  complete serializable guardrail policy, including news/options settings and
  the active timezone-warning flag. Removed partial fallback resets.
- Removed implementation-restating comments while retaining explanations of
  external formats, clocks, provider limitations, and broker behavior.

### Independent review and retained decisions

- Core and domain agents reviewed each other's implementations. Function AST
  comparison verified the shared contract moves; encoding review caught and
  repaired two text artifacts. No remaining actionable issue in those slices.
- Frontend review caught a modal race during capability loading. Apply now
  waits for a valid resolved denoise mode, covered by rendered markup and
  behavior checks. Metadata error states remain retryable.
- Two tiny field-normalization validators and domain warning append helpers
  remain local: identical mechanics alone do not establish shared requirements.
- Analytics' remaining three imports into trading represent actual domain
  collaboration. Bootstrap composition and executable entry points necessarily
  know the tool layer. These are not dependency inversions to erase.
- Optional-provider and platform-dependent test skips are retained because
  their environments genuinely vary.
- Independent review also removed 13 test patches for a nonexistent warning
  flag and corrected snapshot restoration to use the active settings flag.

### Repeatable inventory

Run `python scripts/architecture_inventory.py` for a JSON skeleton of module
definitions, static imports, literal branches, exception/sleep review candidates,
and conditional test skips. It parses source without importing the application
or connecting to MT5. Dynamic registry imports still need manual review.

Using that same analyzer on Git HEAD and the refactored tree, internal edges
changed from 1,591 to 1,588, while cross-domain imports into `core` fell from
24 to 3 (excluding bootstrap composition and `__main__`). Modules changed from
311 to 315: the new files establish owners for policies previously inside tools.
Exception handlers containing `pass` fell from 203 to 201; all 21 sleep calls
remain for external wait/retry requirements. Raw counts are audit aids, not
quotas: a literal branch is not itself a design defect.

The final inventory has 69 conditional skip/import candidates and no
unconditionally skipped test decorators. Their reasons are dependency,
platform, or external-data availability, rather than known failing behavior.

### Final verification

- `python -m pytest tests -q --tb=short -o faulthandler_timeout=45`:
  **10,982 passed, 3 skipped, 15 subtests passed** in 284.53 seconds.
- Shared-contract, symbol, service, and analytics verification after the final
  fixture cleanup: **550 passed**. Configuration/genetic-search checks:
  **78 passed**. Independent symbol/scan run: **109 passed**.
- `python -m ruff check src tests scripts/architecture_inventory.py`: passed.
- `python scripts/schema_evaluate.py`: **83/83 tools**, no errors or warnings.
- In `webui`, `npm test`: **160 passed across 29 files**;
  `npm run typecheck` and `npm run build`: passed.
- `git diff --check`: passed.

The backend reports the existing Starlette/httpx deprecation warning; Vite
reports the existing large-chunk warning. No dependencies or broker execution
policies changed. Tests use the mocked MT5 boundary. No commits were created.
