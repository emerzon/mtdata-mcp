# Web UI Development Goal

**Status:** Active goal  
**Audience:** Contributor  
**Related:** [WEBUI.md](WEBUI.md) (User tour) · [WEB_API.md](WEB_API.md) · [SETUP.md](SETUP.md) · [STYLE.md](STYLE.md) · `webui/AGENTS.md`

---

## North star

Make the bundled chart workspace at `/app` a **first-class research console** for mtdata: modern, fast, usable on real screens (desktop-first, tablet-capable, mobile-usable), and **feature-complete against the Web API**—with a clear path to expose high-value backend capabilities that still exist only on CLI/MCP.

The UI should not be a thin demo. It should be the default interactive surface for market context, levels, denoising, forecasting, volatility, and backtesting—without requiring users to leave the browser for routine research work.

---

## Why this goal exists

| Reality today | Target |
|---|---|
| Modular chart workspace with dedicated panels, feature hooks, and pure client helpers | Keep boundaries clear as dedicated research experiences grow |
| Core routes and the generic Tools runner cover the documented Web API | Preserve complete route classification and add custom UI only where it improves a workflow |
| Responsive toolbar and adaptive panels support desktop, tablet, and mobile widths | Finish keyboard, accessibility, and small-screen polish across every panel |
| Pure Vitest coverage, strict TypeScript, typecheck, and build checks are established | Add component/e2e confidence and a frontend lint gate without slowing iteration |
| The API exposes a broader catalog than should become custom chart widgets | Prioritize high-value read-only research surfaces and keep mutations explicitly guarded |

---

## Current capability map

### What the UI already does well

- **Chart workspace:** symbol search, timeframes, infinite left history, timezone modes (UTC / local / server)
- **Live market:** tick poll (bid / ask / last), live incomplete candle, reload and empty/error status surface
- **Overlays:** pivots, support/resistance, indicators, confluence, volume profile, trade ideas, and read-only exposure
- **Preprocessing:** chart-level denoise with method metadata UI
- **Analysis panel:** price forecast, volatility forecast, rolling backtest (with advanced options: denoise, params)
- **Discovery:** persistent watchlist radar, session strip, trained-model browser, and schema-driven Tools runner
- **Auth:** in-memory Bearer token for remote/tokenized API access
- **Stack:** React 19, Vite, Tailwind, TanStack Query, lightweight-charts, axios → `/api/v1`

### Web API endpoints vs UI usage

The canonical route-by-route matrix is
**[WEBUI_API_COVERAGE.md](WEBUI_API_COVERAGE.md)**. It records each route's
method, status, UI entry, and rationale. Keep endpoint inventory there instead
of duplicating it in this goal. The current dedicated surfaces cover chart
data, geometry/levels, denoise, forecasts, volatility, backtests, trade-idea
previews, watchlist context, models, health/readiness, and the generic catalog.

### Explicit non-parity (historical note, then current boundary)

**Then (dedicated chart routes only):** trading, regimes, patterns, reports, Finviz, options, news, causal tools, and wait-events lived only on CLI/MCP.

**Now:** `GET/POST /api/v1/tools*` plus the SPA **Tools** runner expose almost
the full catalog. Dedicated chart widgets remain the comfortable path for
candles, levels, denoise, forecast, volatility, and backtest. Three tools are
deliberately omitted from synchronous invoke:

- `forecast_tune_optuna` / `forecast_tune_genetic` — no HTTP progress or cancel contract (run via CLI/MCP).
- `wait_event` — blocking waits have no HTTP progress or cancel contract; see [WAIT_EVENT.md](WAIT_EVENT.md).

Live `trade_*` mutations from the SPA still require an explicit confirm gate, dry-run defaults, and the [trading safety](TRADING_SAFETY.md) rules. That is the safety design; it is not “trading is out of the Web API.”

The goal is not “rebuild every tool as a custom widget on day one.” It is:

1. **Parity with every Web API route and its useful parameters**
2. **Deliberate Web API + UI expansion** for the highest-value research features
3. **Never** expose unsafe live trading from the SPA without confirmations, dry-run defaults, and env guardrails — aligned with [TRADING_SAFETY.md](TRADING_SAFETY.md)

---

## Goal pillars

### 1. Preserve Web API parity and expand deliberately

**Parity contract**

- Every documented Web API route is either (a) reachable from the UI with clear affordances, or (b) listed as intentional omission with rationale.
- The route matrix is updated in the same change as any API or UI coverage change.
- Common workflows get typed, dedicated controls; uncommon tools remain usable through the schema-driven runner.
- New mutation surfaces retain confirmation, dry-run defaults, and account guardrails.

**Next expansion track**

Prioritize dedicated experiences where chart context materially improves the
generic tool result:

| Priority | Domain | User value |
|---|---|---|
| P1 | Regime overlays and high-signal pattern markers | Put forecast trust and event context on the chart |
| P1 | Backtest result visualization | Make reliability, history sufficiency, and trade metrics easier to compare |
| P2 | Report export / snapshot | Produce a shareable research artifact from the current workspace |
| P2 | Screener and news discovery | Move from a watchlist to broader read-only symbol discovery |
| P3 | Multi-forecast comparison | Compare a small, bounded set of methods without chart clutter |
| — | Dedicated live-order UI | Out of scope until it has a separate safety-reviewed interaction design |

Expansion always lands as: **backend route contract → client types → UI surface → tests → docs**.

### 2. Responsive, usable, modern UX

**Principles**

- **Chart-first:** the chart remains the hero; chrome collapses under density pressure.
- **Progressive disclosure:** simple defaults; advanced params behind clear “Advanced” sections (already started in Forecast).
- **Responsive, not just “shrinks”:**
  - Desktop (≥1280): toolbar + side panel + metrics as today
  - Tablet: collapsible tool groups, full-height sheet for forecast
  - Mobile: bottom sheet / full-screen panel, overflow menus for toolbar actions, touch-friendly hit targets (≥44px)
- **Modern look:** keep the dark slate terminal aesthetic; tighten spacing, hierarchy, motion, and empty/loading/error states into one consistent design language (reuse `panel` / `btn` / `input` primitives; avoid one-off styles).
- **Usability:**
  - Keyboard: focus order, Esc closes panels/modals, `/` or `s` focuses symbol search where safe
  - Feedback: toast or inline status for long forecast/backtest jobs; never silent failure
  - Persistence: symbol, timeframe, denoise, panel prefs in localStorage (token never persisted—already correct)
  - Accessibility: labels, `aria-*` on toggles, contrast on overlays, reduced-motion respect

### 3. Performance and perceived speed

- Keep chart interactions at 60fps where possible (lightweight-charts stays the renderer; avoid React re-render thrash on every poll).
- Query discipline: stable React Query keys, abort on symbol/tf change (partially present), sensible live poll intervals by timeframe.
- Avoid refetch storms when toggling overlays or opening panels.
- Bundle: code-split heavy panels (forecast/backtest) if main chunk grows; keep production `dist` lean for FastAPI static serve.
- History: efficient merge of live tip + paginated left history; cap client-side bar retention with clear “load more” semantics.
- Long jobs: optimistic UI + cancel where API allows; clear progress for backtests.

### 4. Engineering quality (close the dev gaps)

| Gap | Target |
|---|---|
| Tests emphasize pure lib/API helpers | Add component tests for critical panels and an optional Playwright smoke for `/app` load + symbol select |
| No frontend lint in package scripts | Add ESLint (TS + React hooks) or Biome; wire `npm run lint` |
| File-map drift as the UI grows | Keep `webui/AGENTS.md` aligned with feature hooks, panels, and pure helpers |
| Some large hooks and panels remain | Continue extracting feature-owned state and shared UI primitives |
| Types hand-maintained vs API | Keep `types.ts` aligned with compact Web API payloads; document any intentional subset |
| No visual regression | Not required day one; snapshot only if it pays for itself |
| Docs | This goal + WEB_API stay the source of truth for “what the UI should expose” |

**Definition of done for quality (baseline)**

- `npm run typecheck` and `npm test` clean in CI-or-manual checklist
- New API client methods ship with types + at least one pure test for error/params shaping when non-trivial
- No `any` without justification (strict TS already expected)

---

## Success criteria

The goal is met when all of the following are true:

1. **API coverage scorecard:** 100% of Web API routes classified (used / intentional omit); zero “forgotten” routes like `/models`.
2. **Core workflows (manual or automated smoke):**
   - Open `/app` → select symbol → see candles
   - Toggle live, bid/ask/last, pivots, S/R, denoise
   - Run price forecast → see overlay + metrics
   - Run volatility + backtest with readable results
   - Auth token path works against tokenized server
3. **Responsive:** usable forecast and toolbar flows at 375px, 768px, and 1440px widths without horizontal page scroll traps.
4. **Performance:** symbol switch and live poll remain snappy on a typical desktop; no unbounded memory growth after long live sessions.
5. **Polish:** consistent empty/loading/error states; modern dark UI with coherent density; keyboard Esc closes overlays.
6. **Docs:** WEB_API and this goal describe the same product surface; `webui/AGENTS.md` matches the tree.

---

## Remaining roadmap

The inventory, route parity, responsive shell, and generic catalog bridge in
[Tracking](#tracking) are complete. Continue in this order unless a user-facing
defect takes priority:

1. Add regime/pattern context and improve backtest reliability visualization.
2. Finish keyboard, screen-reader, reduced-motion, and narrow-screen passes on
   every dedicated panel.
3. Add a frontend lint command, critical component tests, and one optional
   browser smoke against a mocked or local API.
4. Audit query cancellation, long-session memory, and code-splitting; record a
   small performance budget.
5. Add further dedicated research modules only as vertical slices: API
   contract → client types → UI → tests → docs.

---

## Working agreements

1. **Transport purity:** UI talks only to Web API (`/api/v1`). No special-casing domain logic in React; adapt and present.
2. **Compact payloads:** Prefer UI-oriented compact responses; request `detail=full` only when the UI shows those diagnostics.
3. **Safety:** Generic mutations keep the Tools runner confirmation gate,
   preview defaults, and environment guardrails. Do not add a dedicated order
   workflow without a separate safety-reviewed design.
4. **Token hygiene:** Auth token stays in memory only (current behavior is correct—do not “improve” it into localStorage).
5. **Small PRs:** Prefer vertical slices (one capability usable end-to-end) over giant refactors.
6. **Commit style:** `webui: <imperative summary>` (or `docs:` when only documentation).

---

## Out of scope (for this goal)

- Replacing CLI/MCP as the full automation surface
- Shipping a light-theme redesign as a priority (dark terminal is the product look)
- Mobile-native apps / offline-first PWA (nice-to-have later, not goal-blocking)
- Full visual parity with TradingView or commercial terminals
- Auto-trading dashboards

---

## Tracking

Use this document as the north star. Practical tracking options:

- Keep the **coverage matrix** updated in PRs that touch API or UI surfaces.
- When a phase completes, mark it here with date and short notes.
- Link major implementation PRs under each phase.

| Phase | Status | Notes |
|---|---|---|
| 0 Inventory | Done | Coverage matrix in `docs/WEBUI_API_COVERAGE.md`; `webui/AGENTS.md` refreshed |
| 1 Parity & reliability | Done | `/models`, health/ready chip, pivot method + S/R params, partial-failure banners |
| 2 Responsive & modern shell | Done | Breakpoint helpers; toolbar More overflow; forecast bottom sheet / drawer; Esc closes panels/modals |
| 3 Chart research depth | Partial | On-chart EMA/RSI/MACD/volume, confluence, volume profile, idea TP/SL, read-only exposure (2026-08-18). Regime tint still later. |
| 4 Backend feature bridge | Done | Full MCP catalog via `GET/POST /api/v1/tools*`; SPA Tools runner; watchlist radar + session strip (2026-08-18) |
| 5 Hardening & speed | Partial | Pure unit tests + typecheck/build; no e2e CI |

---

## One-line summary

**Ship a modern, fast, responsive chart research console that fully exercises the Web API, closes frontend engineering gaps, and grows in lockstep with the backend’s highest-value research capabilities—without compromising safety or transport boundaries.**
