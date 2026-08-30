# Use mtdata from an AI assistant

**Audience:** User

An AI assistant can call mtdata the same way you would type commands — fetch
candles, run a forecast, read news — while you stay in the chat.

**MCP** (Model Context Protocol) is just the plug that lets the assistant
discover and call those tools. You do not need to learn the protocol.

**Dense terms:** [MCP](GLOSSARY.md#mcp-model-context-protocol) · [Dry-run](GLOSSARY.md#dry-run) · [Trade guardrails](GLOSSARY.md#trade-guardrails)

**Related:** [Setup](SETUP.md) · [Trading safety](TRADING_SAFETY.md) · [CLI](CLI.md) · [Run as a service](DEPLOYMENT.md) (Operator)

---

## Safety first

The assistant sees the **full** tool list, including `trade_place`,
`trade_modify`, and `trade_close`. Those talk to the MetaTrader 5 account that
is logged in right now.

- Use a **demo** account until you trust the setup.
- Trading tools default to **preview** (dry-run). A request reaches MT5 only
  when you explicitly pass `dry_run=false`.
- Optional account caps (allowed symbols, max size, max risk %) are in
  [ENV_VARS.md](ENV_VARS.md#trade-guardrails).

A good first prompt:

> Use mtdata to fetch the last 50 H1 candles for EURUSD and a Theta forecast
> for the next 12 hours. Do not place or modify any orders.

---

## Which entry point?

| You want… | Run | Typical client |
|-----------|-----|----------------|
| The assistant **starts** mtdata itself (IDE / desktop app) | `mtdata-stdio` | Claude Desktop, Cursor, VS Code |
| A long-lived server the client connects to over HTTP | `mtdata-sse` (default) or `mtdata-streamable-http` | Remote or browser MCP clients |

`mtdata-stdio` should **not** be installed as a Windows service. The client
spawns it. For a background HTTP server, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Claude Desktop (stdio)

After installing this repo (PyPI package **`mtdata-mcp`**, CLI **`mtdata-cli`**),
add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mtdata": {
      "command": "mtdata-stdio"
    }
  }
}
```

Restart Claude Desktop. You should see mtdata tools in the tool list. Keep
MetaTrader 5 running in the same Windows session.

---

## Other MCP clients

Point the client at `mtdata-stdio` the same way, or start `mtdata-sse` and use
the host/port from your `.env` (`FASTMCP_HOST`, `FASTMCP_PORT`). Bind to
`127.0.0.1` unless you have set an auth token and understand
[remote bind rules](ENV_VARS.md#mcp-server).

---

## What to try first (read-only)

Ask the assistant to:

1. `symbols_list` — can MetaTrader 5 see markets?
2. `data_fetch_candles` on one symbol.
3. `forecast_generate` with method `theta`.
4. `news` if you want headlines next to the chart.

Stay off `trade_*` until you have read [TRADING_SAFETY.md](TRADING_SAFETY.md).

---

## Deeper detail

- Tool names and flags match the [CLI](CLI.md).
- Compact results omit healthy telemetry and surface non-nominal conditions
  once in `warnings`. Use `detail=full` for the consolidated `meta` envelope,
  or `output_fields` for one normally-full path without restoring everything.
- Long-running training belongs in an interactive shell, MCP, or the Web API —
  not a one-shot `mtdata-cli` process. See [FORECAST.md](FORECAST.md).
- Output shape: [OUTPUT.md](OUTPUT.md).
