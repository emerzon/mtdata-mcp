# Publish mtdata to PyPI and the Official MCP Registry

**Audience:** Contributor

This page is for the package owner (GitHub user **emerzon**) when cutting a
public release. It is not an install guide. Day-to-day setup stays in
[SETUP.md](SETUP.md).

Do **not** publish from CI or an automated agent unless Emerson has explicitly
started that release. Agents working this repo must not run `twine upload` or
`mcp-publisher publish`.

**Related:** [Setup](SETUP.md) · [AI assistant](MCP.md) · [Naming](../README.md#naming) · [Env vars](ENV_VARS.md) (Operator)

---

## What you are publishing

| Artifact | Name | Notes |
|----------|------|-------|
| PyPI package | `mtdata-mcp` | Import stays `mtdata`. Version **0.1.0** in `pyproject.toml`. |
| MCP Registry server | `io.github.emerzon/mtdata-mcp` | Metadata only. Authenticate as GitHub user **emerzon**. |
| Human-facing stdio script | `mtdata-stdio` | What [MCP.md](MCP.md) tells IDE configs to run. |
| Registry launch alias | `mtdata-mcp` | Same `main_stdio` entry so `uvx mtdata-mcp` works. |

The Official MCP Registry hosts **metadata**, not the wheel. Publish to PyPI
first so the registry can verify the `mcp-name` ownership marker in the
package README.

Registry metadata is platform-agnostic. Runtime is not: the package depends
on **MetaTrader5**, which needs **Windows** and a **running MT5 terminal**.
macOS and Linux users run mtdata on a Windows machine or VM.

---

## 1. Confirm the ownership marker

The PyPI long description **must** contain this HTML comment (already near
the top of [README.md](../README.md)):

```html
<!-- mcp-name: io.github.emerzon/mtdata-mcp -->
```

Keep it when you edit the README. The registry fetches the published
description; the name after `mcp-name:` **must** match `server.json` `name`
exactly (`io.github.emerzon/mtdata-mcp`).

---

## 2. Build and upload the PyPI package

From the repository root, with credentials for the `mtdata-mcp` project on
PyPI:

```bash
python -m build
twine upload dist/*
```

Upload only after the README marker is present. PyPI forbids replacing an
already-published version, so do not re-upload `0.1.0` once it is live.

PyPI also rejects distributions whose metadata contains a Direct URL
dependency (`pkg @ git+https://...`), even when it lives in an extra.
`news-ycnbc` must depend on the PyPI `ycnbc` package, not a Git pin.
Inspect the built wheel before upload:

```bash
python -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); print(z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode())" dist/mtdata_mcp-0.1.0-py3-none-any.whl
```

Confirm every `Requires-Dist` line is a package-index requirement.

---

## 3. Install mcp-publisher and publish `server.json`

[`server.json`](../server.json) at the repo root describes the server for
[registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io).

1. Install [`mcp-publisher`](https://github.com/modelcontextprotocol/registry)
   (latest release binary, or `brew install mcp-publisher`).
2. Authenticate as **emerzon** (GitHub device flow):

   ```bash
   mcp-publisher login github
   ```

3. From the repository root (reads this `server.json`):

   ```bash
   mcp-publisher publish
   ```

GitHub auth only allows the namespace prefix `io.github.emerzon/`. The server
name is **`io.github.emerzon/mtdata-mcp`**.

---

## Schema notes (`mtdata-stdio`)

`server.json` follows
[`2025-12-11/server.schema.json`](https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json)
— the current Official Registry pin.

Adjustments from a naive copy of older drafts:

- **Description** is at most 100 characters (schema `maxLength`).
- **stdio transport** is only `{ "type": "stdio" }`. There is no command or
  binary field on the transport.
- Official 2026 PyPI examples set `runtimeHint` to `uvx` (not `python`).
  Clients compose `runtimeHint` + `identifier` + `packageArguments` →
  `uvx mtdata-mcp`. `packageArguments` are extra args *after* that token, so
  they cannot rename the binary to `mtdata-stdio`. Empty `packageArguments`
  are omitted.
- Because the human-facing console script is **`mtdata-stdio`**,
  `pyproject.toml` also declares a **`mtdata-mcp`** alias on
  `mtdata.core.server:main_stdio`. That is what makes `uvx mtdata-mcp` start
  the stdio server. Manual client configs should still use
  `"command": "mtdata-stdio"` as in [MCP.md](MCP.md).

---

## Version

Keep **0.1.0** aligned with `pyproject.toml` until you intentionally cut a
new release. Then bump `pyproject.toml` and both `version` fields in
`server.json` together.
