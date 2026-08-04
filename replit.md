# Telegram AI Agent (Userbot)

A sarcastic, lazy-but-functional AI agent running as a Telegram userbot. Powered by Groq (LLaMA 3.3 for text, LLaMA 4 Scout for vision). Responds to private messages, can use tools, remembers notes and reminders across restarts.

## Run & Operate

- `python main.py` — start the bot (configured as the default run command)
- `python generate_session.py` — one-time helper to generate a `SESSION_STRING` for headless auth
- `pnpm --filter @workspace/api-server run dev` — run the separate Express API server (port 5000)
- `pnpm run typecheck` — full typecheck across all TS packages
- `pnpm --filter @workspace/db run push` — push DB schema changes to Postgres (API server only)

### Required secrets (Replit Secrets)

| Secret | Purpose |
|---|---|
| `TELEGRAM_API_ID` | Telegram app ID (from my.telegram.org) |
| `TELEGRAM_API_HASH` | Telegram app hash |
| `GROQ_API_KEY` | Groq API key for LLaMA models |
| `SESSION_STRING` | (optional) Telethon StringSession — if absent, falls back to `session` file |
| `SESSION_SECRET` | Express API server session secret |
| `DATABASE_URL` | Postgres connection string (API server only) |

No web-search API key needed — uses DuckDuckGo (free, no auth).

## Stack

### Python bot (`main.py`, `agent_db.py`, `agent_tools.py`)
- **Runtime**: Python 3.11, uv
- **Telegram**: Telethon (userbot / MTProto — not Bot API)
- **LLM**: Groq SDK — `llama-3.3-70b-versatile` (text + tools), `meta-llama/llama-4-scout-17b-16e-instruct` (vision)
- **Tools**: DuckDuckGo search (`duckduckgo-search`), safe math (`simpleeval`)
- **DB**: SQLite via `sqlite3` stdlib — file `agent.db`
- **Scheduler**: APScheduler 3.x `BackgroundScheduler` (reminder dispatch)
- **Keep-alive**: Flask on port 8099

### Node.js API server (`artifacts/api-server/`)
- **Runtime**: Node.js 24, pnpm workspaces, TypeScript 5.9
- **Framework**: Express 5
- **DB**: PostgreSQL + Drizzle ORM
- **Validation**: Zod v4, drizzle-zod
- **Build**: esbuild (CJS bundle)

## Where things live

```
main.py            — bot entrypoint: Telegram handlers, agent loop, Flask, scheduler
agent_db.py        — SQLite layer: messages, notes, reminders
agent_tools.py     — tool schemas (Groq format) + implementations
agent.db           — SQLite database (auto-created on first run, gitignored)
generate_session.py — one-shot Telethon session string generator
artifacts/api-server/src/  — Express API source
lib/db/            — Drizzle schema (shared between API packages)
lib/api-spec/      — OpenAPI spec (source of truth for API contracts)
```

## Architecture decisions

- **SQLite over in-memory dict**: conversation history, notes, and reminders must survive restarts. SQLite is zero-config and has no external dependency. WAL mode enables concurrent reads without blocking writes.
- **Two-step vision pipeline**: Groq's vision model doesn't support tool calling. Solution: vision model extracts a plain-text description of the image first, then that description is injected into the text agent loop which _does_ support all tools. This lets the agent calculate totals from a receipt photo, search based on image content, etc.
- **APScheduler over raw thread loop**: cleaner lifecycle (`.start()` / `.shutdown()`), handles missed jobs on restart with `coalesce=True`, and separates scheduling concerns from the main event loop.
- **Agent loop capped at 5 iterations**: prevents infinite tool-call chains if the model gets stuck. After 5 rounds without a final text answer, the bot replies with an explicit fallback message.
- **Telethon userbot (not Bot API)**: intentional — gives access to user-level features (view-once media interception, outgoing command triggers). Trade-off: uses your personal account, not a bot account.
- **Rate limiter in-memory**: 20 messages/minute per user via sliding-window deque. Resets on restart — acceptable for a personal-use bot. Move to Redis/DB if multi-instance deployment is needed.

## Outgoing commands (type from any private chat)

| Command | What it does |
|---|---|
| `/help` | Show all capabilities and commands |
| `/notes` | List all saved notes for that chat's user |
| `/remind +30m Minum obat` | Set a reminder (relative: +Nm/+Nh/+Nd) |
| `/remind 2026-08-05T09:00 Meeting` | Set a reminder (absolute ISO UTC) |
| `/clear` | Reset conversation history for that chat |

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- **`SESSION_STRING` vs session file**: `SESSION_STRING` env var takes priority. If neither is set and `session` file doesn't exist, the bot will prompt for a phone number interactively on first run — use `generate_session.py` to create a string session first.
- **Reminder times are UTC**: APScheduler and `agent.db` use UTC throughout. Remind users to account for timezone offset when setting absolute times.
- **DuckDuckGo rate limits**: `duckduckgo-search` is unofficial and may occasionally return empty results or get throttled. The tool gracefully returns an error JSON that the model can explain to the user.
- **Vision model has no tool access**: by design — tool calling doesn't work on multimodal inputs in Groq. The two-step pipeline (vision describe → text agent) is the workaround.
- **`calculate` uses `simpleeval`**: safe evaluator with no `exec`/`eval`. Math functions are exposed via `math.*` namespace (e.g. `math.sqrt(144)` not `sqrt(144)`).
- **APScheduler logs**: suppressed at INFO level to avoid noise — set `apscheduler.executors.default` logger to INFO if you need to debug scheduler internals.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
