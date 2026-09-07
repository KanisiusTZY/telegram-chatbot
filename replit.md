# Telegram AI Agent (Userbot)

A sarcastic, lazy-but-functional AI agent running as a Telegram userbot. Powered by Google Gemini (`gemini-3.6-flash` with native multimodal vision and Google Search grounding). Responds to private messages, can use tools, remembers notes and reminders across restarts.

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
| `GEMINI_API_KEY` | Google Gemini API key (or `GOOGLE_API_KEY`) |
| `SESSION_STRING` | (optional) Telethon StringSession — if absent, falls back to `session` file |
| `SESSION_SECRET` | Express API server session secret |
| `DATABASE_URL` | Postgres connection string (API server only) |

Web search uses native Gemini Google Search Grounding.

## Stack

### Python bot (`main.py`, `agent_db.py`, `agent_tools.py`)
- **Runtime**: Python 3.11 / 3.14
- **Telegram**: Telethon (userbot / MTProto — not Bot API)
- **LLM**: Google GenAI SDK (`google-genai`) — `gemini-3.6-flash` (multimodal vision + text + function calling)
- **Search**: Built-in Google Search Grounding (`types.GoogleSearch()`)
- **Tools**: Reminders, notes, file conversion, media download, background removal, music search, iPhone mockup screenshot, photo upscale HD
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
- **Native Multimodal Vision**: Gemini directly accepts multimodal inputs (images and documents) natively alongside tool calling, eliminating the need for a separate two-step vision pipeline.
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
- **Native Google Search Grounding**: Real-time web facts, news, and queries are grounded directly through Google Search, eliminating unofficial search scrapers and rate-limit throttling.
- **APScheduler logs**: suppressed at INFO level to avoid noise — set `apscheduler.executors.default` logger to INFO if you need to debug scheduler internals.


## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
