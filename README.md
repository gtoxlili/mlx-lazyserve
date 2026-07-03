# mlx-lazyserve

An **Ollama-style, lazy-loading [MLX](https://github.com/ml-explore/mlx) inference server** for Apple Silicon, exposing an **OpenAI-compatible** API. Models load on the first request and unload after an idle timeout, so the service is always reachable but only holds a model in unified memory while it's actually in use.

Built for a headless **Mac mini (M4 Pro, 24 GB)** reached over **Tailscale**.

## Why not just Ollama?

On Apple Silicon, MLX is meaningfully faster than the llama.cpp/Metal path (≈2–3× decode on MoE models). Ollama 0.19+ can use MLX too — **but only on machines with ≥ 32 GB unified memory**. This box has 24 GB, so Ollama falls back to the slower llama.cpp path. Going straight to MLX is the only way to get the speedup here.

This server keeps Ollama's best ergonomic — **lazy load + idle unload** — so RAM is free when the model is idle.

## Features

- OpenAI-compatible `POST /v1/chat/completions` (streaming + non-streaming) with real token `usage` (and `stream_options.include_usage`), `GET /v1/models`, `GET /health`
- **Tool calling** (`tools`/`tool_choice` → `tool_calls` + `finish_reason:"tool_calls"`) and **reasoning** (`enable_thinking` → a separate `reasoning_content`), both streaming + non-streaming; parsing reuses each model's mlx-lm/mlx-vlm `tool_parsers`
- **Structured output** (`response_format` json_object / json_schema) via `llguidance` constrained decoding — *guaranteed* valid JSON, both engines
- Full OpenAI sampling/control: `top_k`, `min_p`, `seed`, `repetition`/`presence`/`frequency_penalty`, `logit_bias`, `stop`, plus optional quantized KV cache (`kv_bits`)
- Lazy model load on first use; single-slot (one model resident at a time — fits 24 GB)
- Idle unload after `IDLE_TIMEOUT` to release unified memory
- Auto engine: tries `mlx-lm`, falls back to `mlx-vlm` for vision-language architectures
- Optional bearer-token auth; otherwise rely on Tailscale for access control
- Ships a `launchd` LaunchAgent for 24/7 operation
- Maintenance mode: pause the service (free memory + reject requests) during scheduled heavy jobs
- **Embedded Telegram bot** (optional): add it to a group and it answers @mentions / replies-to-it with rich formatting (Markdown→entities, expandable blockquotes, 👀 reactions, typing). Reuses the loaded model in-process; **interrupts + merges** if a user sends a follow-up mid-answer

## Models

Configured in [`models.toml`](models.toml). Weights download lazily into `~/.cache/huggingface`.

| name | repo | size | role |
|---|---|---|---|
| `gemma4-26b-uncensored` | `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` | ~14.2 GB | **main** — uncensored SuperGemma4 26B-A4B MoE; comfortable on 24 GB |
| `qwen3.6-35b-a3b` | `TheCluster/Qwen3.6-35B-A3B-Heretic-MLX-mixed-3.9bit` | ~18.7 GB | **main** — Heretic-abliterated 35B-A3B MoE; tight, raise wired limit (see below) |
| `qwen3.5-9b` *(default)* | `TheCluster/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp4` | ~5 GB | fallback — light & fast, always loads |
| `qwythos-9b` | `sahilchachra/Qwythos-9B-Claude-Mythos-5-1M-mxfp4-mlx` | ~4.8 GB | Qwythos/Claude Mythos reasoning tune; text-only MLX 4-bit |

All four are uncensored MLX builds. The two mains are MoE (~3–3.8B active) so they decode fast despite their size.

> The exact HauhauCS Gemma is GGUF-only (no MLX), so the Gemma slots use the closest MLX equivalents.

## Requirements

- Apple Silicon Mac, macOS
- [`uv`](https://docs.astral.sh/uv/)

## Run

```bash
uv sync                  # core deps (fastapi, uvicorn, mlx-lm)
uv sync --extra vision   # add mlx-vlm (needed for Gemma 4 / vision-language models)
uv run mlx-lazyserve
```

The first request for a model is slow (download + load); subsequent ones are fast until it idles out.

## Downloads from mainland China

Most of these repos are **xet-backed**, and the default Xet transfer is slow from China (~0.5 MB/s). `hf-mirror.com` doesn't help for them — for xet repos it just 308-redirects to Hugging Face's xet CDN. The fix is to **disable xet** and use the classic CDN path (~14 MB/s, with auto-resume):

```bash
export HF_HUB_DISABLE_XET=1
```

This is already set in the LaunchAgent plist, so the running service downloads models this way too. If a drop interrupts a large download, just re-run — `hf download` resumes from the cache.

## Configuration (env vars)

| var | default | meaning |
|---|---|---|
| `MLX_LAZYSERVE_HOST` | `127.0.0.1` | bind address (`0.0.0.0` or your Tailscale IP to expose) |
| `MLX_LAZYSERVE_PORT` | `41434` | port (high, avoids dev-server clashes) |
| `MLX_LAZYSERVE_IDLE_TIMEOUT` | `600` | seconds idle before unloading (`0` = never) |
| `MLX_LAZYSERVE_MAX_TOKENS` | `8192` | default max output tokens (headroom for reasoning models; per-request `max_tokens` overrides) |
| `MLX_LAZYSERVE_ENABLE_THINKING` | `false` | default thinking/reasoning state; per-request `enable_thinking` overrides |
| `MLX_LAZYSERVE_KV_BITS` | `0` | if > 0 (e.g. `8`), quantize the KV cache → less memory / longer context, slight quality cost; per-request `kv_bits` overrides. Auto-falls-back to unquantized on sliding-window models (Gemma) that can't quantize a `RotatingKVCache` |
| `MLX_LAZYSERVE_REPETITION_PENALTY` | `1.1` | default repetition penalty (Ollama-style) applied when a request omits one — curbs degenerate loops; per-request `repetition_penalty` overrides (`1.0` disables) |
| `MLX_LAZYSERVE_REPETITION_CONTEXT` | `64` | tokens the repetition penalty looks back over |
| `MLX_LAZYSERVE_MIN_P` | `0.0` | default min-p sampling floor (`0` = off); per-request `min_p` overrides |
| `MLX_LAZYSERVE_LOOP_GUARD` | `true` | cut generation if the output degenerates into a repeat loop; per-request `loop_guard` overrides (auto-off for `response_format` JSON) |
| `MLX_LAZYSERVE_WIRED_LIMIT_MB` | `0` | if > 0, set Metal wired limit on start, reset on stop |
| `MLX_LAZYSERVE_API_KEYS` | *(empty)* | comma-separated bearer tokens; empty = no auth |
| `MLX_LAZYSERVE_MODELS` | `./models.toml` | path to the model registry |
| `MLX_LAZYSERVE_PAUSE_FILE` | `./.maintenance` | maintenance-marker path (present = start paused) |
| `MLX_LAZYSERVE_TG_BOT_TOKEN` | *(empty)* | BotFather token; empty = Telegram bot off (see [Telegram bot](#telegram-bot)) |
| `MLX_LAZYSERVE_TG_MODEL` | *(default model)* | **default** model the bot chats with (each user overrides via `/model`) |
| `MLX_LAZYSERVE_TG_SYSTEM_PROMPT` | *(built-in)* | system persona for the bot |
| `MLX_LAZYSERVE_TG_MAX_TOKENS` | `MAX_TOKENS` | max output tokens per bot reply |
| `MLX_LAZYSERVE_TG_KV_BITS` | `4` | bot KV-cache quant bits (4 = fast/light, 8 = higher quality, 0 = off). Bump to `8` for steadier output if you don't switch to the 35B model |
| `MLX_LAZYSERVE_TG_REPETITION_PENALTY` | `REPETITION_PENALTY` | repetition penalty for bot replies (these uncensored/abliterated builds loop easily) |
| `MLX_LAZYSERVE_TG_MIN_P` | `0.05` | min-p sampling floor for bot replies |
| `MLX_LAZYSERVE_TG_HISTORY_TURNS` | `8` | per-(group,user) turns kept as context |
| `MLX_LAZYSERVE_TG_ENABLE_THINKING` | `false` | **default** for reasoning (each user overrides via `/think`); shown as an expandable blockquote |
| `MLX_LAZYSERVE_TG_DB_PATH` | `./telegram-history.db` | SQLite file for per-(group,user) history (persists across restarts) |
| `MLX_LAZYSERVE_TG_OWNER_IDS` | *(empty)* | owner user ids: only they may add the bot to a group (others → it leaves) and they approve stranger DMs; empty = no restriction. See [Access control](#access-control) |

## API

```bash
curl http://<tailscale-ip>:41434/v1/chat/completions \
  -H 'Authorization: Bearer <key>' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Point any OpenAI SDK at `http://<tailscale-ip>:41434/v1` and set the model to one of the names above.

### Tool calling

Pass OpenAI-style `tools` (and optionally `tool_choice`). When the model calls a tool the
reply carries `tool_calls` with `finish_reason: "tool_calls"` (and `content: null`):

```bash
curl http://<tailscale-ip>:41434/v1/chat/completions \
  -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"weather in Boston?"}],
       "tools":[{"type":"function","function":{"name":"get_current_weather",
         "description":"Get the current weather in a location",
         "parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]}'
# -> choices[0].message.tool_calls[0].function = {"name":"get_current_weather","arguments":"{\"location\":\"Boston, MA\"}"}
```

`tool_choice:"none"` disables calling (the tools still inform the model). Parsing reuses each
model's native `tool_parsers`, so the model's own wire format (Qwen XML, Gemma, …) is
normalized to OpenAI `tool_calls` — streamed as `delta.tool_calls`, or whole when non-streaming.

### Thinking / reasoning

Thinking is **off by default** (clean answers). Enable it per request with `enable_thinking`
(or `chat_template_kwargs.enable_thinking`); the thinking text returns in a separate
`reasoning_content` field (streamed as `delta.reasoning_content`), never mixed into `content`:

```bash
curl http://<tailscale-ip>:41434/v1/chat/completions \
  -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"is 91 prime?"}],"enable_thinking":true}'
```

Flip the server-wide default with `MLX_LAZYSERVE_ENABLE_THINKING=true`.

### Structured output (`response_format`)

Get **guaranteed** valid JSON via constrained decoding (an `llguidance` grammar masks every
token that would break the schema — not best-effort prompting). Works on both engines:

```bash
curl http://<tailscale-ip>:41434/v1/chat/completions \
  -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-26b-uncensored","messages":[{"role":"user","content":"Invent a person."}],
       "response_format":{"type":"json_schema","json_schema":{"name":"person","schema":
         {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},
          "required":["name","age"],"additionalProperties":false}}}}'
# -> {"name":"Elias Thorne","age":42}
```

`{"type":"json_object"}` constrains to any valid JSON object; `{"type":"json_schema",…}`
constrains to your schema (use `additionalProperties:false` to forbid extra keys). Structured
output forces thinking off and can't be combined with `tools` (returns 400).

### Sampling & control

Standard OpenAI knobs are honored: `temperature`, `top_p`, `top_k`, `min_p`, `seed`
(reproducible), `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`,
`stop` (string or list — truncates and halts), `max_tokens` / `max_completion_tokens`.

**Anti-repetition (on by default).** These uncensored/abliterated, heavily-quantized builds
are prone to **degeneration loops** (the model repeats one short phrase forever). Two defaults
guard against it, both overridable per request:

- A default **`repetition_penalty` of `1.1`** over a 64-token window (Ollama's defaults), applied
  whenever a request omits one. Send `"repetition_penalty": 1.0` to disable, or tune the floor
  with `MLX_LAZYSERVE_REPETITION_PENALTY` / `MLX_LAZYSERVE_REPETITION_CONTEXT`.
- A model-agnostic **`loop_guard`** that halts generation if the decoded output still falls into a
  short repeating cycle (so a loop can't run to `max_tokens`). It only trips on pathological runs
  (a ≤32-char block repeated past ~60 chars), so normal repetition is untouched; it's auto-disabled
  for `response_format` JSON. Turn it off with `"loop_guard": false` or `MLX_LAZYSERVE_LOOP_GUARD=false`.

The embedded Telegram bot enables both too (`MLX_LAZYSERVE_TG_REPETITION_PENALTY`,
`MLX_LAZYSERVE_TG_MIN_P`) and additionally **collapses** any leftover repetition before it's shown
or saved — so a looped reply can't poison the next turn's history.

## Telegram bot

An optional Telegram bot is **embedded in the same process** — no second service. In groups it
replies to any message that **@mentions it** or **replies to one of its messages** (ignoring
unrelated chatter and other bots). **Private chats** work too, but are gated: a stranger's first
DM asks an owner to approve before the bot will reply (see [Access control](#access-control)). It
calls the in-process model directly, so it shares the single model slot with the OpenAI API.

### Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token.
2. Set `MLX_LAZYSERVE_TG_BOT_TOKEN` (in `deploy/service.env` for the service, or your shell for
   dev) and install the extra:
   ```bash
   uv sync --extra telegram      # dev; install.sh adds this automatically for the service
   uv run mlx-lazyserve
   ```
3. Add the bot to a group and @mention it.

**Group privacy can stay ON** (BotFather's default): Telegram still delivers @mentions,
replies-to-the-bot, and commands — exactly the bot's triggers — so it never sees unrelated chatter.

### Behavior

- **Rich formatting** via recent Bot API features: the model's Markdown is converted to Telegram
  message *entities* (no MarkdownV2 escaping glitches), long quotes/reasoning collapse into
  **expandable blockquotes**, over-long answers auto-split, and big code blocks become file
  attachments. While working, the bot sets a 👀 reaction + the "typing…" action and threads its
  reply onto your message.
- **Per-user model & thinking**: each user picks their own model with **`/model`** and toggles
  reasoning with **`/think`** — inline-keyboard buttons, or `/model <name>` / `/think on|off`.
  Choices persist per (group, user). `MLX_LAZYSERVE_TG_MODEL` / `MLX_LAZYSERVE_TG_ENABLE_THINKING`
  are just the defaults. (Switching models evicts + loads on the single GPU slot, so the first
  message after a switch is slower.)
- **Per-user memory**: a short rolling history per (group, user), persisted to **SQLite** so it
  survives restarts; `/reset` clears yours. History is kept to the last
  `MLX_LAZYSERVE_TG_HISTORY_TURNS` turns, and the prompt is additionally trimmed (oldest first,
  always keeping the system prompt + newest message) to fit the model's **context window** minus
  the reserved output — so long pastes can't overflow the model. Set each model's window with
  `context` in [`models.toml`](models.toml).
- **Interrupt + merge**: if you send another message while it's still answering your previous one,
  it drops the half-finished generation and re-runs over **both** messages (passed as separate
  native chat turns) — one coherent reply, not two.
- **No streaming**: replies are sent whole.

### Access control

`MLX_LAZYSERVE_TG_OWNER_IDS` (comma-separated user ids) are the bot's owners. They gate two things:

- **Group add**: only an owner may add the bot to a group; if anyone else adds it, it posts a
  short notice and **leaves immediately**.
- **Private chat**: DMs are open but gated — a **stranger's first DM** triggers an approve/deny
  prompt sent to the owner(s); the bot only replies once approved, and the approved user id is
  **saved to SQLite** (so they stay approved across restarts). Owners are always allowed.

Empty `MLX_LAZYSERVE_TG_OWNER_IDS` = no restrictions (anyone can add it / DM it). **Note:** for
the approval DM to reach an owner, that owner must have started a chat with the bot at least once
(Telegram forbids bots from messaging users first).

Commands: **`/model`** (choose your model), **`/think`** (toggle your reasoning), **`/reset`**
(clear your context). The bot self-registers these on startup.

Tune defaults with the `MLX_LAZYSERVE_TG_*` env vars (table above). By default the bot uses the
fast `qwen3.5-9b` model with a **4-bit KV cache** for snappy chat replies (the OpenAI API path
keeps its own `MLX_LAZYSERVE_KV_BITS`).

## Run as a service (24/7)

This repo is the **source**; the running service is a separate **runtime entity** that
lives OUTSIDE `~/Downloads`. That matters: `~/Downloads` (and `~/Documents`, `~/Desktop`)
are macOS **TCC-protected** folders, and a launchd agent running the venv's python from
there is silently denied (`Operation not permitted`, no prompt). [`deploy/install.sh`](deploy/install.sh)
syncs the source into `~/.mlx-lazyserve/`, builds a venv there, and installs a LaunchAgent
that runs that runtime venv's python **directly** — no `uv` wrapper, no TCC denial, no
authorization popup, and clean signals (no orphaned process holding the port on restart).

```bash
cp deploy/service.env.example deploy/service.env   # fill in API key, host, port…
bash deploy/install.sh                             # build runtime + (re)install the service
tail -f ~/.mlx-lazyserve/logs/stderr.log           # watch it
```

Re-run `bash deploy/install.sh` after code changes to redeploy (idempotent). `service.env`
is gitignored so your API key never gets committed; `MLX_LAZYSERVE_HOME` overrides the
runtime path. The wired-memory limit needs the one-time sudoers rule (below).

A LaunchAgent runs inside your login session, so on a headless mini enable **auto-login**
(System Settings → Users & Groups) — otherwise it won't start after a reboot with nobody
logged in.

With `MLX_LAZYSERVE_API_KEYS` set, send `Authorization: Bearer <key>` on `/v1/*` and
`/admin/*` (`/health` stays open).

## Reverse proxy (nginx)

[`deploy/nginx/mlx-lazyserve.conf`](deploy/nginx/mlx-lazyserve.conf) is an nginx vhost for
exposing the service publicly: **Cloudflare (Full strict) → nginx → Tailscale →
`office:41434`**. Streaming-friendly (no buffering, 600s timeout on `/v1/chat/completions`),
forwards the `Authorization` bearer, allows 25 MB bodies for image uploads.

Drop it in nginx `conf.d/`, set `server_name` + a Cloudflare DNS record, then
`nginx -t && nginx -s reload`. Cloudflare's proxy has a ~100s first-byte timeout — prefer
streaming (`stream:true`) for long generations, or grey-cloud / Tunnel that subdomain.

## Big models on 24 GB — Metal wired-memory limit

macOS caps the GPU at ~75% of unified memory — measured **17.76 GB** on this M4 Pro
(`mx.device_info()["max_recommended_working_set_size"]`). The Qwen3.6-A3B build is
18.7 GB, so it needs the cap raised.

The service handles this automatically: set `MLX_LAZYSERVE_WIRED_LIMIT_MB` (the plist
uses `22000` ≈ 21.5 GB) and it runs `sysctl iogpu.wired_limit_mb=<n>` on **startup** and
resets it to `0` on **graceful shutdown**. That needs a one-time, passwordless sudo rule
scoped to *only* that sysctl (no password is stored anywhere):

```bash
sudo install -m 0440 -o root -g wheel launchd/mlx-lazyserve.sudoers /etc/sudoers.d/mlx-lazyserve
sudo visudo -c    # validate
```

Without the rule the service still runs (it just logs a warning and stays on the default
cap). Manual one-off: `sudo sysctl iogpu.wired_limit_mb=22000` (resets on reboot). Keep
`max_tokens` / context modest to bound KV-cache growth.

## Maintenance mode

For a scheduled heavy job (e.g. a weekend CPU task), pause the service so it gives back
GPU/RAM and politely turns requests away:

```bash
# pause: unload the model + reject inference with HTTP 503
curl -X POST http://127.0.0.1:41434/admin/maintenance \
  -H 'Authorization: Bearer <key>' \
  -H 'Content-Type: application/json' -d '{"enabled": true}'

# resume
curl -X POST http://127.0.0.1:41434/admin/maintenance \
  -H 'Authorization: Bearer <key>' \
  -H 'Content-Type: application/json' -d '{"enabled": false}'

# check
curl http://127.0.0.1:41434/admin/maintenance \
  -H 'Authorization: Bearer <key>'
```

While paused, `/v1/chat/completions` returns `503` with an OpenAI-style error
(`code: "maintenance"`). The state is persisted to the pause-marker file
(`MLX_LAZYSERVE_PAUSE_FILE`), so it survives a service restart during the window;
`/health` reports `"maintenance": true`.

## Status / caveats

- ✅ Smoke-tested with `qwen3.5-9b`: loads via mlx-lm (~2 s from cache); streaming + non-streaming chat, **tool calling**, and **thinking → `reasoning_content`** all verified, plus abort-on-disconnect and idle-unload. The two mains (SuperGemma4, Qwen3.6 Heretic) aren't load-tested yet.
- Gemma 4 is new; use a recent `mlx-vlm` (≥ 0.4.3) or you may hit `Model type gemma4 not supported`.
- Image input is wired (`image_url` content parts → mlx-vlm) but not yet exercised with a real image.
- `usage` token counts are returned as 0 (not computed yet).

## License

MIT
