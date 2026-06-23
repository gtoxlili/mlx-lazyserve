# mlx-lazyserve

An **Ollama-style, lazy-loading [MLX](https://github.com/ml-explore/mlx) inference server** for Apple Silicon, exposing an **OpenAI-compatible** API. Models load on the first request and unload after an idle timeout, so the service is always reachable but only holds a model in unified memory while it's actually in use.

Built for a headless **Mac mini (M4 Pro, 24 GB)** reached over **Tailscale**.

## Why not just Ollama?

On Apple Silicon, MLX is meaningfully faster than the llama.cpp/Metal path (≈2–3× decode on MoE models). Ollama 0.19+ can use MLX too — **but only on machines with ≥ 32 GB unified memory**. This box has 24 GB, so Ollama falls back to the slower llama.cpp path. Going straight to MLX is the only way to get the speedup here.

This server keeps Ollama's best ergonomic — **lazy load + idle unload** — so RAM is free when the model is idle.

## Features

- OpenAI-compatible `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`, `GET /health`
- **Tool calling** (`tools`/`tool_choice` → `tool_calls` + `finish_reason:"tool_calls"`) and **reasoning** (`enable_thinking` → a separate `reasoning_content`), both streaming + non-streaming; parsing reuses each model's mlx-lm/mlx-vlm `tool_parsers`
- Lazy model load on first use; single-slot (one model resident at a time — fits 24 GB)
- Idle unload after `IDLE_TIMEOUT` to release unified memory
- Auto engine: tries `mlx-lm`, falls back to `mlx-vlm` for vision-language architectures
- Optional bearer-token auth; otherwise rely on Tailscale for access control
- Ships a `launchd` LaunchAgent for 24/7 operation
- Maintenance mode: pause the service (free memory + reject requests) during scheduled heavy jobs

## Models

Configured in [`models.toml`](models.toml). Weights download lazily into `~/.cache/huggingface`.

| name | repo | size | role |
|---|---|---|---|
| `gemma4-26b-uncensored` *(default)* | `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` | ~14.2 GB | **main** — uncensored SuperGemma4 26B-A4B MoE; comfortable on 24 GB |
| `qwen3.6-35b-a3b` | `TheCluster/Qwen3.6-35B-A3B-Heretic-MLX-mixed-3.9bit` | ~18.7 GB | **main** — Heretic-abliterated 35B-A3B MoE; tight, raise wired limit (see below) |
| `qwen3.5-9b` | `TheCluster/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp4` | ~5 GB | fallback — light & fast, always loads |

All three are uncensored MLX builds. The two mains are MoE (~3–3.8B active) so they decode fast despite their size.

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
| `MLX_LAZYSERVE_WIRED_LIMIT_MB` | `0` | if > 0, set Metal wired limit on start, reset on stop |
| `MLX_LAZYSERVE_API_KEYS` | *(empty)* | comma-separated bearer tokens; empty = no auth |
| `MLX_LAZYSERVE_MODELS` | `./models.toml` | path to the model registry |
| `MLX_LAZYSERVE_PAUSE_FILE` | `./.maintenance` | maintenance-marker path (present = start paused) |

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
