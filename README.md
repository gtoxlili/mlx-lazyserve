# mlx-lazyserve

An OpenAI-compatible inference server for Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx). Like Ollama, it loads a model on the first request and unloads it after an idle timeout, so unified memory is free when nothing is running. Meant to run headless (e.g. a Mac mini reached over Tailscale).

Why MLX instead of Ollama: on Apple Silicon MLX decodes MoE models noticeably faster, and Ollama only uses MLX on machines with 32 GB or more. On a 24 GB box, going straight to MLX is the way to get that speed.

## Features

- OpenAI-compatible `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, `/health`
- Tool calling, reasoning (`enable_thinking` returns a separate `reasoning_content`), and structured output (`response_format`, guaranteed-valid JSON via constrained decoding)
- Full sampling controls (`top_k`, `min_p`, `seed`, the penalties, `logit_bias`, `stop`) and an optional quantized KV cache
- Lazy load on first use, idle unload; one model resident at a time (sized for 24 GB)
- Text and vision-language models (tries `mlx-lm`, falls back to `mlx-vlm`)
- Optional bearer-token auth, a `launchd` service for 24/7, and a maintenance mode
- Optional embedded Telegram bot (with web search + page/PDF reading via Firecrawl)

## Models

Configured in [`models.toml`](models.toml); weights download lazily into `~/.cache/huggingface`.

| name | repo | size |
|---|---|---|
| `gemma4-26b-uncensored` | `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` | ~14 GB |
| `qwen3.6-35b-a3b` | `TheCluster/Qwen3.6-35B-A3B-Heretic-MLX-mixed-3.9bit` | ~19 GB |
| `qwen3.5-9b` | `TheCluster/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp4` | ~5 GB |
| `qwythos-9b` | `sahilchachra/Qwythos-9B-Claude-Mythos-5-1M-mxfp4-mlx` | ~5 GB |
| `cpmopus-fable5-1b` (default) | local convert of `GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking` | ~1.1 GB |

Edit `models.toml` to add your own; any MLX repo on Hugging Face works.

## Requirements

An Apple Silicon Mac, macOS, and [`uv`](https://docs.astral.sh/uv/).

## Run (dev)

```bash
uv sync                  # core deps
uv sync --extra vision   # add mlx-vlm, for vision-language models
uv run mlx-lazyserve
```

The first request for a model downloads and loads it (slow); after that it stays fast until it idles out. Then point any OpenAI client at `http://<host>:41434/v1`:

```bash
curl http://localhost:41434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

If `MLX_LAZYSERVE_API_KEYS` is set, add `-H 'Authorization: Bearer <key>'`.

## Run as a service (24/7)

On macOS, `~/Downloads`, `~/Documents`, and `~/Desktop` are TCC-protected, and a launchd agent can't run a venv's python out of them (it's denied with no prompt). So [`deploy/install.sh`](deploy/install.sh) syncs this repo into `~/.mlx-lazyserve/`, builds a venv there, and installs a LaunchAgent that runs it directly.

```bash
cp deploy/service.env.example deploy/service.env   # fill in host, port, API key…
bash deploy/install.sh                             # build + (re)install the service
tail -f ~/.mlx-lazyserve/logs/stderr.log
```

Re-run `install.sh` after code changes; it's idempotent. `service.env` is gitignored, so your keys aren't committed. On a headless mini, enable auto-login so the agent starts after a reboot.

## Configuration

Everything is configured with environment variables, set in `deploy/service.env` (start from [`service.env.example`](deploy/service.env.example)). The common ones:

| var | default | meaning |
|---|---|---|
| `MLX_LAZYSERVE_HOST` | `127.0.0.1` | bind address (`0.0.0.0` to expose over Tailscale) |
| `MLX_LAZYSERVE_PORT` | `41434` | port |
| `MLX_LAZYSERVE_IDLE_TIMEOUT` | `600` | seconds idle before unloading (`0` = never) |
| `MLX_LAZYSERVE_MAX_TOKENS` | `8192` | default max output tokens |
| `MLX_LAZYSERVE_KV_BITS` | `0` | quantize the KV cache to N bits (`8` = less memory, longer context) |
| `MLX_LAZYSERVE_WIRED_LIMIT_MB` | `0` | raise the Metal wired-memory limit on start (see below) |
| `MLX_LAZYSERVE_API_KEYS` | *(empty)* | comma-separated bearer tokens; empty = no auth |

The sampling defaults and all the Telegram bot (`MLX_LAZYSERVE_TG_*`) settings are documented inline in [`service.env.example`](deploy/service.env.example).

## Big models on 24 GB

macOS caps the GPU at about 75% of unified memory (~17.8 GB on an M4 Pro). The 19 GB Qwen3.6 build needs that raised. Set `MLX_LAZYSERVE_WIRED_LIMIT_MB` (e.g. `22000`) and the service runs `sysctl iogpu.wired_limit_mb` on start and resets it on stop. That needs a one-time passwordless sudo rule scoped to just that sysctl:

```bash
sudo install -m 0440 -o root -g wheel launchd/mlx-lazyserve.sudoers /etc/sudoers.d/mlx-lazyserve
```

Without the rule the service still runs on the default cap (it just logs a warning).

## Extras

- **Reverse proxy**: [`deploy/nginx/mlx-lazyserve.conf`](deploy/nginx/mlx-lazyserve.conf) is an SSE-friendly nginx vhost (Cloudflare → nginx → Tailscale). Edit `server_name` and the upstream host for your setup.
- **Maintenance mode**: `POST /admin/maintenance {"enabled":true}` unloads the model and returns 503 for inference, for when a scheduled job needs the GPU/RAM back; `{"enabled":false}` resumes.
- **Telegram bot**: set `MLX_LAZYSERVE_TG_BOT_TOKEN` (from [@BotFather](https://t.me/BotFather)) and `uv sync --extra telegram`. It answers @mentions and replies in groups, keeps a short per-user history in SQLite, and lets each user pick a model (`/model`) and toggle reasoning (`/think`). It can also **search the web and read pages/PDFs** on demand via [Firecrawl](https://firecrawl.dev) — keyless by default (set `MLX_LAZYSERVE_FIRECRAWL_API_KEY` for higher limits, or `MLX_LAZYSERVE_TG_WEB_TOOLS=false` to turn it off). `MLX_LAZYSERVE_TG_OWNER_IDS` gates who can add it to a group or DM it. All `TG_*` options are in [`service.env.example`](deploy/service.env.example).

## Downloads from mainland China

Most of these repos are xet-backed, and xet transfer is slow from China. Set `HF_HUB_DISABLE_XET=1` to use the classic Hugging Face CDN instead (already set in the LaunchAgent). Interrupted downloads resume when you re-run.

## License

[GPL-3.0-or-later](LICENSE). Copyright (C) 2026 gtoxlili.
