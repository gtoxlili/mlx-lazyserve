# mlx-lazyserve

An **Ollama-style, lazy-loading [MLX](https://github.com/ml-explore/mlx) inference server** for Apple Silicon, exposing an **OpenAI-compatible** API. Models load on the first request and unload after an idle timeout, so the service is always reachable but only holds a model in unified memory while it's actually in use.

Built for a headless **Mac mini (M4 Pro, 24 GB)** reached over **Tailscale**.

## Why not just Ollama?

On Apple Silicon, MLX is meaningfully faster than the llama.cpp/Metal path (≈2–3× decode on MoE models). Ollama 0.19+ can use MLX too — **but only on machines with ≥ 32 GB unified memory**. This box has 24 GB, so Ollama falls back to the slower llama.cpp path. Going straight to MLX is the only way to get the speedup here.

This server keeps Ollama's best ergonomic — **lazy load + idle unload** — so RAM is free when the model is idle.

## Features

- OpenAI-compatible `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`, `GET /health`
- Lazy model load on first use; single-slot (one model resident at a time — fits 24 GB)
- Idle unload after `IDLE_TIMEOUT` to release unified memory
- Auto engine: tries `mlx-lm`, falls back to `mlx-vlm` for vision-language architectures
- Optional bearer-token auth; otherwise rely on Tailscale for access control
- Ships a `launchd` LaunchAgent for 24/7 operation

## Models

Configured in [`models.toml`](models.toml). Weights download lazily into `~/.cache/huggingface`.

| name | repo | size (4-bit) | notes |
|---|---|---|---|
| `qwen3.5-9b` *(default)* | `TheCluster/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp4` | ~5 GB | exact HauhauCS model, community MLX conversion |
| `gemma4-12b-qat` | `mlx-community/gemma-4-12B-it-qat-4bit` | ~7 GB | stock Google QAT (not abliterated) |
| `gemma4-26b-uncensored` | `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` | ~15 GB | uncensored 26B-A4B MoE, different lineage than HauhauCS |
| `qwen3.6-35b-a3b` | `TheCluster/Qwen3.6-35B-A3B-Heretic-MLX-mixed-3.9bit` | ~18.7 GB | uncensored 35B-A3B MoE; biggest pick that fits 24 GB (raise wired limit, see below) |

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

## Configuration (env vars)

| var | default | meaning |
|---|---|---|
| `MLX_LAZYSERVE_HOST` | `127.0.0.1` | bind address (`0.0.0.0` or your Tailscale IP to expose) |
| `MLX_LAZYSERVE_PORT` | `11435` | port |
| `MLX_LAZYSERVE_IDLE_TIMEOUT` | `600` | seconds idle before unloading (`0` = never) |
| `MLX_LAZYSERVE_MAX_TOKENS` | `2048` | default max output tokens |
| `MLX_LAZYSERVE_API_KEYS` | *(empty)* | comma-separated bearer tokens; empty = no auth |
| `MLX_LAZYSERVE_MODELS` | `./models.toml` | path to the model registry |

## API

```bash
curl http://<tailscale-ip>:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Point any OpenAI SDK at `http://<tailscale-ip>:11435/v1` and set the model to one of the names above.

## Run as a service (24/7)

```bash
cp launchd/dev.influo.mlx-lazyserve.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/dev.influo.mlx-lazyserve.plist
```

A LaunchAgent runs inside your login session, so on a headless mini enable **auto-login** (System Settings → Users & Groups) — otherwise it won't start after a reboot with nobody logged in. Logs go to `logs/`.

## Big models on 24 GB

For the 26B/MoE model (~15 GB) leave headroom, or raise the Metal wired limit on this dedicated box:

```bash
sudo sysctl iogpu.wired_limit_mb=22000   # not persistent across reboots
```

Keep `max_tokens` / context modest to bound KV-cache growth.

## Status / caveats

- ⚠️ Scaffold committed; **not yet smoke-tested against live weights**. The `mlx-vlm` text path targets mlx-vlm ≥ 0.4.3 and is verified on first run.
- Gemma 4 is new; use a recent `mlx-vlm` (≥ 0.4.3) or you may hit `Model type gemma4 not supported`.
- Vision/image input isn't wired into the API yet (text-only); the engine supports it and it's a small addition.

## License

MIT
