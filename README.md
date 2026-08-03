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
- Optional video generation (MiniMax-H3): `/v1/videos`, video with synchronized stereo audio

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

## Video generation (MiniMax-H3)

Optional, off unless configured. Generates video with a synchronized stereo soundtrack from a prompt, optionally anchored to a first and/or last frame.

This one does not run in-process. The model is [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), a 33B video DiT, and the MLX implementation lives in [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) — a native Zig server that talks to MLX's C API. mlx-lazyserve spawns it as a child process, owns the queue and the job lifecycle, and makes the mp4. So the split is: Python owns the API, Zig owns the math.

Three reasons it's a subprocess rather than a binding. A job runs for minutes to hours, so IPC cost is noise. Killing the process is the only way to be certain the DiT's ~11 GB actually goes back to the OS, and on 24 GB that reclamation is the whole game. And an MLX over-commit hard-crashes the backend — as a child that's a failed job, in-process it would take the chat API down with it.

> **On the box this was built on**, all of that is already done: the 38 GB of weights and the compiled backend live at `/Volumes/PRIBNOW/mlx-h3/`, and that drive has its own README with the measured numbers, the resume steps and the traps. Read it before rebuilding anything.

### Setup

Build the backend and stage the weights:

```bash
git clone --recurse-submodules -b feature/minmax-h3 https://github.com/ddalcu/mlx-serve
cd mlx-serve && bash scripts/fetch-zig.sh && bash scripts/fetch-llama.sh
bash scripts/build-mlx.sh          # or stage brew's mlx + mlx-c into lib/mlx/
.zig-toolchain/zig build -Doptimize=ReleaseFast

hf download ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit --local-dir /path/to/weights
```

`build-mlx.sh` compiles MLX from source only to enable the M5 neural-accelerator kernels. On M1–M4 that changes nothing, and it needs the Metal compiler (full Xcode, not Command Line Tools) — so on those machines copy Homebrew's `mlx` + `mlx-c` into `lib/mlx/{lib,include}` instead.

Then point `models.toml` at the weights and the server at the binary:

```toml
[models."minimax-h3"]
kind = "video"
path = "/path/to/weights"
```

```bash
export MLX_LAZYSERVE_VIDEO_BINARY=/path/to/mlx-serve/zig-out/bin/mlx-serve
```

Without the binary the `/v1/videos` routes answer 501 and nothing else changes. `ffmpeg` is required for muxing.

### API

Job-shaped, not request-shaped — a job outlives any reverse proxy's idle timeout:

```bash
# submit
curl -s localhost:41434/v1/videos -H 'Content-Type: application/json' -d '{
  "prompt": "A red cube rotating on a white table, soft studio light.",
  "width": 768, "height": 768, "num_frames": 56, "steps": 50
}'                                          # -> {"id":"vid_...","status":"queued"}

curl -s localhost:41434/v1/videos/vid_...   # status + step progress + ETA
curl -sO localhost:41434/v1/videos/vid_.../content   # the mp4
curl -X DELETE localhost:41434/v1/videos/vid_...     # cancel
```

`width`/`height` must be multiples of 32. `num_frames` snaps **up** to the model's 17k+5 ladder (5, 22, 39, 56 … 362) — at 24 fps that's 0.2 s to 15.1 s. Pass `duration_seconds` instead if you'd rather think in seconds.

### It shares the one slot

A video job and a text model cannot both be resident: the DiT needs ~11 GB and the larger text models need up to 19 GB, against 24 GB total. So video is an occupant of the *same* single slot the text models use. Submitting a job evicts the loaded text model; while the job runs, `/v1/chat/completions` returns 503 with a `Retry-After` and the job's ETA rather than queueing behind something that may run for hours. When the queue drains, the backend is killed and the memory goes back.

### Context-IR

The full H3 system is three modules and only the middle one is open-weights: prompt expansion (H3-Context-IR) and 2K upscaling (H3-Regenerate-2K) stay behind MiniMax's Open Platform API. So the local ceiling is 768p, on any hardware.

Expansion matters more than it sounds — H3-Base is trained on a heavily structured prompt (shot-by-shot blocking with timecodes, a separate soundscape track, a separate score track), and Context-IR is what turns one ordinary sentence into that. Set `MLX_LAZYSERVE_MINIMAX_API_KEY` and prompts get expanded before generation; without it they pass through as written. Send `"prompt_mode": "raw"` when you've written the structure yourself. If the API call fails the raw prompt is used — losing a queued multi-hour job to a remote blip would be the worse outcome.

### Speed

Measured on an M4 Pro (16-core GPU, 24 GB) with the 4-bit weights. Time per denoise step against sequence length, where `tokens = latent_t × (H/32) × (W/32)` and `latent_t = ((frames-5)/17)×5 + 2`:

| shape | tokens | s/step |
|---|---|---|
| 256×256, 5f | 128 | 1.3 |
| 512×512, 5f | 512 | 4.2 |
| 768×768, 5f | 1152 | 9.6 |
| 512×512, 22f | 1792 | 15.5 |
| 768×768, 22f | 4032 | 33.2 |
| 1344×768, 22f | 7056 | 64.0 |

Those fit `t ≈ 0.56 + 7.40e-3·N + 2.23e-7·N²` seconds, within a few percent across the whole range. The linear term is the FFN and the projections; the quadratic one is attention, and past ~10k tokens it takes over. Add a fixed ~45 s per job for text encode, DiT load and VAE decode.

That curve is compute only, and on 24 GB compute is not what stops you first. The DiT holds ~10.6 GB resident and the attention activations grow with N on top of it, so past a certain sequence length the box starts paging and wall-clock stops tracking the formula entirely. Measured here: 7056 tokens runs clean, 17136 tokens (1344×768, 2.3 s) drove swap from 5 GB to 11 GB and the run had to be killed.

So treat the fit as a lower bound that only applies below the paging threshold, and size jobs by tokens rather than by resolution or duration alone. Watch `sysctl vm.swapusage` on the first run of any new shape: if swap grows by more than a few hundred MB, that shape does not fit, and no amount of waiting will fix it.

Quantization buys footprint, not speed — the workload is compute-bound and the DiT holds ~10.6 GB either way. What quantization does buy is headroom for a longer sequence before paging, which on this box is the constraint that actually binds.

## Extras

- **Reverse proxy**: [`deploy/nginx/mlx-lazyserve.conf`](deploy/nginx/mlx-lazyserve.conf) is an SSE-friendly nginx vhost (Cloudflare → nginx → Tailscale). Edit `server_name` and the upstream host for your setup.
- **Maintenance mode**: `POST /admin/maintenance {"enabled":true}` unloads the model and returns 503 for inference, for when a scheduled job needs the GPU/RAM back; `{"enabled":false}` resumes.
- **Telegram bot**: set `MLX_LAZYSERVE_TG_BOT_TOKEN` (from [@BotFather](https://t.me/BotFather)) and `uv sync --extra telegram`. It answers @mentions and replies in groups, keeps a short per-user history in SQLite, and lets each user pick a model (`/model`) and toggle reasoning (`/think`). It can also **search the web and read pages/PDFs** on demand via [Firecrawl](https://firecrawl.dev) — keyless by default (set `MLX_LAZYSERVE_FIRECRAWL_API_KEY` for higher limits, or `MLX_LAZYSERVE_TG_WEB_TOOLS=false` to turn it off). `MLX_LAZYSERVE_TG_OWNER_IDS` gates who can add it to a group or DM it. All `TG_*` options are in [`service.env.example`](deploy/service.env.example).

## Downloads from mainland China

Most of these repos are xet-backed, and xet transfer is slow from China. Set `HF_HUB_DISABLE_XET=1` to use the classic Hugging Face CDN instead (already set in the LaunchAgent). Interrupted downloads resume when you re-run.

## License

[GPL-3.0-or-later](LICENSE). Copyright (C) 2026 gtoxlili.
