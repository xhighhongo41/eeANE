# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANE runs text embedding and reranking models on the Apple Neural
Engine (ANE) of Apple Silicon Macs. Models are taken as-is in their
Hugging Face distribution form and compiled locally into Core ML
artifacts that load in seconds and run on the ANE — keeping your GPU
and most of your unified memory free for other work.

> **Status: early development (v0.5).**
> v0.1–v0.3 proved the concept:
> [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)
> (a Japanese ModernBERT embedding model) and
> [cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)
> (its cross-encoder reranker) are converted to Core ML through a
> shared pipeline, verified for ANE inference and numerical accuracy,
> and benchmarked on an M2 Mac mini at up to ~13,600 effective
> tokens/s for embeddings (2–3x the MPS GPU baseline). v0.4 added the
> first HTTP server: an OpenAI-compatible `/v1/embeddings` endpoint
> and an Infinity-compatible `/rerank` endpoint served from the ANE,
> with responses that match direct Core ML inference exactly (a
> 36-document rerank over HTTP takes ~2.0–5.6 s depending on chunk
> length, ~3–8x faster than the same request against an
> Infinity_emb/MPS deployment on the same machine, with ~750 MB
> resident vs. 6–8 GB for the setup it replaces). **v0.5 makes the
> server configurable and operable**: a TOML config file plus an
> `eeane serve` / `eeane check-config` CLI (bind address, port,
> models and their buckets, log level), optional Bearer API key
> authentication for serving beyond localhost, an OpenAI-compatible
> `GET /models` listing, and rate limiting on `/health`. Packaged
> one-command installs and `eeane compile` arrive in later
> milestones.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 or later
- [uv](https://docs.astral.sh/uv/) (for the development environment)

## Running the server (v0.5)

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync

# Place the models in HF distribution form under models/ruri-v3-310m and
# models/ruri-v3-reranker-310m (e.g. download with
# `huggingface-cli download cl-nagoya/ruri-v3-310m`), then compile the
# Core ML artifacts the server loads (one-time, ~30 s each):
uv run python poc/convert_embedding.py --seq-len 128
uv run python poc/convert_embedding.py --seq-len 512
uv run python poc/convert_embedding.py --seq-len 1024
uv run python poc/convert_reranker.py --seq-len 512
uv run python poc/convert_reranker.py --seq-len 1024

# Start the server (defaults: 127.0.0.1:7997, the two models above)
uv run python -m eeane serve
```

### Configuration

The server runs with built-in defaults out of the box. To change them,
copy [`eeane.example.toml`](eeane.example.toml) to `./eeane.toml` (or
`~/.config/eeane/eeane.toml`) and edit it. Config files are searched in
this order: `--config PATH` > `./eeane.toml` >
`~/.config/eeane/eeane.toml` > built-in defaults. The
`--host`/`--port`/`--log-level` CLI flags and the `EEANE_API_KEY`
environment variable override the file.

```sh
uv run python -m eeane serve --config /path/to/eeane.toml
uv run python -m eeane serve --host 192.168.1.20 --port 7997

# Validate a config file and print the resolved effective configuration
# (the API key value is never printed) without starting the server:
uv run python -m eeane check-config --config /path/to/eeane.toml
```

The config file defines the served models (HF model directory, compiled
artifact per sequence-length bucket, L2 normalization), so buckets can
be added or removed without touching code. The reranker entry may be
omitted for an embedding-only server (`/rerank` then answers 503).
`uv run python -m eeane.server` (the v0.4 entry point) remains as an
alias for `eeane serve`.

### Serving beyond localhost

Binding to a non-loopback address (`--host` or `server.host`) exposes
the server to your network. Set an API key — `api_key` in the config
file (keep it `chmod 600`) or the `EEANE_API_KEY` environment variable
— and every endpoint except `GET /health` will require an
`Authorization: Bearer <key>` header; the server logs a warning when it
serves a non-loopback address without one. `/health` stays open for
monitoring and is rate-limited instead (`server.health_rate_limit`,
default 60 requests/min per client IP, `0` disables). These are
application-level safeguards only: for exposure beyond a trusted
LAN/VPN, put the server behind a reverse proxy or firewall.

### Endpoints

- `GET /health` — status and the sequence-length buckets in service
  (unauthenticated, rate-limited)
- `GET /models` (alias: `GET /v1/models`) — OpenAI-compatible model
  listing
- `POST /v1/embeddings` (alias: `POST /embeddings`) — OpenAI-compatible
  (`input` as string or list, `encoding_format` `float`/`base64`);
  embeddings are L2-normalized, matching Infinity_emb's behavior
- `POST /rerank`, `POST /v1/rerank` — Infinity-compatible
  (`query`/`documents`/`top_n`/`return_documents`/`raw_scores`)

The embeddings and rerank endpoints are served both under `/v1` and at
the root, so a base URL with or without the `/v1` suffix works. Each input is routed to the
smallest fitting sequence-length bucket (defaults — embeddings:
128/512/1024 tokens; reranker: 512/1024) and truncated to the largest
bucket when longer, with a server-side warning.

To use eeANE from [Open WebUI](https://github.com/open-webui/open-webui):
set the embedding engine to OpenAI with base URL
`http://127.0.0.1:7997/v1`, and the reranking engine to External with URL
`http://127.0.0.1:7997/rerank`. If you configured an API key, enter it
as the OpenAI API key / External reranker API key — Open WebUI sends it
as the `Authorization` header eeANE expects.

To verify a running server end to end (accuracy vs. direct Core ML
inference, API compatibility, latency):

```sh
uv run python tools/verify_server.py all
```

## Trying the PoC (development snapshot)

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync

# Place the models in HF distribution form under models/ruri-v3-310m and
# models/ruri-v3-reranker-310m (e.g. download with
# `huggingface-cli download cl-nagoya/ruri-v3-310m`), then:

# Embedding model (v0.1)
uv run python poc/convert_embedding.py --seq-len 512   # HF -> .mlmodelc
uv run python poc/verify_accuracy.py --seq-len 512     # accuracy vs FP32
uv run python poc/benchmark_latency.py --seq-len 512 --compute-units CPU_AND_NE --compute-plan

# Reranker model (v0.2)
uv run python poc/convert_reranker.py --seq-len 512
uv run python poc/verify_reranker_accuracy.py --seq-len 512
uv run python poc/benchmark_latency.py --model reranker --seq-len 512 --compute-units CPU_AND_NE --compute-plan

# Performance study (v0.3)
uv run python poc/run_sweep.py --seq-lens 128,512 --batches 1,2      # S x B latency matrix
uv run python poc/benchmark_throughput.py --model embedding --chunk-tokens 128 --batch 2
uv run python poc/benchmark_mps.py --model embedding --chunk-tokens 512 --batch 32  # GPU baseline
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

The test corpus under `testdata/corpus/` consists of public-domain
literary works from [Aozora Bunko](https://www.aozora.gr.jp/) and is
not covered by the GPL; see `testdata/corpus/README.md`.

---

日本語のREADMEは [README_ja.md](README_ja.md) を参照してください。
