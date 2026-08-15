# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANE runs text embedding and reranking models on the Apple Neural
Engine (ANE) of Apple Silicon Macs. Models are taken as-is in their
Hugging Face distribution form and compiled locally into Core ML
artifacts that load in seconds and run on the ANE — keeping your GPU
and most of your unified memory free for other work.

> **Status: early development (v0.6).**
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
> `GET /models` listing, and rate limiting on `/health`. **v0.6 turns
> model conversion into a product**: `eeane compile <model>` takes a
> local directory or a Hugging Face model ID (auto-downloaded),
> converts it into per-bucket `.mlmodelc` artifacts under
> `~/.cache/eeane/`, freezes the tokenizer alongside them (verified
> token-for-token against the original), runs a self-check (accuracy
> vs. FP32, ANE placement, warm latency) whose report doubles as a
> hardware compatibility report, and prints a ready-to-paste config
> snippet. The server itself no longer needs torch or transformers —
> heavyweight dependencies moved to the optional `[compile]` extra.
> Packaged one-command installs arrive in later milestones.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 or later
- Xcode Command Line Tools (`xcode-select --install`) — `eeane compile`
  uses `xcrun coremlcompiler`
- [uv](https://docs.astral.sh/uv/) (for the development environment)

## Compiling models and running the server (v0.6)

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync --extra compile   # torch/transformers are needed only for compiling

# Compile a model straight from its Hugging Face ID (auto-downloaded)
# or from a local directory in HF distribution form. One-time; the
# artifacts land under ~/.cache/eeane/ and each bucket takes ~30-100 s:
uv run python -m eeane compile cl-nagoya/ruri-v3-310m
uv run python -m eeane compile cl-nagoya/ruri-v3-reranker-310m

# Each run ends with a ready-made [[models]] TOML snippet on stdout --
# paste the snippets into ./eeane.toml (see eeane.example.toml for the
# [server] section), then start the server:
uv run python -m eeane serve
```

`eeane compile` picks the model backend from the model's `config.json`
(v0.6 supports the ModernBERT architecture), detects whether it is an
embedding model or a reranker, and defaults to buckets 128/512/1024
(embedding) or 512/1024 (reranker); `--buckets 512,2048` compiles a
custom set (S2048 is verified on M2 at ~518 ms/inference). Re-running
skips up-to-date artifacts (`--force` reconverts). After every
conversion a **self-check** verifies accuracy against the FP32 original,
measures how many operations landed on the Neural Engine, and records
warm latency — the printed summary doubles as a compatibility report:
if you run eeANE on hardware we have not verified (M1/M3/M4...), please
paste it into an issue. The tokenizer is frozen into the artifact
directory and verified to reproduce the original tokenization exactly,
so the server needs neither the original model files nor the
transformers library at run time (see
[docs/dependency-policy.md](docs/dependency-policy.md)).

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

The config file defines the served models (frozen `tokenizer.json`,
compiled artifact per sequence-length bucket, L2 normalization) — which
is exactly what the `eeane compile` snippet fills in — so buckets can
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

## Trying the PoC (historical development snapshot)

The `poc/` scripts are the frozen v0.1–v0.3 research record; the
supported conversion path is `eeane compile` above. They remain runnable
for benchmarking studies:

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
