# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANE runs text embedding and reranking models on the Apple Neural
Engine (ANE) of Apple Silicon Macs. Models are taken as-is in their
Hugging Face distribution form and compiled locally into Core ML
artifacts that load in seconds and run on the ANE — keeping your GPU
and most of your unified memory free for other work.

> **Status: early development (v0.8).**
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
> **v0.7 lays the multi-architecture, multi-model foundation**: a
> defined backend interface makes adding an architecture a matter of
> writing one backend module, and the first non-ModernBERT backend
> (XLM-RoBERTa) is verified end to end on
> [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base),
> [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
> and [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
> (compile → serve → exact HTTP/Core ML agreement, 93–98% ANE
> placement). The server now serves any number of embedding and
> reranker models at once and routes each request by its `model`
> field, `eeane compile` records a per-machine calibration in the
> artifact cache, and a `[[models]]` config entry can be just an
> `id = "..."` line — everything else is resolved from that cache.
> **v0.8 adds on-demand loading and idle unload**: `on_demand` is now
> the default `load_policy` for a `[[models]]` entry (`[server]
> default_load_policy` can change the default), so the server starts
> without loading any model and loads one the moment a request first
> needs it -- well under a second once its artifacts have loaded before
> (0.3-0.8 s measured on a development Mac), except for the very first
> load of a freshly compiled artifact, which can take tens of seconds
> while macOS builds its Neural Engine cache for it (later loads, even
> after a restart, are fast again). An on-demand model idle past its
> `keep_alive` (`[server] keep_alive`, default 300 s, per-model
> override) is unloaded automatically and reloaded on demand;
> `load_policy = "resident"` keeps a model always loaded as before v0.8,
> and `load_policy = "disabled"` removes an entry from service while
> leaving it in the config. `[server] max_loaded_models` caps how many
> models stay in memory at once, evicting the longest-idle on-demand
> model first. `GET /health` now reports each model's `loaded` state.
> Packaged one-command installs arrive in later milestones.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 or later
- Xcode Command Line Tools (`xcode-select --install`) — `eeane compile`
  uses `xcrun coremlcompiler`
- [uv](https://docs.astral.sh/uv/) (for the development environment)

## Compiling models and running the server

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync --extra compile   # torch/transformers are needed only for compiling

# Compile models straight from their Hugging Face IDs (auto-downloaded)
# or from local directories in HF distribution form. One-time; the
# artifacts land under ~/.cache/eeane/ and each bucket takes ~30-100 s:
uv run python -m eeane compile cl-nagoya/ruri-v3-310m
uv run python -m eeane compile cl-nagoya/ruri-v3-reranker-310m
uv run python -m eeane compile intfloat/multilingual-e5-base

# Each run ends with a ready-made [[models]] TOML snippet on stdout.
# Since v0.7 the snippet is minimal -- usually just the model id --
# because the server resolves everything else from the compiled-model
# cache. Paste the snippets into ./eeane.toml (see eeane.example.toml),
# then start the server:
uv run python -m eeane serve
```

`eeane compile` picks the model backend from the model's `config.json`.
Two architecture families are supported: **ModernBERT** (verified on
cl-nagoya/ruri-v3-310m and its reranker) and **XLM-RoBERTa** (verified
on intfloat/multilingual-e5-base, intfloat/multilingual-e5-large and
BAAI/bge-reranker-v2-m3; for embedding models the mean/CLS pooling
declared by the model directory is applied automatically). More
families are planned after 1.0. The compiler detects whether a model is
an embedding model or a reranker and defaults to buckets 128/512/1024
(embedding) or 512/1024 (reranker), clipped to the model's maximum
sequence length (multilingual-e5, capped at 512 tokens, compiles as
128/512); `--buckets 512,2048` compiles a custom set (S2048 is verified
on M2 at ~518 ms/inference). Re-running
skips up-to-date artifacts (`--force` reconverts). After every
conversion a **self-check** verifies accuracy against the FP32 original,
measures how many operations landed on the Neural Engine, and records
warm latency — the printed summary doubles as a compatibility report:
if you run eeANE on hardware we have not verified (M1/M3/M4...), please
paste it into an issue. The per-bucket measurements are aggregated into
a calibration record (`model_info.json`) in the cache; buckets whose
self-check failed are dropped from the recommended set that
cache-resolved configs load. The tokenizer is frozen into the artifact
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

The config file lists the served models — any number of embedding and
reranker entries. A `[[models]]` entry usually needs only its
`id = "..."`: the kind, frozen tokenizer, per-bucket artifacts and
embedding width are then resolved from the compiled-model cache
(`server.cache_root`, default `~/.cache/eeane/`), honouring the
calibration's recommended buckets. Spelling out `kind`, `tokenizer` and
`[models.artifacts]` explicitly (the pre-v0.7 form) still works and
pins the entry independently of the cache. Within each kind the
first-listed entry is the default model, used when a request does not
name one. Reranker entries may be omitted entirely for an
embedding-only server (`/rerank` then answers 503).
`uv run python -m eeane.server` remains as an alias for `eeane serve`.

### Model loading

Since v0.8 the default `load_policy` for a `[[models]]` entry is
`"on_demand"` (`[server] default_load_policy` can change the default;
see `eeane.example.toml` for the setting): the server does not load
any model at start-up, and loads one the moment a request first needs
it. Once a model's artifacts have loaded once, a load is well under a
second (0.3-0.8 s measured on a development Mac); the exception is the
very first load of an artifact right after `eeane compile` produced
it, which can take tens of seconds while macOS builds its Neural
Engine cache for it — a one-time cost that later loads, even after a
server restart, do not pay again. That wait is included in the
response time of whichever request triggers it.

An on-demand model that has answered no request for `keep_alive`
seconds (`[server] keep_alive`, default 300, overridable per model;
`0` unloads it as soon as it goes idle) is unloaded automatically and
reloaded on the next request that needs it. Set `load_policy =
"resident"` on an entry to load it at start-up and keep it in memory
for the server's whole run, matching the pre-v0.8 default. Set
`load_policy = "disabled"` to keep an entry in the config file without
serving it: it is absent from `GET /models` and `GET /health`, and a
request naming its id gets a 404.

`[server] max_loaded_models` caps how many models may be in memory at
once (unset means no limit). When loading a model would exceed it,
the longest-idle `on_demand` model is unloaded to make room;
`resident` models and models currently handling a request are never
evicted this way, so a configuration whose `resident` entries alone
exceed the cap is rejected at start-up.

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

- `GET /health` — status and one entry per served model (`id`, `kind`,
  buckets in service, `loaded`), unauthenticated, rate-limited
- `GET /models` (alias: `GET /v1/models`) — OpenAI-compatible listing
  of every served model
- `POST /v1/embeddings` (alias: `POST /embeddings`) — OpenAI-compatible
  (`input` as string or list, `encoding_format` `float`/`base64`);
  embeddings are L2-normalized by default (per-model `normalize`)
- `POST /rerank`, `POST /v1/rerank` — Infinity-compatible
  (`query`/`documents`/`top_n`/`return_documents`/`raw_scores`)

The optional `model` field of the embeddings and rerank requests
selects the served model by its configured id; omitting it selects the
first-listed model of the endpoint's kind. An unknown id gets a 404
listing the servable ids, and naming a model of the other kind gets a
400. The embeddings and rerank endpoints are served both under `/v1`
and at the root, so a base URL with or without the `/v1` suffix works.
Each input is routed to the smallest fitting sequence-length bucket of
its model and truncated to the largest bucket when longer, with a
server-side warning.

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
# Check one specific served model against direct Core ML inference:
uv run python tools/verify_server.py verify-embedding --model intfloat/multilingual-e5-base
uv run python tools/verify_server.py verify-rerank --model BAAI/bge-reranker-v2-m3
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
