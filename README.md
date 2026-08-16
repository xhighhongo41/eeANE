# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANE runs text embedding and reranking models on the Apple Neural
Engine (ANE) of Apple Silicon Macs. Models are taken as-is in their
Hugging Face distribution form and compiled locally into Core ML
artifacts that load in seconds and run on the ANE — keeping your GPU
and most of your unified memory free for other work.

> **Status: early development (v0.10).**
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
> **v0.9 hardens the server under load**: inference requests now pass
> admission control — `[server] max_pending_requests` (default 500)
> caps how many are accepted at once and the excess is rejected
> immediately with `429` plus a `Retry-After` header, while a request
> that waits longer than `[server] queue_timeout` (default 600 s) for
> its turn is answered with `503` instead of waiting forever (a
> request whose inference has started always runs to completion, and
> shutdown drains in-flight requests, bounded by
> `graceful_shutdown_timeout` if set). Identical concurrent requests
> are served from a single computation (`coalesce_requests`, on by
> default; 8 identical requests measured ~7x faster than without it),
> and a model output containing NaN or infinite values — the sign of
> an unsupported compute path — is reported as a clear `500` error
> instead of being passed on. Embedding throughput for requests full
> of short inputs can be raised ~25% (measured on a development Mac)
> by compiling a batch-2 artifact (`eeane compile <model> --buckets
> <S> --batch 2`), which the server picks up automatically for
> id-only entries and uses to predict two same-bucket inputs of one
> request together. **v0.10 makes eeANE installable directly from
> GitHub**: `uv`, `pipx` and `pip` can each install it straight from
> this repository (see Installation below), and the package now
> provides an `eeane` console command in place of `python -m eeane`.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 or later
- Python 3.11 or 3.12 (3.13 and later are not yet supported). `uv`
  resolves a matching interpreter automatically; installing with pipx
  or pip + venv instead means providing one yourself.
- Xcode Command Line Tools (`xcode-select --install`) — `eeane compile`
  uses `xcrun coremlcompiler`
- [uv](https://docs.astral.sh/uv/) — the recommended way to install
  eeANE (see Installation below), and required for the development
  workflow

## Installation

Install eeANE directly from this GitHub repository with any of the
methods below.

### uv (recommended)

A single command installs eeANE together with the `[compile]` extra
(torch/transformers), so the same environment can both compile models
and serve them:

```sh
uv tool install "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@v0.10.0"
```

Pin a released tag (`@v0.10.0`, the latest release) for a reproducible
install, or use `@main` to track the latest development version
instead. To upgrade, run the same command with a newer tag and add
`--force` to replace the existing install.

### pipx

```sh
pipx install --python python3.12 "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@v0.10.0"
```

pipx's default Python interpreter may be 3.13 or later, which eeANE
does not yet support, so pass `--python` naming a Python 3.11 or 3.12
executable available on your machine (for example `python3.11`, or a
full path to one).

### pip + venv

```sh
python3.12 -m venv eeane-env
eeane-env/bin/pip install "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@v0.10.0"
```

Substitute `python3.11` if that is the supported interpreter you have
available instead.

### Lightweight install (server only)

The `[compile]` extra pulls in torch and transformers, but the server
itself never imports them — they are needed only when running `eeane
compile` to convert a model into Core ML artifacts. Keeping them
installed alongside the server costs disk space (a few GB) but has no
effect on the server's memory use or behavior, so the combined install
above is a reasonable default. If you would rather keep the
always-installed environment down to eeANE's five runtime dependencies,
install eeANE without the extra and run `eeane compile` from a
disposable environment instead:

```sh
uv tool install "eeane @ git+https://github.com/xhighhongo41/eeANE@v0.10.0"
uvx --from "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@v0.10.0" eeane compile <model>
```

## Compiling models and running the server

```sh
# Compile models straight from their Hugging Face IDs (auto-downloaded)
# or from local directories in HF distribution form. One-time; the
# artifacts land under ~/.cache/eeane/ and each bucket takes ~30-100 s:
eeane compile cl-nagoya/ruri-v3-310m
eeane compile cl-nagoya/ruri-v3-reranker-310m
eeane compile intfloat/multilingual-e5-base

# Each run ends with a ready-made [[models]] TOML snippet on stdout.
# Since v0.7 the snippet is minimal -- usually just the model id --
# because the server resolves everything else from the compiled-model
# cache. Paste the snippets into ./eeane.toml (see eeane.example.toml),
# then start the server:
eeane serve
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
eeane serve --config /path/to/eeane.toml
eeane serve --host 192.168.1.20 --port 7997

# Validate a config file and print the resolved effective configuration
# (the API key value is never printed) without starting the server:
eeane check-config --config /path/to/eeane.toml
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
embedding-only server (`/rerank` then answers 503). `python -m
eeane.server` and `python -m eeane <subcommand>` remain available as
backward-compatible aliases for `eeane serve` and the `eeane` command
respectively (prefix both with `uv run` in the development
environment).

### Batch-2 artifacts for embedding requests

Embedding models (not rerankers) may optionally be compiled with a
second artifact per bucket that packs two inputs into one Neural Engine
call: `eeane compile <model> --buckets <S> --batch 2`, run alongside
the normal batch-1 compile. Serving it is opt-in through
`[models.batch_artifacts]` on a `[[models]]` entry — a bucket ->
artifact-path table mirroring `[models.artifacts]`. When a request
routes two or more of its inputs to the same bucket, they are paired up
and inferred through the batch-2 artifact instead of one at a time,
which raised throughput by about 25% on our test machine for requests
carrying many short inputs. An id-only entry resolves
`batch_artifacts` automatically from the compiled-model cache once a
batch-2 artifact has been compiled for it; the explicit form (which
spells out `[models.artifacts]`) must spell out
`[models.batch_artifacts]` too — it cannot be set on its own. A
configuration without any batch-2 artifacts behaves exactly as before.

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

### Request admission, queueing and shutdown

`server.max_pending_requests` caps how many inference requests the
server admits at once, counting both requests currently running and
requests still waiting their turn (default 500; `0` means unlimited).
A request that arrives once the cap is reached is rejected immediately
with `429 Too Many Requests` and a `Retry-After` header.

`server.queue_timeout` caps how long an admitted request may wait
between being accepted and actually starting inference (default 600
seconds; `0` disables the timeout). A request that waits past this
limit is abandoned with `503 Service Unavailable` and a `Retry-After`
header. Once a request has started inference it is never interrupted
by this timeout, however long it runs. Either way, `Retry-After` tells
the client how long to wait before retrying.

`server.coalesce_requests` (default `true`) merges an incoming request
with an identical one (same model, same input) that is already being
processed: instead of running inference twice, the second request
attaches to the first and receives the same result once it completes.

`server.graceful_shutdown_timeout` bounds how long the server waits
for in-flight requests to finish when it receives SIGTERM or Ctrl-C
(default: unset, meaning it waits for all of them to finish however
long that takes; no new connections are accepted while it waits). Set
it to a number of seconds to cap that wait instead.

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

### Known limitations

eeANE targets the Apple Neural Engine; running a compiled model on a
CPU-only compute path is not supported. On a machine or configuration
where the Neural Engine is not actually available to a compiled model,
inference can produce non-finite (NaN/Inf) output — this has been
observed across every architecture eeANE supports, with how often it
happens depending on the architecture and the input's sequence length.
Rather than silently return such a result, the server detects
non-finite output at inference time and answers with `500 Internal
Server Error` (for example: `model '<id>' produced a non-finite output
for bucket <N>; the compiled model may have run on an unsupported
compute path`). Seeing this error is a strong signal that the Neural
Engine is not actually being used in your environment; see
Troubleshooting below.

### Troubleshooting

- **`404 model not found`**: the `model` field a client sends must
  match a served model's configured `id` exactly. Check the ids eeANE
  actually serves with `GET /models`. Clients migrated from an older
  eeANE version should note that the `model` field used to be ignored
  entirely, so a request naming anything (or nothing) used to succeed
  — that leniency is gone.
- **`500 ... produced a non-finite output ...`**: see "Known
  limitations" above. This means the model ran off the Neural Engine;
  verify Neural Engine availability on the machine serving the
  request.

## Development

To work on eeANE itself, or to use the repository-only tools below,
clone the repository and run commands with `uv run` from the checkout
instead of installing the package:

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync --extra compile   # torch/transformers are needed only for compiling
uv run eeane compile cl-nagoya/ruri-v3-310m
uv run eeane serve
```

`uv run eeane <subcommand>` runs the same `compile`/`serve`/
`check-config` subcommands described above, from the checkout rather
than an installed package.

To verify a running server end to end (accuracy vs. direct Core ML
inference, API compatibility, latency), and to lint and test the
codebase in one step — both assume a repository checkout:

```sh
uv run python tools/verify_server.py all
# Check one specific served model against direct Core ML inference:
uv run python tools/verify_server.py verify-embedding --model intfloat/multilingual-e5-base
uv run python tools/verify_server.py verify-rerank --model BAAI/bge-reranker-v2-m3
./tools/check.sh   # ruff lint + format check + pytest, in one step
```

### Trying the PoC (historical development snapshot)

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
