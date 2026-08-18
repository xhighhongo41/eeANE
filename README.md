# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

[![PyPI](https://img.shields.io/pypi/v/eeane)](https://pypi.org/project/eeane/)
[![CI](https://github.com/xhighhongo41/eeANE/actions/workflows/ci.yml/badge.svg)](https://github.com/xhighhongo41/eeANE/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

eeANE runs text embedding and reranking models on the Apple Neural
Engine (ANE) of Apple Silicon Macs. Models are taken as-is in their
Hugging Face distribution form and compiled locally into Core ML
artifacts that load in seconds and run on the ANE — keeping your GPU
and most of your unified memory free for other work.

## Highlights

- **ANE inference**: embeddings at up to ~13,600 effective tokens/s on
  an M2 Mac mini, 2–3x the same model served from the MPS GPU by
  PyTorch — while leaving the GPU idle (see Performance below).
- **No model modification, no re-distribution**: `eeane compile` takes
  a Hugging Face model ID (or a local directory in HF distribution
  form) and converts it on your machine, then verifies the result
  against the FP32 original with a built-in self-check.
- **Standard APIs**: OpenAI-compatible `/v1/embeddings` and
  Infinity-compatible `/rerank`, so existing clients (Open WebUI among
  them) connect by changing a base URL.
- **Cheap to keep running**: models load on demand in well under a
  second and unload after an idle timeout; the always-on server itself
  is a small Python process with five runtime dependencies (no torch,
  no transformers).
- **Multi-model serving** with per-request routing, admission control
  (429/503 + `Retry-After`), identical-request coalescing, and graceful
  shutdown.
- **Two architecture families supported today**: ModernBERT and
  XLM-RoBERTa, both embedding and cross-encoder reranker models. More
  are planned.

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

eeANE requires the Apple Neural Engine: CPU-only execution is not
supported (see Known limitations). Docker is not supported either —
containers on macOS run inside a Linux VM, and the ANE is not passed
through to it.

## Installation

eeANE is on [PyPI](https://pypi.org/project/eeane/). The `[compile]`
extra adds torch/transformers, which only `eeane compile` needs; the
combined install below gives one environment that can both compile
models and serve them.

### uv (recommended)

```sh
uv tool install "eeane[compile]"
```

To upgrade later, run `uv tool upgrade eeane`.

### pipx

```sh
pipx install --python python3.12 "eeane[compile]"
```

pipx's default Python interpreter may be 3.13 or later, which eeANE
does not yet support, so pass `--python` naming a Python 3.11 or 3.12
executable available on your machine (for example `python3.11`, or a
full path to one).

### pip + venv

```sh
python3.12 -m venv eeane-env
eeane-env/bin/pip install "eeane[compile]"
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
uv tool install eeane
uvx --from "eeane[compile]" eeane compile <model>
```

### Installing from GitHub

To install a development snapshot or pin an exact repository revision,
install from a git URL instead of PyPI — with any of the tools above,
for example:

```sh
uv tool install "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@main"
```

`@main` tracks the latest development version; `@v1.0.0` (or any other
release tag) pins a released revision. To switch an existing install,
run the same command with `--force`.

## Quick start

```sh
# Compile models straight from their Hugging Face IDs (auto-downloaded)
# or from local directories in HF distribution form. One-time; the
# artifacts land under ~/.cache/eeane/ and each bucket takes ~30-100 s:
eeane compile cl-nagoya/ruri-v3-310m
eeane compile cl-nagoya/ruri-v3-reranker-310m
eeane compile intfloat/multilingual-e5-base

# Each run ends with a ready-made [[models]] TOML snippet on stdout.
# The snippet is minimal -- usually just the model id -- because the
# server resolves everything else from the compiled-model cache. Paste
# the snippets into ./eeane.toml (see eeane.example.toml), then start
# the server:
eeane serve
```

Then, from another shell:

```sh
curl -s http://127.0.0.1:7997/health

curl -s http://127.0.0.1:7997/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "intfloat/multilingual-e5-base", "input": "hello eeANE"}'
```

### About `eeane compile`

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

## Configuration

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
`[models.artifacts]` explicitly still works and
pins the entry independently of the cache. Within each kind the
first-listed entry is the default model, used when a request does not
name one. Reranker entries may be omitted entirely for an
embedding-only server (`/rerank` then answers 503). `python -m
eeane.server` and `python -m eeane <subcommand>` remain available as
backward-compatible aliases for `eeane serve` and the `eeane` command
respectively (prefix both with `uv run` in the development
environment).

## Serving and operations

### Model loading

The default `load_policy` for a `[[models]]` entry is
`"on_demand"` (`[server] default_load_policy` can change the default;
see `eeane.example.toml` for the setting): the server does not load
any model at start-up, and loads one the moment a request first needs
it. Once a model's artifacts have loaded once, a load is well under a
second (0.3-0.8 s measured on an M2 Mac); the exception is the
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
for the server's whole run. Set
`load_policy = "disabled"` to keep an entry in the config file without
serving it: it is absent from `GET /models` and `GET /health`, and a
request naming its id gets a 404.

`[server] max_loaded_models` caps how many models may be in memory at
once (unset means no limit). When loading a model would exceed it,
the longest-idle `on_demand` model is unloaded to make room;
`resident` models and models currently handling a request are never
evicted this way, so a configuration whose `resident` entries alone
exceed the cap is rejected at start-up.

### Batch-2 artifacts for embedding requests

Embedding models (not rerankers) may optionally be compiled with a
second artifact per bucket that packs two inputs into one Neural Engine
call: `eeane compile <model> --buckets <S> --batch 2`, run alongside
the normal batch-1 compile. Serving it is opt-in through
`[models.batch_artifacts]` on a `[[models]]` entry — a bucket ->
artifact-path table mirroring `[models.artifacts]`. When a request
routes two or more of its inputs to the same bucket, they are paired up
and inferred through the batch-2 artifact instead of one at a time,
which raised throughput for requests carrying many short inputs by
about 25% in benchmarks on an M2 Mac. An id-only entry resolves
`batch_artifacts` automatically from the compiled-model cache once a
batch-2 artifact has been compiled for it; the explicit form (which
spells out `[models.artifacts]`) must spell out
`[models.batch_artifacts]` too — it cannot be set on its own. A
configuration without any batch-2 artifacts behaves exactly as before.

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

### Running as a service

To start the server automatically at login and keep it running, set it
up as a macOS launchd agent — see [docs/launchd.md](docs/launchd.md)
for a step-by-step guide and a ready-made plist template. Thanks to
on-demand loading, an always-on eeANE agent costs almost nothing while
idle.

## API

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

## Performance

Figures below were measured on an M2 Mac mini (macOS 13+, 16 GB), with
the same models served from the MPS GPU by PyTorch (sentence-transformers)
as the baseline:

- **Embedding throughput**: up to ~13,600 effective (padding-excluded)
  tokens/s on the ANE — 2–3x the MPS baseline, at a similar power draw
  but 2.6–3.8x the energy efficiency per token, with the GPU left
  entirely free.
- **Reranking**: a 36-document rerank over HTTP completes in
  ~2.0–5.6 s depending on chunk length, ~2–8x faster than the same
  request against MPS-based serving of the same model.
- **Memory**: a server holding one 310M-class embedding model and one
  reranker resident stays around 750 MB; compiled weights live mostly
  outside the Python process, and on-demand entries release their
  memory when idle.
- **Load times**: ~0.2–0.8 s per model once macOS has cached a
  compiled artifact (the very first load after compiling takes tens of
  seconds, once).

Responses over HTTP are verified to match direct Core ML inference
exactly (`tools/verify_server.py` in a repository checkout).

## Troubleshooting

- **`404 model not found`**: the `model` field a client sends must
  match a served model's configured `id` exactly. Check the ids eeANE
  actually serves with `GET /models`. Clients migrated from an older
  eeANE version should note that the `model` field used to be ignored
  entirely, so a request naming anything (or nothing) used to succeed
  — that leniency is gone.
- **`500 ... produced a non-finite output ...`**: see Known
  limitations below. This means the model ran off the Neural Engine;
  verify Neural Engine availability on the machine serving the
  request.
- **A request occasionally takes much longer than usual**: the first
  request after a model was idle past `keep_alive` pays the on-demand
  reload (typically well under a second), and the very first request
  after `eeane compile` pays the one-time Neural Engine cache build
  (tens of seconds). Both are expected; use `load_policy = "resident"`
  if you need to avoid even the sub-second reload.

## Known limitations

- **ANE only**: eeANE targets the Apple Neural Engine; running a
  compiled model on a CPU-only compute path is not supported. On a
  machine or configuration where the Neural Engine is not actually
  available to a compiled model, inference can produce non-finite
  (NaN/Inf) output — this has been observed across every architecture
  eeANE supports. Rather than silently return such a result, the
  server detects non-finite output at inference time and answers with
  `500 Internal Server Error`. Seeing this error is a strong signal
  that the Neural Engine is not actually being used in your
  environment.
- **Verified hardware**: all published measurements and verifications
  were run on an M2 Mac. Other Apple Silicon generations (M1/M3/M4...)
  are expected to work but are unverified by the maintainer; the
  self-check summary that `eeane compile` prints doubles as a
  compatibility report for exactly this reason. Reports from other
  machines — success or failure — are very welcome as GitHub issues.
- **Long documents**: each input is truncated to its model's largest
  compiled bucket (add larger buckets with `eeane compile --buckets`
  if you need them); rerankers have no sliding-window handling for
  documents beyond that.

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

## Acknowledgments and related projects

eeANE was inspired by
[Infinity](https://github.com/michaelfeil/infinity), the open-source
serving engine that showed how convenient a self-hosted,
API-compatible embedding and reranking server can be — eeANE's
`/rerank` API deliberately follows Infinity's schema so that clients
can switch between the two by changing a URL.

eeANE exists in the first place because its author could run
ModernBERT-based embedding models on a GPU with Infinity. The two
projects complement rather than compete with each other: eeANE runs
models exclusively on the Apple Neural Engine of Apple Silicon Macs,
and supports a deliberately small set of model architectures. If you
want to serve embedding or reranking models on Linux or Windows, on
NVIDIA/AMD GPUs or CPUs, or need a much wider model catalogue, by all
means use Infinity.

## Changelog

| Version | Highlights |
|---|---|
| 1.0.0 | First stable release: published on PyPI, launchd service guide, documentation overhaul |
| 0.10.0 | Installable straight from GitHub with uv/pipx/pip; `eeane` console command |
| 0.9.0 | Admission control (429/503 + `Retry-After`), identical-request coalescing, graceful shutdown, non-finite output guard, opt-in batch-2 artifacts |
| 0.8.0 | On-demand loading, idle unload (`keep_alive`), `max_loaded_models` eviction |
| 0.7.0 | Multi-architecture backends (XLM-RoBERTa joins ModernBERT), multi-model serving and routing, cache auto-resolution with per-machine calibration |
| 0.6.0 | `eeane compile`: HF ID/local directory -> Core ML artifacts with self-check and frozen tokenizer; torch-free server runtime |
| 0.5.0 | TOML config + CLI, API key auth, `GET /models`, `/health` rate limit, CI |
| 0.4.0 | First HTTP server: OpenAI-compatible embeddings, Infinity-compatible rerank |
| 0.1.0–0.3.0 | Proof of concept: ANE conversion and inference of an embedding model and a reranker, accuracy verification, performance study vs. GPU |

Details for each release: [GitHub Releases](https://github.com/xhighhongo41/eeANE/releases).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

The test corpus under `testdata/corpus/` consists of public-domain
literary works from [Aozora Bunko](https://www.aozora.gr.jp/) and is
not covered by the GPL; see `testdata/corpus/README.md`.

---

日本語のREADMEは [README_ja.md](README_ja.md) を参照してください。
