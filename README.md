# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANE runs text embedding and reranking models on the Apple Neural
Engine (ANE) of Apple Silicon Macs. Models are taken as-is in their
Hugging Face distribution form and compiled locally into Core ML
artifacts that load in seconds and run on the ANE — keeping your GPU
and most of your unified memory free for other work.

> **Status: early development (v0.4).**
> v0.1–v0.3 proved the concept:
> [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)
> (a Japanese ModernBERT embedding model) and
> [cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)
> (its cross-encoder reranker) are converted to Core ML through a
> shared pipeline, verified for ANE inference and numerical accuracy,
> and benchmarked on an M2 Mac mini at up to ~13,600 effective
> tokens/s for embeddings (2–3x the MPS GPU baseline). **v0.4 adds the
> first HTTP server**: an OpenAI-compatible `/v1/embeddings` endpoint
> and an Infinity-compatible `/rerank` endpoint served from the ANE,
> with responses that match direct Core ML inference exactly. Over
> HTTP, a 36-document rerank takes ~2.0 s (~8x faster than the same
> request against an Infinity_emb/MPS deployment on the same machine)
> and the resident server uses ~750 MB (vs. 6–8 GB for the setup it
> replaces). Configuration files, on-demand model loading and packaged
> installs arrive in later milestones — for now the server uses fixed
> built-in settings.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 or later
- [uv](https://docs.astral.sh/uv/) (for the development environment)

## Running the server (v0.4)

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

# Start the server (fixed settings: 127.0.0.1:7997)
uv run python -m eeane.server
```

Endpoints:

- `GET /health` — status and the sequence-length buckets in service
- `POST /v1/embeddings` (alias: `POST /embeddings`) — OpenAI-compatible
  (`input` as string or list, `encoding_format` `float`/`base64`);
  embeddings are L2-normalized, matching Infinity_emb's behavior
- `POST /rerank`, `POST /v1/rerank` — Infinity-compatible
  (`query`/`documents`/`top_n`/`return_documents`/`raw_scores`)

The embeddings and rerank endpoints are served both under `/v1` and at
the root, so a base URL with or without the `/v1` suffix works. Each input is routed to the
smallest fitting sequence-length bucket (embeddings: 128/512/1024
tokens; reranker: 512/1024) and truncated to the largest bucket when
longer, with a server-side warning.

To use eeANE from [Open WebUI](https://github.com/open-webui/open-webui):
set the embedding engine to OpenAI with base URL
`http://127.0.0.1:7997/v1`, and the reranking engine to External with URL
`http://127.0.0.1:7997/rerank`.

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
