# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANE runs text embedding and reranking models on the Apple Neural
Engine (ANE) of Apple Silicon Macs. Models are taken as-is in their
Hugging Face distribution form and compiled locally into Core ML
artifacts that load in seconds and run on the ANE — keeping your GPU
and most of your unified memory free for other work.

> **Status: early development (v0.3, proof of concept).**
> The PoC covers both
> [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)
> (a Japanese ModernBERT embedding model, v0.1) and
> [cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)
> (its cross-encoder reranker, v0.2): both are converted to Core ML
> through a shared pipeline and verified for ANE inference, numerical
> accuracy against the PyTorch FP32 baseline, and latency. v0.3 added
> batch-size-N conversion and a full performance study on an M2 Mac
> mini — sequence-length/batch latency matrix, real-document
> throughput, and head-to-head comparisons against PyTorch on the GPU
> (MPS) and an Infinity_emb deployment. Highlights: up to ~13,600
> effective tokens/s for embeddings (2–3x the MPS GPU baseline), a
> 36-document rerank in ~2.0 s (vs ~4.7 s on MPS), and both models
> resident in ~420 MB, loading in ~0.2 s each. There is no server or
> installable package yet — the planned OpenAI-compatible embeddings /
> rerank server arrives in later milestones.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 or later
- [uv](https://docs.astral.sh/uv/) (for the development environment)

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
