# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANE runs text embedding (and, in the future, reranking) models on the
Apple Neural Engine (ANE) of Apple Silicon Macs. Models are taken
as-is in their Hugging Face distribution form and compiled locally into
Core ML artifacts that load in seconds and run on the ANE — keeping
your GPU and most of your unified memory free for other work.

> **Status: early development (v0.1, proof of concept).**
> The current milestone converts
> [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)
> (a Japanese ModernBERT embedding model) to Core ML and verifies
> ANE inference, numerical accuracy against the PyTorch FP32 baseline,
> and latency. There is no server or installable package yet — the
> planned OpenAI-compatible embeddings / rerank server arrives in
> later milestones.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 or later
- [uv](https://docs.astral.sh/uv/) (for the development environment)

## Trying the PoC (development snapshot)

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync

# Place the model in HF distribution form under models/ruri-v3-310m
# (e.g. download with `huggingface-cli download cl-nagoya/ruri-v3-310m`),
# then:
uv run python poc/convert_embedding.py --seq-len 512   # HF -> .mlmodelc
uv run python poc/verify_accuracy.py --seq-len 512     # accuracy vs FP32
uv run python poc/benchmark_latency.py --seq-len 512 --compute-units CPU_AND_NE --compute-plan
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

The test corpus under `testdata/corpus/` consists of public-domain
literary works from [Aozora Bunko](https://www.aozora.gr.jp/) and is
not covered by the GPL; see `testdata/corpus/README.md`.

---

日本語のREADMEは [README_ja.md](README_ja.md) を参照してください。
