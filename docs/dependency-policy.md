# Dependency policy

eeANE splits its dependencies into two sets and pins the conversion side
tightly. This document records what is pinned, why, and what any version
bump must be accompanied by.

## The two dependency sets

| Set | Packages | Used by |
|---|---|---|
| Runtime (`[project] dependencies`) | fastapi, uvicorn, coremltools, numpy, tokenizers | `eeane serve` / `eeane check-config` |
| `[compile]` extra | torch, transformers, huggingface_hub, sentencepiece | `eeane compile` only |

The server deliberately does **not** depend on torch or transformers: it
loads compiled `.mlmodelc` artifacts with Core ML and tokenizes with a
`tokenizer.json` frozen at compile time (the `tokenizers` library alone).
A unit test enforces that no runtime module imports torch/transformers.

## Pinned versions and why

| Package | Range | Rationale |
|---|---|---|
| Python | >=3.11,<3.13 | Version the project is developed and verified on. |
| torch | >=2.7,<2.8 | coremltools 9.0 officially supports torch 2.7.x; newer torch versions are outside the tested matrix and have broken conversions before. |
| coremltools | >=9.0,<10 | Conversion (`ct.convert`, `torch.jit.trace` route) and runtime loading (`CompiledMLModel`) are verified against 9.0. |
| transformers | >=5.20.1,<5.21 | The series every verified model was re-converted and checked on. eeANE's conversion depends on transformers *internals* that are not a stable API and have changed within 5.x: the ModernBERT patches rebind functions and the attention `forward` of `modeling_modernbert`, the Gemma 3 and Qwen3 patches rebind module-level helpers of their modeling modules, and every encoder backend relies on the framework passing an already built 4-D attention mask through unchanged. |
| huggingface_hub | >=2,<3 | Required range of transformers 5.20 (`>=1.31,<3`), narrowed to the 2.x series the downloads and their error handling are verified with. |
| tokenizers | >=0.22,<0.24 | The compile side gets 0.23 through transformers 5.20 (`>=0.23.1,<0.24`). The runtime accepts 0.22 as well: a `tokenizer.json` frozen with 0.23 loads with 0.22 and produces the same tokens. |
| sentence-transformers (development only) | >=5.2,<7 | Used only to cross-check converted models against their reference implementation; 5.2 is the first release that supports transformers 5. |

The optional `kernels` package, which lets transformers swap some modeling
functions for Hub kernels at run time, is not supported in the conversion
environment: it can bypass the functions eeANE's patches rebind.

### Moving from transformers 4.57 to 5.20

eeANE 2.0 moved conversion from transformers 4.57 to the 5.20 series.
Compiled artifacts do not use transformers at serve time, so artifacts
compiled by eeANE 1.x keep working with the 2.0 server unchanged. Running
`eeane compile` again rebuilds them on its own, because the dependency
versions recorded with each artifact no longer match. To keep compiling
with transformers 4.57, use eeANE 1.7.x.

Under transformers 5 the encoder families build their attention masks
through the framework's shared mask helpers, which record an op the Core
ML converter cannot translate and fill masked positions with a value FP16
cannot hold. Every encoder backend therefore builds its additive mask
inside the traced wrapper, with static-shape operations and a finite fill
value, and hands it to the model in 4-D. The BERT and XLM-RoBERTa
backends rebind no transformers code, but they depend on that
pass-through as much as the patched families depend on the symbols they
rebind — the update rule below applies to every architecture equally: an
unverified dependency bump is an unverified conversion.

## The update rule

**A dependency update is only complete together with a full conversion
re-verification.** Concretely, any change to torch, transformers,
coremltools, numpy, or tokenizers must be accompanied by:

1. `eeane compile` re-run with `--force` for the reference models of
   every supported architecture (ModernBERT: ruri-v3-310m and
   ruri-v3-reranker-310m; BERT: bge-base-en-v1.5, embedding only;
   XLM-RoBERTa: multilingual-e5-base,
   bge-reranker-v2-m3, and bge-m3, the last of which also exercises the
   `--allow-pickle` path; Gemma 3: embeddinggemma-300m, embedding only;
   Qwen3: Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B, which also
   exercise splitting a model into parts),
2. all self-checks passing (accuracy sanity, NE placement, warm latency
   recorded — the self-check is the designed detector for a conversion
   silently degrading),
3. the re-compiled reference models compared with the artifacts compiled
   before the update, on the same inputs: the frozen tokenizers must
   produce the same tokens, and the outputs must agree, or agree with the
   reference implementation at least as closely as before,
4. the full test suite (`tools/check.sh`) passing, and
5. `tools/verify_server.py all` passing against a server running on the
   re-compiled artifacts.

The conversion patches fail loudly (AttributeError at compile time) if a
transformers update removes or renames the patched symbols; the
self-check exists to catch the quiet failures (numerical drift, ANE
placement loss, NaNs).

## What CI does and does not guarantee

GitHub Actions macOS runners are VMs without a usable Apple Neural
Engine. CI runs lint and the unit suite (including conversion-logic tests
on a tiny synthetic model), but **a green CI does not validate a
dependency update**: ANE conversion, placement, and performance checks
run only on a real Apple Silicon machine, via the steps above.
