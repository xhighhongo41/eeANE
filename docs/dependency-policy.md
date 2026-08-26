# Dependency policy

eeANE splits its dependencies into two sets and pins the conversion side
tightly. This document records what is pinned, why, and what any version
bump must be accompanied by.

## The two dependency sets

| Set | Packages | Used by |
|---|---|---|
| Runtime (`[project] dependencies`) | fastapi, uvicorn, coremltools, numpy, tokenizers | `eeane serve` / `eeane check-config` |
| `[compile]` extra | torch, transformers, sentencepiece (huggingface_hub comes with transformers) | `eeane compile` only |

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
| transformers | >=4.57,<4.58 | 4.48–4.55 fall into a period of known ModernBERT bugs. More importantly, eeANE's conversion patches (`rotate_half`, the rank-4 eager-attention rewrite) target the *internal* implementation in `transformers.models.modernbert.modeling_modernbert`, which is only verified for the 4.57 series. |
| tokenizers | >=0.22,<0.24 | Must stay co-installable with transformers 4.57.x, which requires `tokenizers<=0.23.0`. |
| huggingface_hub | (transitive) | transformers 4.57.x requires `<1.0`; installing hub 1.x alongside it breaks transformers at import time. |

Note that transformers 5.x is a different world: it requires
huggingface_hub >= 1.0 and redesigned its tokenizer backend. Migrating to
it is a dedicated, verified task — never a casual bump.

The XLM-RoBERTa backend uses stock transformers modeling code with no
graph patches, so it is less sensitive to transformers internals than
the ModernBERT backend — but the update rule below applies to every
architecture equally: an unverified dependency bump is an unverified
conversion.

## The update rule

**A dependency update is only complete together with a full conversion
re-verification.** Concretely, any change to torch, transformers,
coremltools, numpy, or tokenizers must be accompanied by:

1. `eeane compile` re-run with `--force` for the reference models of
   every supported architecture (ModernBERT: ruri-v3-310m and
   ruri-v3-reranker-310m; BERT: bge-base-en-v1.5, embedding only;
   XLM-RoBERTa: multilingual-e5-base,
   bge-reranker-v2-m3, and bge-m3, the last of which also exercises the
   `--allow-pickle` path),
2. all self-checks passing (accuracy sanity, NE placement, warm latency
   recorded — the self-check is the designed detector for a conversion
   silently degrading),
3. the full test suite (`tools/check.sh`) passing, and
4. `tools/verify_server.py all` passing against a server running on the
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
