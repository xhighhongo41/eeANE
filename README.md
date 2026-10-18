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

> **On macOS 27 or later? Recompile large models with eeANE 1.7.0 or
> later.** macOS 27 lowered the size of model the Neural Engine accepts
> from 2 GiB to 1 GiB, without any error or warning: a larger model
> silently runs on the CPU. The Qwen3 0.6B models and the XLM-RoBERTa
> large-class models are above 1 GiB (the XLM-RoBERTa ones then give NaN
> output). Upgrade eeANE to 1.7.0 or later and recompile them on macOS 27
> with `eeane compile <id>`: they are now split automatically into parts
> that fit. See
> [Large models and the Neural Engine size limit](#large-models-and-the-neural-engine-size-limit).

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
- **Five architecture families supported today**: ModernBERT and
  XLM-RoBERTa (both embedding and cross-encoder reranker models), BERT
  (embedding models only), Qwen3 (decoder-only: embedding models
  that pool the sequence's last token, and generative rerankers that
  score from a yes/no logit pair instead of a classification head), and
  Gemma 3 (bidirectional embedding models only). More are planned.

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

## Large models and the Neural Engine size limit

**The limit.** The Neural Engine accepts a compiled Core ML model only up
to a certain size, judged for each compiled model on its own: about
2 GiB up to macOS 26, and 1 GiB (1,073,741,824 bytes) from macOS 27 on.
A larger model is not rejected: conversion, loading and inference all
succeed and the system logs nothing, but every operation runs on the CPU
instead. The 1 GiB figure was measured on an M2 Mac mini with macOS
27.0.1; Apple has not documented the limit, so it may differ on other
hardware or change again.

**What eeANE does.** Since 1.7.0, when a compiled model is above the
limit of the macOS version `eeane compile` runs on, `eeane compile`
splits the converted model into parts that each fit (with coremltools'
`bisect_model`; a part that is still too large is split again, up to 16
parts). It stores them as a directory `<stem>.parts/`, holding
`segments.json` and `part01.mlmodelc`, `part02.mlmodelc`, ..., instead
of a single `<stem>.mlmodelc`. `eeane serve` loads the parts together
and runs them one after another; the API, the configuration and the
results are the same as for an unsplit model. Every weight keeps its
FP16 precision, and splitting does not change the numbers: the same
weights run split and unsplit on the Neural Engine give bit-identical
outputs.

| Model | Compiled size | Unsplit, on macOS 27 | eeANE 1.7.0 on macOS 27 |
|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B` | 1.11 GiB | Runs on the CPU: correct results, but about 3.5x slower at 128 tokens | Split into 2 parts |
| The seven XLM-RoBERTa large-class models (listed below) | 1.04–1.06 GiB | Runs on the CPU: the output is NaN, and the self-check fails | Split into 2 parts |
| Every other verified model, including the default ruri-v3 pair | under 1 GiB | Unaffected | Compiled exactly as before |

**What to do: upgrade to eeANE 1.7.0 or later and recompile.** Run
`eeane compile` again for each model above 1 GiB, on the Mac that
serves it:

```
eeane compile Qwen/Qwen3-Embedding-0.6B
eeane compile intfloat/multilingual-e5-large
```

No option is needed, and neither is `--force`: artifacts made by an
older eeANE are rebuilt automatically, and so is any artifact (or any
part of a split one) larger than the limit of the current system. So
**after upgrading a Mac from macOS 26 to 27, run `eeane compile <id>`
again** for each model compiled there earlier. (An artifact split on
macOS 27 also runs on macOS 26, since its parts are below both limits.)
The output of `eeane compile` names the limit it uses and reports the
split:

```
NE size limit   : 1 GiB (from the macOS version)
...
s128: the compiled model (1.111 GiB) is above the Neural Engine size limit (1 GiB); splitting it into parts that each fit
s128: split into 2 parts (0.557 GiB, 0.553 GiB) in 7.5 s
```

**Measured results.** On macOS 27.0.1 with eeANE 1.7.0 (M2 Mac mini,
16 GB). Placement is the share of operations placed on the Neural Engine
over all parts; "vs CPU-only" compares with the same artifact run on the
CPU only.

| Model | Bucket | Neural Engine placement | Warm latency | vs CPU-only |
|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | 128 | 99.4% | 20.1 ms | 3.5x |
| Qwen/Qwen3-Embedding-0.6B | 512 | 99.4% | 104.4 ms | 2.6x |
| Qwen/Qwen3-Embedding-0.6B | 1024 | 99.4% | 292.9 ms | 2.4x |
| Qwen/Qwen3-Reranker-0.6B | 512 | 99.3% | 104.2 ms | 2.5x |
| Qwen/Qwen3-Reranker-0.6B | 1024 | 99.3% | 292.0 ms | 2.3x |

Placement and speed are close to what these models showed unsplit on
macOS 26 (99.3–99.4%, 18.76 ms at 128 tokens); each part boundary costs
well under 1 ms (a 4-part split of the same model measured 20.9 ms at
128 tokens, against 20.1 ms for 2 parts). The built-in self-check gives
cosine 0.99994 for the embedding model. Against the reference
implementations, the embedding model's row-wise cosine is 0.9998 or
higher and the reranker's score differs by at most 0.21 in logit space
(the acceptance threshold is 0.3); the top document is the same for all
9 queries, and none of the 567 document pairs whose reference scores
differ by more than 0.2 changes order.

The XLM-RoBERTa large-class models are all split into 2 parts. Warm
latency per bucket; "—" means the model has no such bucket:

| Model | Neural Engine placement | 128 | 512 | 1024 |
|---|---|---|---|---|
| intfloat/multilingual-e5-large | 96.7% | 12.2 ms | 70.3 ms | — |
| intfloat/multilingual-e5-large-instruct | 96.7% | 11.6 ms | 70.0 ms | — |
| BAAI/bge-m3 | 97.8% | 11.6 ms | 73.9 ms | 220.8 ms |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 97.8% | 11.6 ms | 74.1 ms | 222.0 ms |
| BAAI/bge-reranker-v2-m3 | 97.8% | — | 73.8 ms | 220.9 ms |
| BAAI/bge-reranker-large | 97.8% | — | 70.4 ms | — |
| hotchpotch/japanese-bge-reranker-v2-m3-v1 | 97.8% | — | 74.1 ms | 220.8 ms |

Every bucket passes the self-check and the output is finite; placement
and speed match what these models showed on macOS 26 (96.7–97.8%, about
11.6 / 71.5 ms at 128 / 512 tokens). Against the reference
implementations on short sentences in eight languages and on short
pairs, the embedding cosine is 0.99998 or higher, the reranker
probability differs by 0.0014 or less, and the top document is the same
everywhere. On long Japanese passages, multilingual-e5-large-instruct's
cosine goes down to 0.9987 and bge-reranker-large's probability differs
by up to 0.024. That is the precision of these models in FP16 on the
Neural Engine, not an effect of the split: the same weights give
bit-identical outputs split and unsplit.

**Overriding the limit.** `eeane compile --ne-size-limit SIZE` replaces
the limit derived from the macOS version. SIZE is `1GiB`, `2GiB`,
`800MiB` or a number of bytes; decimal units such as `1GB` are refused
on purpose. Use it to compile for a Mac that serves on another macOS
version — for example on macOS 26 for a Mac running macOS 27:

```
eeane compile Qwen/Qwen3-Embedding-0.6B --ne-size-limit 1GiB
```

`--ne-size-limit none` disables splitting: a model above the limit of
the serving Mac then stays a single compiled model and runs on the CPU.

**The server tells you.** At start-up, `eeane serve` checks every
configured model and logs a warning for one that will not run on the
Neural Engine, either because only a low share of its operations was
placed there when it was compiled, or because it is above the limit of
the running macOS. A split model is judged part by part:

```
WARNING eeane.server: model 'Qwen/Qwen3-Embedding-0.6B': the compiled model for bucket(s) [128] will not run on the Neural Engine (only 0.0% of its operations were placed on the Neural Engine when it was compiled; it is 1.11 GiB, above the 1 GiB the Neural Engine accepts on macOS 27 and later); it will run on the CPU instead, which is slower and, for some models, gives invalid results. Run `eeane compile Qwen/Qwen3-Embedding-0.6B` with this version of eeANE to rebuild it: a model above the size limit is split into parts that each fit, but a rebuild does not raise a low placement for every model. See the README section "Large models and the Neural Engine size limit"
```

This example is from a model compiled with `--ne-size-limit none` on
macOS 27. For a split model with a part that is too large, the reason
reads `N of its M compiled parts are above the 1 GiB the Neural Engine
accepts on macOS 27 and later (the largest is X GiB)`. The server still
starts; the warning is the only sign. Recompiling fixes a model that is
above the limit; it does not raise a low placement for every model.

**Compatibility.** Caches written by eeANE 1.7.0 use `model_info.json`
format 4. eeANE 1.6.1 and earlier refuse such a record with an error
asking to upgrade eeANE, since they cannot serve a split model. eeANE
1.7.0 still reads caches written by earlier versions. The macOS 26
(2 GiB) behaviour of 1.7.0 was checked by tests only, not on a macOS 26
machine.

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
Five architecture families are supported: **ModernBERT** and
**XLM-RoBERTa** (both embedding and cross-encoder reranker models),
**BERT** (embedding models only — a BERT cross-encoder reranker is
rejected instead, because the compiled graph would have to pin its
segment ids to zero, which changes the meaning of a query/document pair
for this architecture), and **Qwen3** (decoder-only: embedding models
that pool the sequence's last token, and generative rerankers that score
a query/document pair from two vocabulary logits instead of a
classification head — see below), and **Gemma 3** (bidirectional
embedding models only — see below). `RobertaModel`-architecture models are
routed to the XLM-RoBERTa backend as well — transformers implements
RoBERTa and XLM-RoBERTa as the same encoder, differing only in
vocabulary. More families are planned. For embedding models, every
backend detects the pooling declared by the model directory's
sentence-transformers `1_Pooling/config.json` and compiles the matching
graph — mean or CLS pooling for the three encoder backends, mean
pooling for Gemma 3, and last-token pooling for Qwen3 only, since last-token pooling assumes a
causal (left-to-right) architecture that the encoder backends do not
have; an embedding model that does not declare a supported pooling mode
is rejected with an error rather than compiled on a guess, because an
artifact built with the wrong pooling still looks plausible while
returning vectors with a different meaning. Rerankers are unaffected by
this declaration — their scoring is part of the model's own
classification head (or, for Qwen3's generative reranker, its own
instruction template — see below), not a separate sentence-transformers
module. The compile log names the pooling it detected for each embedding
model. See [Verified models](#verified-models) for the models that have
been run end to end.

A sentence-transformers model directory may also declare a **Dense**
module chain in its `modules.json`: `eeane compile` supports the
`Transformer -> Pooling -> Dense (zero or more) -> Normalize (optional,
trailing)` sequence, baking each declared Dense linear projection
(Identity or Tanh activation) into the compiled graph and following the
output width to match — a model with a Dense module ends up with a
different embedding width than its raw hidden size. A module chain this
sequence cannot describe is rejected with an error before any weights are
read, rather than silently compiling something with a different meaning.
The Normalize module itself is not compiled into the graph — the server
applies L2 normalization instead, according to each model's `normalize`
setting (on by default; see [API](#api)). Dense checkpoints follow the
same safetensors-first policy as the model's main weights (`--allow-pickle`
for `.bin`-only Dense checkpoints; see [Checkpoint
formats](#checkpoint-formats)).

The compiler detects whether a model is an embedding model or a
reranker from its `config.json` architecture name, for the three
encoder backends: a name ending in `ForSequenceClassification` is a
reranker, `...Model` an embedding model. Qwen3 embedding and generative
reranker checkpoints are both published under the same architecture
name, `Qwen3ForCausalLM`, so a `ForCausalLM`-suffixed name is resolved
differently instead: `eeane compile` reads the model directory's
sentence-transformers module declaration (`modules.json`) — a pooling
module means embedding, a scoring module means reranker. A directory
declaring neither is rejected with an error that names `--kind` as the
fix; detection for every other architecture is unchanged. Either way,
the compiler then defaults to buckets 128/512/1024
(embedding) or 512/1024 (reranker), clipped to the model's maximum
sequence length — a model capped at 512 tokens compiles as 128/512
(embedding) or just 512 (reranker), and the compile log names each
bucket it drops; `--buckets 512,2048` compiles a custom set (S2048 is
verified on M2 at ~518 ms/inference). Re-running
skips up-to-date artifacts (`--force` reconverts). After every
conversion a **self-check** verifies accuracy against the FP32 original,
measures how many operations landed on the Neural Engine, and records
warm latency — the printed summary doubles as a compatibility report:
if you run eeANE on hardware we have not verified (M1/M3/M4...), please
paste it into an issue. The accuracy sanity check is evaluated on three
fixed input sets — English, Japanese and Chinese — and a variant passes
as soon as any one set clears the threshold: an input in a language the
model's tokenizer has little or no vocabulary for produces an amplified
fp16-vs-fp32 difference that reflects the input, not the model, so
scoring against a language the model can actually read is what the
self-check needs to judge it fairly. The compile log names each set's
measured result and which one was accepted, and the per-set numbers are
recorded in the artifact metadata alongside the accepted set. The
per-bucket measurements are aggregated into
a calibration record (`model_info.json`) in the cache; buckets whose
self-check failed are dropped from the recommended set that
cache-resolved configs load. The tokenizer is frozen into the artifact
directory and verified to reproduce the original tokenization exactly,
so the server needs neither the original model files nor the
transformers library at run time (see
[docs/dependency-policy.md](docs/dependency-policy.md)).

Qwen3's generative reranker variant carries no classification head:
`eeane compile` embeds the query and document in a fixed chat-style
instruction template, and the compiled graph outputs a single value —
the "yes" vocabulary logit minus the "no" vocabulary logit at the
sequence's final position — instead of a classification score. How to
assemble that template (its preamble, per-pair body, and closing suffix)
is resolved once at compile time from the model's own declaration and
recorded in the compiled artifact; the server reads it back and applies
it to every request. When a query/document pair is too long for the
compiled bucket, only the body is truncated — the preamble and suffix
are always kept intact, since the suffix is where the template asks the
model to write its answer. That single output value is exactly the
argument of the two-way identity `softmax([no, yes])[yes] =
sigmoid(yes - no)`, so it plugs into `/rerank`'s existing sigmoid scoring
with no change in meaning; `raw_scores=true` returns that raw logit
difference, just as it returns a raw logit for the other reranker
architectures. The API itself does not change at all — callers do not
need to know whether a served reranker is generative.

The Gemma 3 backend compiles bidirectional embedding models such as
`google/embeddinggemma-300m` (mean pooling followed by two Dense
projections, 768-dimensional output; L2 normalization is applied by the
server). Generative (causal) Gemma 3, multimodal Gemma 3 and Gemma 3n
models are not supported: `eeane compile` rejects them with an error that
gives the reason. Gemma 3's activations do not fit in float16 as they
are, so the compiler applies function-preserving rewrites to the copy of
the model it converts; see [Measured accuracy and speed
(EmbeddingGemma)](#measured-accuracy-and-speed-embeddinggemma).

### Verified models

Every model below was compiled from its stock Hugging Face
distribution, passed the self-check, and had its output compared
against the reference `sentence-transformers` / `CrossEncoder`
implementation on an M2 Mac. **Buckets** is what `eeane compile`
produces with no `--buckets` flag, after clipping to the model's
maximum sequence length. Models are grouped by the backend their
`config.json` selects — note that a model's name does not always
predict it (`paraphrase-multilingual-mpnet-base-v2` is an
XLM-RoBERTa model, and `multilingual-e5-small` is a BERT one).

Any other model built on one of these five architectures is likely to
work as well; these are simply the ones that have been run end to end.

**ModernBERT**

| Model | Type | Buckets |
|---|---|---|
| cl-nagoya/ruri-v3-30m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-70m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-130m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-310m | embedding | 128/512/1024 |
| hotchpotch/bekko-embedding-v1-a25m | embedding | 128/512/1024 |
| Alibaba-NLP/gte-modernbert-base | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-small-english-r2 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-english-r2 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-97m-multilingual-r2 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-311m-multilingual-r2 | embedding | 128/512/1024 |
| nomic-ai/modernbert-embed-base | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-reranker-310m | reranker | 512/1024 |
| hotchpotch/japanese-reranker-tiny-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-xsmall-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-small-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-base-v2 | reranker | 512/1024 |
| ibm-granite/granite-embedding-reranker-english-r2 | reranker | 512/1024 |

**XLM-RoBERTa**

Models built on the plain RoBERTa architecture are also served by this
backend (transformers implements RoBERTa as the same encoder, differing
only in vocabulary), which is why the granite-embedding r1 English models
below (`RobertaModel` in their `config.json`) sit in this table rather
than BERT's.

| Model | Type | Buckets |
|---|---|---|
| BAAI/bge-m3 <sup>1</sup> | embedding | 128/512/1024 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-107m-multilingual | embedding | 128/512 |
| ibm-granite/granite-embedding-278m-multilingual | embedding | 128/512 |
| ibm-granite/granite-embedding-30m-english | embedding | 128/512 |
| ibm-granite/granite-embedding-125m-english | embedding | 128/512 |
| intfloat/multilingual-e5-base | embedding | 128/512 |
| intfloat/multilingual-e5-large | embedding | 128/512 |
| intfloat/multilingual-e5-large-instruct | embedding | 128/512 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | embedding | 128/512 |
| BAAI/bge-reranker-v2-m3 | reranker | 512/1024 |
| BAAI/bge-reranker-base | reranker | 512 |
| BAAI/bge-reranker-large | reranker | 512 |
| hotchpotch/japanese-reranker-cross-encoder-xsmall-v1 | reranker | 512 |
| hotchpotch/japanese-reranker-cross-encoder-small-v1 | reranker | 512 |
| hotchpotch/japanese-bge-reranker-v2-m3-v1 | reranker | 512/1024 |

The seven large-class models in this table (bge-m3,
snowflake-arctic-embed-l-v2.0, multilingual-e5-large,
multilingual-e5-large-instruct, bge-reranker-v2-m3, bge-reranker-large
and japanese-bge-reranker-v2-m3-v1) are above 1 GiB once compiled, so on
macOS 27 and later `eeane compile` splits each of them into two parts;
they were verified that way on macOS 27.0.1 with eeANE 1.7.0, and on
macOS 26 before. See
[Large models and the Neural Engine size limit](#large-models-and-the-neural-engine-size-limit).

**BERT** (embedding models only)

| Model | Type | Buckets |
|---|---|---|
| BAAI/bge-small-en | embedding | 128/512 |
| BAAI/bge-small-en-v1.5 | embedding | 128/512 |
| BAAI/bge-base-en-v1.5 | embedding | 128/512 |
| BAAI/bge-large-en-v1.5 | embedding | 128/512 |
| BAAI/bge-small-zh-v1.5 | embedding | 128/512 |
| BAAI/bge-base-zh-v1.5 <sup>1</sup> | embedding | 128/512 |
| BAAI/bge-large-zh-v1.5 <sup>1</sup> | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-xs | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-s | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-m | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-m-v1.5 | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-l | embedding | 128/512 |
| intfloat/e5-small-v2 | embedding | 128/512 |
| intfloat/e5-base-v2 | embedding | 128/512 |
| intfloat/e5-large-v2 | embedding | 128/512 |
| intfloat/multilingual-e5-small | embedding | 128/512 |
| thenlper/gte-base | embedding | 128/512 |
| thenlper/gte-large | embedding | 128/512 |
| mixedbread-ai/mxbai-embed-large-v1 | embedding | 128/512 |
| sentence-transformers/all-MiniLM-L6-v2 | embedding | 128/512 |
| sentence-transformers/all-MiniLM-L12-v2 | embedding | 128/512 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | embedding | 128/512 |
| sentence-transformers/paraphrase-MiniLM-L6-v2 | embedding | 128/512 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | embedding | 128/512 |
| MongoDB/mdbr-leaf-mt | embedding | 128/512 |
| MongoDB/mdbr-leaf-ir | embedding | 128/512 |
| sentence-transformers/LaBSE | embedding | 128/512 |

<sup>1</sup> Ships `pytorch_model.bin` only, so compiling it needs
`--allow-pickle` (see [Checkpoint formats](#checkpoint-formats)).

**Qwen3 (decoder-only)**

A generative reranker's score carries the same meaning as any other
reranker's (see About `eeane compile` above) — `/rerank` treats it no
differently once compiled.

| Model | Type | Buckets |
|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | embedding | 128/512/1024 |
| Qwen/Qwen3-Reranker-0.6B | reranker | 512/1024 |

These two models were verified on macOS 26.6.2 and, compiled with eeANE
1.7.0 (split into two parts), on macOS 27.0.1. On macOS 27 or later,
compile them with eeANE 1.7.0 or later; see
[Large models and the Neural Engine size limit](#large-models-and-the-neural-engine-size-limit).

**Gemma 3 (bidirectional embedding models only)**

| Model | Type | Buckets |
|---|---|---|
| google/embeddinggemma-300m <sup>2</sup> | embedding | 128/512/1024 |

<sup>2</sup> Requires accepting the model's terms of use on Hugging Face
before it can be downloaded (see Troubleshooting). eeANE converts models
on your machine and does not distribute model weights; you are
responsible for complying with each model's license and terms of use,
including Gemma's.

### Measured accuracy and speed (EmbeddingGemma)

Measured with `google/embeddinggemma-300m` on an M2 Mac mini (16 GB,
macOS 27.0.1), FP16 on the Neural Engine. The built-in self-check passes
for every default bucket; its measurements are in the table below.

| Bucket | Cosine vs. FP32 (en / ja / zh) | NE placement | Warm latency (median) |
|---|---|---|---|
| 128 | 0.99936 / 0.99934 / 0.99914 | 99.3% | 5.05 ms |
| 512 | 0.99936 / 0.99934 / 0.99926 | 99.2% | 24.98 ms |
| 1024 | 0.99936 / 0.99934 / 0.99926 | 99.2% | 68.94 ms |

**Agreement with the original model.** The reference is
sentence-transformers 5.7.0 `encode` run on the CPU in FP32, one input at
a time; each row is the per-input cosine against the compiled model run
on the Neural Engine.

| Condition | Inputs | Bucket (tokens) | Cosine min | Cosine mean |
|---|---|---|---|---|
| No prefix (short en / ja / zh texts) | 9 | 128 (11–37) | 0.99914 | 0.99938 |
| Query (`prompt_name="query"` vs. the same string concatenated client-side) | 9 | 128 (18–45) | 0.99930 | 0.99945 |
| Document (`prompt_name="document"` vs. client-side concatenation) | 9 | 128 (17–44) | 0.99941 | 0.99957 |
| Japanese paragraphs | 60 | 512 (94–512) | 0.99841 | 0.99867 |
| Long texts | 12 | 512 (302–483) | 0.99845 | 0.99867 |
| Long texts | 12 | 1024 (604–961) | 0.99821 | 0.99843 |
| `dimensions` 512 / 256 / 128 | 9 each | 128 | 0.99920 / 0.99932 / 0.99944 | — |

Running the same conversion steps in FP32 (in PyTorch) agrees with the
reference at cosine 0.9999998 or better, so the differences above come
from the arithmetic precision of FP16 and the Neural Engine, not from
the conversion. The query and document rows show that prepending the
prefix strings on the client gives the same result as the model's own
`prompt_name`.

**Speed.** One input at a time (batch 1), the median of 20 runs after 3
warm-up runs, measured serially on the same Mac mini M2. eeANE's time
covers tokenization, inference and normalization; PyTorch is
sentence-transformers `encode` in FP32 (torch 2.7.1, 4 CPU threads).

| Sequence length | eeANE (Neural Engine) | PyTorch CPU | PyTorch MPS | vs. CPU | vs. MPS |
|---|---|---|---|---|---|
| 128 | 7.56 ms (16,925 tokens/s) | 67.63 ms (1,893) | 82.47 ms (1,552) | 8.9x | 10.9x |
| 512 | 27.27 ms (18,774) | 189.58 ms (2,701) | 136.32 ms (3,756) | 7.0x | 5.0x |
| 1024 | 70.40 ms (14,545) | 510.24 ms (2,007) | 246.21 ms (4,159) | 7.2x | 3.5x |

**How FP16 is handled.** The upstream model card states that this
model's activations do not fit in float16. `eeane compile` applies
function-preserving rewrites to the copy of the model it converts, always
and without any flag:

1. The residual stream is divided by a power of two, because values in the
   later layers exceed float16's maximum.
2. Each layer's MLP output is scaled up by a power of two, because it
   becomes too small in the later layers and loses precision on the
   Neural Engine.
3. From sequence length 1024 up, attention is computed in chunks of the
   query range, because a single head's score matrix reaching 2^20
   elements gives corrupted results on the Neural Engine only.

The factors are measured on short texts at conversion time and recorded
in the artifact metadata.

### Checkpoint formats

`eeane compile` accepts safetensors checkpoints by default. A Hugging
Face repo id or local model directory that ships `pytorch_model.bin`
weights only (no `.safetensors` file) is rejected with a clear error,
whether the source is a Hub download or a local directory. Pass
`--allow-pickle` to opt into pickle-based `.bin` weights instead:
`eeane compile` then forces transformers to load them with
`torch.load(weights_only=True)`, and logs a WARNING. `weights_only=True`
reduces but does not eliminate the risk of loading a pickle file —
bypasses have been found before (e.g. CVE-2026-24747, fixed in torch
2.10.0), and the torch version `eeane compile` depends on is pinned,
for compatibility with the Core ML conversion toolchain, to a release
that predates that fix (see
[docs/dependency-policy.md](docs/dependency-policy.md)). Only use
`--allow-pickle` with checkpoints from publishers you trust. When a
repository does have safetensors, `--allow-pickle` changes nothing:
safetensors are always preferred, and the `.bin` files are never
downloaded. BAAI/bge-m3, for example, ships `pytorch_model.bin` only,
so compiling it needs the flag:

```sh
eeane compile BAAI/bge-m3 --allow-pickle
```

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
  (`input` as string or list, `encoding_format` `float`/`base64`,
  optional `dimensions`); embeddings are L2-normalized by default
  (per-model `normalize`)
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

The optional `dimensions` field of an embeddings request (OpenAI-compatible)
truncates each returned embedding to its first `dimensions` components
before re-normalizing (when normalization is enabled), returning a
smaller vector without re-running inference. It is meaningful for models
trained with a Matryoshka representation learning (MRL) objective, whose
embeddings stay meaningful at any prefix length; requesting more
dimensions than the model's embedding width gets a 400.

eeANE never modifies request text — `input` and `query` reach the
tokenizer exactly as sent. Some embedding models expect an instruction
string prepended to the search query only (never to documents); when a
served model is trained this way, adding that string is the caller's
responsibility. Qwen3-Embedding-0.6B, for example, publishes this
format:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{the actual query text}
```

EmbeddingGemma is designed the same way, with a prefix on documents as
well. For retrieval, put `task: search result | query: ` before a search
query and `title: none | text: ` before a document, in each case ahead of
the text itself (the model card lists the formats for other tasks).

From [Open WebUI](https://github.com/open-webui/open-webui) (v0.6.0 or
later), set the `RAG_EMBEDDING_QUERY_PREFIX` environment variable to the
instruction string so it is added to queries only, and leave
`RAG_EMBEDDING_CONTENT_PREFIX` (documents) empty. This has been checked
against letting `sentence-transformers` build the same embedding through
the model's own prompt feature instead: the two agree at cosine 0.9999.
For EmbeddingGemma, set `RAG_EMBEDDING_QUERY_PREFIX` to
`task: search result | query: ` and `RAG_EMBEDDING_CONTENT_PREFIX` to
`title: none | text: ` (the query and document rows in Measured accuracy
and speed show that client-side concatenation matches the model's own
prompts).

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
  request. On macOS 27 or later, a model above 1 GiB that was not
  split (compiled by an eeANE older than 1.7.0, on an older macOS, or
  with `--ne-size-limit none`) is the usual cause: recompile it with
  `eeane compile <id>`; see
  [Large models and the Neural Engine size limit](#large-models-and-the-neural-engine-size-limit).
- **A start-up warning says a model "will not run on the Neural
  Engine"**, or a model became several times slower after a macOS
  upgrade: recompile it with `eeane compile <id>` on the current
  system. Since 1.7.0, a model above the size limit is split into parts
  that each fit; a model compiled with `--ne-size-limit none` is not
  split and, above the limit, runs on the CPU. See
  [Large models and the Neural Engine size limit](#large-models-and-the-neural-engine-size-limit).
- **A request occasionally takes much longer than usual**: the first
  request after a model was idle past `keep_alive` pays the on-demand
  reload (typically well under a second), and the very first request
  after `eeane compile` pays the one-time Neural Engine cache build
  (tens of seconds). Both are expected; use `load_policy = "resident"`
  if you need to avoid even the sub-second reload.
- **`eeane compile` fails with `no .safetensors weights are available
  ...`**: the model ships a pickle-based `pytorch_model.bin` checkpoint
  only, and `eeane compile` requires safetensors by default. The error
  message already names the fix: pass `--allow-pickle` (see Checkpoint
  formats above) — only for checkpoints from publishers you trust.
- **`eeane compile` fails with `... must declare its pooling in the
  sentence-transformers '1_Pooling/config.json' ...`**:
  compiling an embedding model needs a `1_Pooling/config.json`
  sentence-transformers pooling declaration; when it is missing,
  unreadable, or names an unsupported mode (anything other than mean or
  CLS), `eeane compile` stops with an error instead of guessing. Check
  that the model is actually distributed in sentence-transformers form
  — for a local model directory, check that the `1_Pooling/config.json`
  the publisher ships alongside it is present. eeANE does not fall back
  to assuming mean pooling: a wrongly-pooled artifact would silently
  return vectors with a different meaning.
- **`eeane compile` fails with `cannot tell whether ... is an embedding
  model or a reranker ...`**: this only happens for a
  `ForCausalLM`-architecture model directory (Qwen3, for example) that
  declares neither a sentence-transformers pooling module nor a scoring
  module in its `modules.json` — the only signal `eeane compile` has for
  telling that architecture's embedding and generative-reranker
  checkpoints apart. Pass `--kind embedding` or `--kind reranker`
  explicitly to compile it anyway.
- **`eeane compile` cannot download a model that requires accepting
  terms (for example `google/embeddinggemma-300m`)**: the model's
  Hugging Face page asks you to agree to its terms of use before the
  files can be downloaded. Accept them on the model page, log in with
  `hf auth login`, and run `eeane compile` again. eeANE does not handle
  your token itself; that is left to the Hugging Face libraries. When a
  download fails because the terms are not accepted or you are not logged
  in, the error now carries a line saying what to do.

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
- **BAAI/bge-m3 is dense-only**: eeANE compiles and serves bge-m3's
  dense embedding output; its separate sparse and multi-vector
  (ColBERT-style) representations are additional weight files that
  `eeane compile` does not fetch or expose.
- **BERT cross-encoder rerankers are not supported**: see About `eeane
  compile` above — pinning the compiled graph's segment ids to zero
  would change the meaning of a query/document pair for this
  architecture. BERT embedding models are unaffected.
- **Accuracy on out-of-vocabulary input**: compiled models run in
  fp16. For input a model's tokenizer cannot represent well — feeding
  English text to a Chinese-only model, say — the rounding difference
  against an fp32 reference grows noticeably, because the input is
  already far outside what the model was trained on. `eeane compile`'s
  own self-check sidesteps this by scoring three fixed language sets
  (English, Japanese, Chinese) and accepting the variant on whichever
  set the model actually has vocabulary for (see About `eeane compile`
  above), but the same caution still applies whenever you compare a
  model's fp16 and fp32 output on input outside its intended languages
  yourself. Within a model's intended languages the agreement is far
  tighter (cosine ≥ 0.9999 on every model listed above except
  EmbeddingGemma, whose measurements are in Measured accuracy and speed).
- **Memory scales with buckets loaded, not duplicated per bucket**:
  compiled weights are memory-mapped, so serving several buckets of the
  same model does not multiply its resident memory by the bucket count.
  Measured on a 0.6B-class embedding model: loading all three buckets
  added about 150 MB of physical footprint to the server process (313 MB
  unloaded -> 456 MB with 128/512/1024 all loaded); an eight-model
  configuration held workers at 592 MB (peak 730 MB).
- **Neural Engine model-size ceiling**: a compiled model above a fixed
  size is not placed on the Neural Engine at all and everything falls
  back to the CPU — conversion, loading and inference all still succeed.
  The ceiling is about 2 GiB up to macOS 26 and **1 GiB from macOS 27
  on**. Since 1.7.0, `eeane compile` splits a model above it into parts
  that each fit; a model compiled by an older eeANE or on an older macOS
  must be recompiled, and with `--ne-size-limit none` a model above the
  ceiling is not split and runs on the CPU. See
  [Large models and the Neural Engine size limit](#large-models-and-the-neural-engine-size-limit).
  The 4B and 8B models in the Qwen3 family are far above even 2 GiB and
  remain unsupported: they have not been tried with splitting, and the
  memory converting and serving them needs on a 16 GB machine is
  untested.
- **EmbeddingGemma at bucket 2048**: the model's maximum length is 2048,
  and `--buckets 2048` compiles, but in measurements here that bucket is
  not placed on the Neural Engine and runs on the CPU (0% placement,
  about 3.3 s per input). Stay with the default 128/512/1024.

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

### Trying the decoder-model study

Decoder-only (causal language model) architectures are now handled by
the regular `eeane compile` — see Qwen3 in About `eeane compile` and
Verified models above. The `poc_qwen/` scripts remain in the repository
as the research record behind that support, and are still runnable. One
part of that record has no equivalent in the main product: probing, with
synthetic models of increasing size (no large checkpoints downloaded),
how large a model can get before the Neural Engine stops accepting it at
all — see Known limitations above for what that probe found:

```sh
# Convert; the model is downloaded from the Hub on first use
uv run python poc_qwen/convert_embedding.py --seq-len 128

# Check the result against the FP32 and sentence-transformers references
uv run python poc_qwen/verify_accuracy.py \
    --mlmodelc models/compiled/qwen3-embedding-0.6b/s128_b1_fp16_macos13.mlmodelc --seq-len 128

# Latency plus which compute unit each operation landed on
uv run python poc_qwen/benchmark_latency.py \
    --mlmodelc models/compiled/qwen3-embedding-0.6b/s128_b1_fp16_macos13.mlmodelc \
    --seq-len 128 --compute-plan

# Where Neural Engine placement breaks down, using synthetic models of
# increasing size (no large checkpoints are downloaded)
uv run python poc_qwen/size_sweep.py

# The generative reranker variant, which scores a pair from the yes/no
# logits at the final position instead of a classification head
uv run python poc_qwen/convert_reranker.py --seq-len 256
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
| 1.7.0 | A compiled model above the Neural Engine size limit is split automatically into parts that each fit, which `eeane serve` runs one after another (same API, configuration and results; every weight stays FP16); the limit is decided by the macOS version `eeane compile` runs on (2 GiB up to macOS 26, 1 GiB from macOS 27) and can be overridden with `--ne-size-limit`; the seven XLM-RoBERTa large-class models become usable on macOS 27; Qwen3 models no longer need their embedding table stored as 8-bit integers (that workaround is removed); the start-up warning judges a split model part by part; `model_info.json` format 4, which eeANE 1.6.1 and earlier cannot read |
| 1.6.1 | Qwen3 models run on the Neural Engine again on macOS 27, where the size of model it accepts dropped from 2 GiB to 1 GiB: when a Qwen3 model compiled on macOS 27 or later exceeds the limit, `eeane compile` stores its vocabulary embedding table as 8-bit integers and the compiled model fits again (placement, speed and accuracy match macOS 26); `eeane serve` warns at start-up about a model that will not run on the Neural Engine; XLM-RoBERTa large-class models are documented as not usable on macOS 27 for now |
| 1.6.0 | Gemma 3 (bidirectional embedding models) becomes eeANE's fifth supported architecture family, compiled through the regular `eeane compile` path and served as-is; generative, multimodal and Gemma 3n variants are rejected with a reason; function-preserving rewrites keep the model within the FP16 range; one more verified model (google/embeddinggemma-300m; 62 -> 63); fetch failures for models that require accepting terms now say what to do |
| 1.5.0 | Qwen3 (decoder-only) becomes eeANE's fourth supported architecture family, compiled and served through the regular `eeane compile`/`eeane serve` path: last-token pooling for embedding models, and a generative reranker scored from a yes/no logit pair instead of a classification head; model-kind detection for `ForCausalLM` architectures now reads the model's sentence-transformers module declaration instead of the architecture name alone; two verified models (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B) |
| 1.4.5 | Adds `poc_qwen/`, a study of whether decoder-only (causal LM) embedding models run on the Neural Engine, and of how large a model can get before it stops being accepted there; no engine changes |
| 1.4.0 | Compile self-check now scores three fixed language sets (English, Japanese, Chinese) and accepts whichever clears the threshold, instead of one fixed set that could fail on a model with different vocabulary; support for sentence-transformers Dense projection modules (`Transformer -> Pooling -> Dense -> Normalize`); `RobertaModel`-architecture models now route to the XLM-RoBERTa backend; OpenAI-compatible `dimensions` parameter on `/v1/embeddings`; nine more verified models (51 -> 60) |
| 1.3.0 | ModernBERT backend detects mean/CLS pooling from the model's sentence-transformers declaration instead of compiling mean pooling only, so CLS-pooling ModernBERT embedding models (e.g. the granite-embedding-*-r2 family) now compile correctly; the resolved pooling is recorded in the compile log and artifact metadata; five more verified models (gte-modernbert-base and four granite-embedding-*-r2 models) |
| 1.2.0 | 35 more verified models across all three backends (granite, Snowflake Arctic Embed, GTE, mxbai, MiniLM, e5, small ruri-v3, Chinese bge v1.5, Japanese rerankers) and a Verified models table; no engine changes |
| 1.1.0 | BERT embedding backend; six more verified BAAI/bge models (bge-m3, bge-reranker-base/large, bge-small/base/large-en-v1.5); `--allow-pickle` opt-in for pickle-based checkpoints; expanded PyPI metadata |
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
