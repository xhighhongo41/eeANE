"""Hard-coded server settings for eeANE v0.4.

Config-file and CLI support arrive in a later version; until then this
module is the single place where deployment constants live.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HOST = "127.0.0.1"
PORT = 7997

# HuggingFace-format model directories (tokenizer source).
EMBEDDING_MODEL_DIR = REPO_ROOT / "models" / "ruri-v3-310m"
RERANKER_MODEL_DIR = REPO_ROOT / "models" / "ruri-v3-reranker-310m"

_COMPILED_ROOT = REPO_ROOT / "models" / "compiled"
# Fixed-length Core ML artifacts (all batch=1, rank-4 eager graph, macOS13 target).
EMBEDDING_COMPILED = {
    128: _COMPILED_ROOT / "ruri-v3-310m" / "s128_b1_eager_macos13.mlmodelc",
    512: _COMPILED_ROOT / "ruri-v3-310m" / "s512_b1_eager_macos13.mlmodelc",
    1024: _COMPILED_ROOT / "ruri-v3-310m" / "s1024_b1_eager_macos13.mlmodelc",
}
RERANKER_COMPILED = {
    512: _COMPILED_ROOT / "ruri-v3-reranker-310m" / "s512_b1_eager_macos13.mlmodelc",
}

# Model ids reported in API responses.
EMBEDDING_MODEL_ID = "ruri-v3-310m"
RERANKER_MODEL_ID = "ruri-v3-reranker-310m"

# Output tensor names chosen at conversion time (poc/convert_*.py).
EMBEDDING_OUTPUT_NAME = "embedding"
RERANKER_OUTPUT_NAME = "logits"

# L2-normalize embeddings before returning them (matches Infinity_emb's
# behaviour so eeANE is numerically compatible as a drop-in replacement).
NORMALIZE_EMBEDDINGS = True
