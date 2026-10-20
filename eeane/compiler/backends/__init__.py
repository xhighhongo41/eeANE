"""Model-architecture-specific compile backends.

The :class:`~eeane.compiler.backends.base.CompileBackend` protocol in
``base.py`` defines the interface every architecture family implements
(loading, graph patching, tracing and FP32 reference computation); the
per-family modules (``bert.py``, ``modernbert.py``, ``xlm_roberta.py``,
``qwen3.py``, ``gemma3.py``, for the bidirectional embedding models of the
Gemma 3 family, and ``embedding_gemma2.py``, for the text embeddings of
EmbeddingGemma 2) implement it, sharing plumbing (pooling, tokenization,
traceable wrappers) via ``common.py``. The graph rewrites more than one
decoder-style family needs are shared via ``decoder_patches.py`` and
``attention_split.py`` (the eager attention computed over ranges of query
rows), and the power-of-two factors that rescale a model's activations
into the float16 range via ``fp16_ranges.py``.
"""
