"""Model-architecture-specific compile backends.

The :class:`~eeane.compiler.backends.base.CompileBackend` protocol in
``base.py`` defines the interface every architecture family implements
(loading, graph patching, tracing and FP32 reference computation); the
per-family modules (``bert.py``, ``modernbert.py``, ``xlm_roberta.py``,
``qwen3.py`` and ``gemma3.py``, the last one for the bidirectional
embedding models of the Gemma 3 family) implement it, sharing plumbing
(pooling, tokenization, traceable wrappers) via ``common.py``. The graph
rewrites more than one decoder-style family needs are shared via
``decoder_patches.py``.
"""
