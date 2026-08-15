"""Model-architecture-specific compile backends.

The :class:`~eeane.compiler.backends.base.CompileBackend` protocol in
``base.py`` defines the interface every architecture family implements
(loading, graph patching, tracing and FP32 reference computation); the
per-family modules (``modernbert.py``, ``xlm_roberta.py``) implement it,
sharing plumbing (pooling, tokenization, traceable wrappers) via
``common.py``.
"""
