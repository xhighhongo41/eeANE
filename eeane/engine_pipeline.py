"""Read-ahead tokenization for the eeANE Core ML engine.

One request is predicted one input at a time under a process-wide
prediction lock. The pieces here let the engine tokenize a request's
next input on a worker thread while the current one is being predicted,
so the tokenization cost of a multi-input request hides behind its
predictions instead of adding to them. Nothing here touches Core ML or
any engine state: :mod:`eeane.engine` supplies the tokenize callable and
the worker pool, and drives the read-ahead as a context manager.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass

import numpy as np

# Threads the engine tokenizes ahead on. One request reads one input
# ahead, so two workers let two requests overlap their tokenization with
# a prediction without the pool ever growing with the load.
_TOKENIZE_WORKERS = 2

# Name prefix of those threads, so they are recognizable in a stack dump.
_TOKENIZE_THREAD_PREFIX = "eeane-tokenize"


@dataclass(frozen=True)
class _TokenizedInput:
    """One request input, tokenized and routed to its bucket.

    Attributes:
        inputs: ``input_ids``/``attention_mask`` arrays of shape
            ``(1, S)``, dtype int32, padded to :attr:`bucket`.
        bucket: Sequence-length bucket the input was routed to, i.e. the
            compiled artifact that has to predict it.
        n_tokens: Token count before truncation, as the request's token
            accounting reports it.
        truncated: Whether the input did not fit into the largest bucket.
    """

    inputs: dict[str, np.ndarray]
    bucket: int
    n_tokens: int
    truncated: bool


class _TokenizeAhead:
    """Hands one request's inputs out tokenized, one prediction ahead.

    A request is predicted one input at a time under the process-wide
    prediction lock, so tokenizing inline leaves the tokenizer idle for
    the whole of every prediction and makes every input pay for its own
    tokenization. Here the tokenization of input ``i + 1`` is started on
    a worker thread *before* input ``i`` is predicted, so every
    tokenization of a request but the first runs while the compute unit
    is busy.

    The read-ahead depth is one: at most one tokenization is ever
    pending, which is all a strictly ordered one-at-a-time prediction
    loop can hide, and it keeps the extra memory a request holds down to
    a single tokenized input. Inputs are handed out in request order and
    each is tokenized exactly once, so a request's results are the ones
    an inline loop would produce, prediction for prediction.

    The worker takes the model's tokenizer lock, exactly as an inline
    tokenization would, and takes no other lock; the request thread holds
    no lock while waiting for a worker. The engine's lock order is
    therefore untouched.

    Instances are used as a context manager by the one thread serving the
    request. Leaving the block gives up whatever is still pending, which
    is what keeps a worker from touching the tokenizer of a model the
    request has already let go of.
    """

    def __init__(
        self,
        tokenize: Callable[[int], _TokenizedInput],
        count: int,
        executor: Executor | None,
    ) -> None:
        """Register a read-ahead that has tokenized nothing yet.

        Args:
            tokenize: Tokenizes the input at the index it is given. It is
                called once per input, from a worker thread while reading
                ahead and from the request's own thread otherwise.
            count: Number of inputs the request carries, at least one.
            executor: Workers the tokenizations run on, or ``None`` to
                tokenize every input inline, which is what a request with
                nothing to read ahead of does.
        """
        self._tokenize = tokenize
        self._count = count
        self._executor = executor
        # Index of the next input to hand out, and of the next one to
        # start tokenizing: they differ by the one pending tokenization.
        self._taken = 0
        self._started = 0
        self._pending: Future[_TokenizedInput] | None = None

    def __enter__(self) -> _TokenizeAhead:
        """Start the first input's tokenization and return the read-ahead."""
        self._start_next()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Finish or give up the pending tokenization, whatever happened."""
        self.close()

    def take(self) -> _TokenizedInput:
        """Return the next input in request order, tokenized.

        Returns:
            The tokenized input, once its tokenization has finished. The
            input after it is started before returning, so it is
            tokenized while the caller predicts this one.

        Raises:
            Exception: Whatever tokenizing this input raised, in the
                caller's thread and at the point the input is needed, so
                a failure reads exactly as an inline tokenization's does.
        """
        index = self._taken
        self._taken += 1
        pending = self._pending
        if pending is None:
            # Nothing was read ahead (no workers, or nothing left to read
            # ahead of): tokenize here, as an inline loop would.
            return self._tokenize(index)
        self._pending = None
        current = pending.result()
        self._start_next()
        return current

    def close(self) -> None:
        """Give up the tokenization that is still pending, if any.

        A tokenization that has not started is cancelled; one that is
        already running is waited for, because it holds the model's
        tokenizer and the request must not let go of the model while a
        worker is still using it. Whatever a given-up tokenization raised
        is dropped: nothing will ever look at its result, and it must not
        mask the reason the request is being torn down.

        Idempotent, so a request can close a read-ahead it has already
        drained.
        """
        pending = self._pending
        self._pending = None
        # Nothing may be started from here on: this is the point after
        # which the request is free to release the model.
        self._started = self._count
        if pending is None:
            return
        if not pending.cancel():
            with contextlib.suppress(BaseException):
                pending.result()

    def _start_next(self) -> None:
        """Start tokenizing the input after the ones already started."""
        if self._executor is None or self._started >= self._count:
            return
        index = self._started
        try:
            self._pending = self._executor.submit(self._tokenize, index)
        except RuntimeError:
            # The workers were stopped (the engine was closed, possibly
            # underneath this request): fall back to tokenizing inline,
            # which serves the request exactly as an unpipelined loop
            # would, just without hiding the tokenization.
            self._executor = None
            return
        self._started = index + 1
