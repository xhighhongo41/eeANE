"""Input preparation for the eeANE Core ML engine.

Two decisions one request makes before anything is predicted live here.

First, how its inputs are grouped: an input is normally predicted on its
own, but the inputs that share a bucket a batched artifact is configured
for are predicted together, in groups of
:data:`~eeane.config.BATCH_ARTIFACT_BATCH_SIZE`. Grouping happens within
a single request only -- a request never waits for another one to fill a
group.

Second, how those groups are tokenized: the engine predicts one group at
a time under a process-wide prediction lock, so the read-ahead here
tokenizes the next group on a worker thread while the current one is
being predicted, and the tokenization cost of a multi-input request
hides behind its predictions instead of adding to them.

Nothing here touches Core ML or any engine state: :mod:`eeane.engine`
supplies the tokenize callable and the worker pool, and drives the
read-ahead as a context manager.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from eeane.config import BATCH_ARTIFACT_BATCH_SIZE

# Threads the engine tokenizes ahead on. One request reads one input
# ahead, so two workers let two requests overlap their tokenization with
# a prediction without the pool ever growing with the load.
_TOKENIZE_WORKERS = 2

# Name prefix of those threads, so they are recognizable in a stack dump.
_TOKENIZE_THREAD_PREFIX = "eeane-tokenize"

# What one read-ahead hands out: a single tokenized input, or a whole
# tokenized group of them.
_Prepared = TypeVar("_Prepared")


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


@dataclass(frozen=True)
class _InputRoute:
    """Where one request input goes, decided before anything is predicted.

    Attributes:
        bucket: Sequence-length bucket the input was routed to.
        n_tokens: Token count before truncation, as the request's token
            accounting reports it.
        truncated: Whether the input did not fit into the largest bucket.
    """

    bucket: int
    n_tokens: int
    truncated: bool


@dataclass(frozen=True)
class _InputGroup:
    """Request inputs that are predicted by one call, in one shape.

    Attributes:
        bucket: Sequence-length bucket every input of the group shares,
            i.e. the compiled artifact that has to predict them.
        indices: Positions of those inputs within the request, ascending.
            A group of one is predicted by the bucket's ordinary
            artifact, a larger one by its batched artifact.
    """

    bucket: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class _TokenizedGroup:
    """One input group, tokenized and ready to be predicted.

    Attributes:
        group: The inputs this covers and the bucket they share.
        inputs: ``input_ids``/``attention_mask`` arrays of shape
            ``(len(group.indices), S)``, dtype int32, padded to the
            group's bucket, with one row per index of
            :attr:`_InputGroup.indices`, in that order.
    """

    group: _InputGroup
    inputs: dict[str, np.ndarray]


def _plan_groups(
    routes: Sequence[_InputRoute], batched_buckets: Collection[int]
) -> list[_InputGroup]:
    """Decide which of one request's inputs are predicted together.

    Inputs routed to a bucket a batched artifact is configured for are
    paired up with the next input of that same bucket, in request order,
    up to :data:`~eeane.config.BATCH_ARTIFACT_BATCH_SIZE` per
    group; an input left without a partner is predicted on its own, as is
    every input of every other bucket. Groups come back in the order of
    their first input, so a request's predictions still follow its input
    order as closely as grouping allows, and every index appears in
    exactly one group.

    Args:
        routes: Routing decision of every input, in request order.
        batched_buckets: Buckets a batched artifact is loaded for.

    Returns:
        The groups to predict, ordered by their first input's position.
    """
    groups: list[_InputGroup] = []
    # One open group per bucket: the inputs seen so far that are still
    # waiting for the partners that would fill their group.
    waiting: dict[int, list[int]] = {}
    for index, route in enumerate(routes):
        if route.bucket not in batched_buckets:
            groups.append(_InputGroup(bucket=route.bucket, indices=(index,)))
            continue
        open_group = waiting.setdefault(route.bucket, [])
        open_group.append(index)
        if len(open_group) >= BATCH_ARTIFACT_BATCH_SIZE:
            groups.append(_InputGroup(bucket=route.bucket, indices=tuple(open_group)))
            open_group.clear()
    # A group the request could not fill has no artifact of its shape, so
    # what is left over is predicted one input at a time.
    for bucket, open_group in waiting.items():
        groups += [_InputGroup(bucket=bucket, indices=(index,)) for index in open_group]
    groups.sort(key=lambda group: group.indices[0])
    return groups


class _TokenizeAhead(Generic[_Prepared]):
    """Hands one request's inputs out tokenized, one prediction ahead.

    A request is predicted one input -- or one group of them -- at a time
    under the process-wide prediction lock, so tokenizing inline leaves
    the tokenizer idle for the whole of every prediction and makes every
    input pay for its own tokenization. Here the tokenization of unit
    ``i + 1`` is started on a worker thread *before* unit ``i`` is
    predicted, so every tokenization of a request but the first runs
    while the compute unit is busy.

    The read-ahead depth is one: at most one tokenization is ever
    pending, which is all a strictly ordered one-at-a-time prediction
    loop can hide, and it keeps the extra memory a request holds down to
    a single tokenized unit. Units are handed out in the planned order
    and each is tokenized exactly once, so a request's results are the
    ones an inline loop would produce, prediction for prediction.

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
        tokenize: Callable[[int], _Prepared],
        count: int,
        executor: Executor | None,
    ) -> None:
        """Register a read-ahead that has tokenized nothing yet.

        Args:
            tokenize: Tokenizes the unit at the index it is given. It is
                called once per unit, from a worker thread while reading
                ahead and from the request's own thread otherwise.
            count: Number of units the request is predicted in, at least
                one.
            executor: Workers the tokenizations run on, or ``None`` to
                tokenize every unit inline, which is what a request with
                nothing to read ahead of does.
        """
        self._tokenize = tokenize
        self._count = count
        self._executor = executor
        # Index of the next unit to hand out, and of the next one to
        # start tokenizing: they differ by the one pending tokenization.
        self._taken = 0
        self._started = 0
        self._pending: Future[_Prepared] | None = None

    def __enter__(self) -> _TokenizeAhead[_Prepared]:
        """Start the first unit's tokenization and return the read-ahead."""
        self._start_next()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Finish or give up the pending tokenization, whatever happened."""
        self.close()

    def take(self) -> _Prepared:
        """Return the next unit in the planned order, tokenized.

        Returns:
            The tokenized unit, once its tokenization has finished. The
            unit after it is started before returning, so it is tokenized
            while the caller predicts this one.

        Raises:
            Exception: Whatever tokenizing this unit raised, in the
                caller's thread and at the point the unit is needed, so a
                failure reads exactly as an inline tokenization's does.
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
        """Start tokenizing the unit after the ones already started."""
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
