"""Tests for the eager attention that computes ranges of query rows apart.

Three layers:

* How many query rows one score matrix may hold, under the module's own
  limit, under a limit handed in, and under a limit changed at run time.
* The attention the factory builds, checked against the upstream eager
  attention it stands in for: unsplit and split into ranges of every
  shape, with and without a mask and a soft cap, and with the masks and
  keyword arguments an architecture may hand it.
* The traced graph of that attention, which must hold no shape
  arithmetic and no tensor wider than rank 4.

Nothing here builds a model: every attention call receives small random
states and a stand-in for the attention module calling it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers.models.gemma3 import modeling_gemma3

from eeane.compiler.backends import attention_split, decoder_patches

# The upstream implementations, captured at import time (that is, before
# any test can have replaced them). The eager attention is the comparison
# baseline of every test; both are what the autouse fixture restores.
_UPSTREAM_REPEAT_KV = modeling_gemma3.repeat_kv
_UPSTREAM_EAGER_ATTENTION = modeling_gemma3.eager_attention_forward

# Geometry of the attention calls: two query heads per key/value head, so
# the key/value expansion really has to expand something.
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_BATCH = 2

# Largest absolute difference tolerated between an attention computed in
# ranges of query rows and the upstream one; what is left is accumulation
# order.
_SPLIT_TOLERANCE = 1e-6

# Additive value of the masked-out positions in the hand-built masks.
_MASK_FILL = decoder_patches.MASK_FILL_VALUE

# Graph node the Core ML converter cannot fold into a static slice; it is
# what a Python-level division or index on a traced shape lowers to.
_DYNAMIC_SIZE_NODE = "aten::Int"

# Highest tensor rank the traced graph may hold: the Neural Engine
# compiler rejects a subgraph working on anything wider.
_MAX_GRAPH_RANK = 4

# Graph node of the attention softmax: one per range of query rows.
_SOFTMAX_NODE = "aten::softmax"

# Sequence length and score limit of the split comparisons, with the
# number of ranges each limit cuts the query rows into.
_SPLIT_CASES: dict[str, tuple[int, int, int]] = {
    "two-ranges": (6, 19, 2),
    "three-ranges": (6, 13, 3),
    "uneven-ranges": (7, 25, 4),
    "single-rows": (7, 8, 7),
}


def _restore_upstream_symbols() -> None:
    """Point the symbols a test may rebind back at their upstream functions."""
    modeling_gemma3.repeat_kv = _UPSTREAM_REPEAT_KV
    modeling_gemma3.eager_attention_forward = _UPSTREAM_EAGER_ATTENTION


@pytest.fixture(autouse=True)
def _restore_transformers_patches() -> Iterator[None]:
    """Undo any rebinding of the transformers module after every test.

    The attention built here looks the key/value expansion up in the
    module it was built for, and the baseline looks it up in the same
    module, so each test starts and ends on the upstream implementations.
    """
    _restore_upstream_symbols()
    try:
        yield
    finally:
        _restore_upstream_symbols()


def _attention_module() -> SimpleNamespace:
    """Build the stand-in for the attention module calling the attention."""
    return SimpleNamespace(
        head_dim=_HEAD_DIM, num_key_value_groups=_HEADS // _KV_HEADS, training=False
    )


def _row_mask(seq_len: int, num_keys: int | None = None) -> torch.Tensor:
    """Build an additive mask with one row per query, masking differently per row.

    The first batch row hides its last two keys from every query; the
    second hides from each query the keys more than two positions away,
    so that no two query rows see the same keys.
    """
    num_keys = seq_len if num_keys is None else num_keys
    mask = torch.zeros(_BATCH, 1, seq_len, num_keys)
    mask[0, :, :, seq_len - 2 : seq_len] = _MASK_FILL
    queries = torch.arange(seq_len)[:, None]
    keys = torch.arange(num_keys)[None, :]
    mask[1, 0] = torch.where((queries - keys).abs() > 2, _MASK_FILL, 0.0)
    return mask


def _attention_arguments(
    seq_len: int = 6, masked: bool = True, seed: int = 3
) -> tuple[SimpleNamespace, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build the arguments one attention call receives."""
    torch.manual_seed(seed)
    query = torch.randn(_BATCH, _HEADS, seq_len, _HEAD_DIM)
    key = torch.randn(_BATCH, _KV_HEADS, seq_len, _HEAD_DIM)
    value = torch.randn(_BATCH, _KV_HEADS, seq_len, _HEAD_DIM)
    mask = _row_mask(seq_len) if masked else None
    return _attention_module(), query, key, value, mask


def _softmax_count(
    attention: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    **kwargs: Any,
) -> int:
    """Count the softmax nodes one traced attention call records.

    Each range of query rows is attended with a softmax of its own, so
    the count is the number of ranges the call was split into.
    """

    def run(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return attention(module, query, key, value, mask, **kwargs)

    traced = torch.jit.trace(run, (query, key, value), check_trace=False)
    return sum(1 for node in traced.inlined_graph.nodes() if node.kind() == _SOFTMAX_NODE)


def _traced_ranks(traced: torch.jit.ScriptModule) -> list[int]:
    """Return the rank of every tensor a traced graph produces."""
    ranks: list[int] = []
    for node in traced.inlined_graph.nodes():
        for output in node.outputs():
            value_type = output.type()
            if isinstance(value_type, torch.TensorType) and value_type.dim() is not None:
                ranks.append(int(value_type.dim()))
    return ranks


# --- how many query rows one score matrix may hold ---------------------------


def test_the_default_limit_is_two_to_the_twentieth() -> None:
    """A score matrix of 2**20 elements is the first the Neural Engine gets wrong."""
    assert attention_split.ATTENTION_SCORE_ELEMENT_LIMIT == 1 << 20


@pytest.mark.parametrize(
    ("num_queries", "num_keys", "expected"),
    [
        (512, 512, 512),
        # The last sequence length whose score matrix stays below the limit.
        (1023, 1023, 1023),
        # Exactly the limit: it has to be split, in two.
        (1024, 1024, 512),
        (1040, 1040, 520),
        # Half the rows still reach the limit, so they are halved again.
        (2048, 2048, 256),
        (1, 1, 1),
        # Rows cannot be split below one, however many keys there are.
        (1, 10**7, 1),
        (4, 1 << 21, 1),
    ],
)
def test_query_rows_stay_strictly_below_the_default_limit(
    num_queries: int, num_keys: int, expected: int
) -> None:
    """The row count is halved until one score matrix holds fewer than 2**20 elements."""
    rows = attention_split.attention_query_rows(num_queries, num_keys)

    assert rows == expected
    assert rows == 1 or rows * num_keys < attention_split.ATTENTION_SCORE_ELEMENT_LIMIT


@pytest.mark.parametrize(
    ("num_queries", "num_keys", "limit", "expected"),
    [
        (6, 6, 37, 6),
        (6, 6, 36, 3),
        (6, 6, 19, 3),
        (6, 6, 13, 2),
        # Halving rounds up: 7 rows become 4, then 2.
        (7, 7, 25, 2),
        (7, 7, 8, 1),
    ],
)
def test_query_rows_follow_a_given_limit(
    num_queries: int, num_keys: int, limit: int, expected: int
) -> None:
    """A limit handed in is used instead of the module's own."""
    rows = attention_split.attention_query_rows(num_queries, num_keys, limit)

    assert rows == expected
    assert rows == 1 or rows * num_keys < limit


def test_a_given_limit_wins_over_the_modules_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handing in the limit makes the module's value irrelevant."""
    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", 13)

    assert attention_split.attention_query_rows(6, 6, limit=1 << 20) == 6


def test_query_rows_read_the_modules_limit_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a limit handed in, the module's value is read when the rows are asked for."""
    assert attention_split.attention_query_rows(6, 6) == 6

    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", 13)

    assert attention_split.attention_query_rows(6, 6) == 2


# --- the split attention against the upstream one ----------------------------


@pytest.mark.parametrize("masked", [True, False], ids=["masked", "unmasked"])
@pytest.mark.parametrize("softcap", [None, 5.0], ids=["uncapped", "softcapped"])
@pytest.mark.parametrize("scaling", [None, 0.25], ids=["default-scaling", "given-scaling"])
def test_the_unsplit_attention_is_the_upstream_attention(
    masked: bool, softcap: float | None, scaling: float | None
) -> None:
    """Below the limit nothing is split, and the numbers are upstream's, bit for bit."""
    module, query, key, value, mask = _attention_arguments(masked=masked)
    expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(
        module, query, key, value, mask, scaling=scaling, softcap=softcap
    )
    attention = attention_split.split_eager_attention(modeling_gemma3)

    output, weights = attention(module, query, key, value, mask, scaling=scaling, softcap=softcap)

    assert _softmax_count(attention, module, query, key, value, mask) == 1
    assert torch.equal(output, expected_output)
    assert torch.equal(weights, expected_weights)


@pytest.mark.parametrize("masked", [True, False], ids=["masked", "unmasked"])
@pytest.mark.parametrize("softcap", [None, 5.0], ids=["uncapped", "softcapped"])
@pytest.mark.parametrize("case", list(_SPLIT_CASES))
def test_the_split_attention_returns_what_upstream_returns(
    monkeypatch: pytest.MonkeyPatch, case: str, masked: bool, softcap: float | None
) -> None:
    """Both returned tensors keep their shape and their values when the rows are split."""
    seq_len, limit, ranges = _SPLIT_CASES[case]
    module, query, key, value, mask = _attention_arguments(seq_len=seq_len, masked=masked)
    expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(
        module, query, key, value, mask, softcap=softcap
    )
    attention = attention_split.split_eager_attention(modeling_gemma3)
    # Changed after the attention was built: the limit is read per call.
    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", limit)

    output, weights = attention(module, query, key, value, mask, softcap=softcap)

    rows = attention_split.attention_query_rows(seq_len, seq_len)
    assert math.ceil(seq_len / rows) == ranges  # the case is what its name says
    assert _softmax_count(attention, module, query, key, value, mask, softcap=softcap) == ranges
    assert output.shape == expected_output.shape
    assert weights.shape == expected_weights.shape
    assert torch.allclose(output, expected_output, rtol=0, atol=_SPLIT_TOLERANCE)
    assert torch.allclose(weights, expected_weights, rtol=0, atol=_SPLIT_TOLERANCE)


def test_the_split_attention_broadcasts_a_mask_without_query_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mask holding a single row for every query is broadcast to each range, not cut."""
    module, query, key, value, _ = _attention_arguments(seq_len=6, masked=False)
    mask = torch.zeros(_BATCH, 1, 1, 6)
    mask[0, 0, 0, 4:] = _MASK_FILL
    expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(module, query, key, value, mask)
    attention = attention_split.split_eager_attention(modeling_gemma3)
    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", 13)

    output, weights = attention(module, query, key, value, mask)

    assert _softmax_count(attention, module, query, key, value, mask) == 3  # really split
    assert torch.allclose(output, expected_output, rtol=0, atol=_SPLIT_TOLERANCE)
    assert torch.allclose(weights, expected_weights, rtol=0, atol=_SPLIT_TOLERANCE)


def test_the_split_attention_uses_only_the_mask_columns_of_its_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mask wider than the keys is cut to them: its extra columns change nothing."""
    module, query, key, value, mask = _attention_arguments(seq_len=6)
    assert mask is not None
    # Columns that would hide every key, were they read.
    wider = torch.cat([mask, torch.full((_BATCH, 1, 6, 3), _MASK_FILL)], dim=-1)
    expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(module, query, key, value, mask)
    attention = attention_split.split_eager_attention(modeling_gemma3)

    unsplit, unsplit_weights = attention(module, query, key, value, wider)
    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", 13)
    split, split_weights = attention(module, query, key, value, wider)

    assert torch.equal(unsplit, expected_output)
    assert torch.equal(unsplit_weights, expected_weights)
    assert torch.allclose(split, expected_output, rtol=0, atol=_SPLIT_TOLERANCE)
    assert torch.allclose(split_weights, expected_weights, rtol=0, atol=_SPLIT_TOLERANCE)


@pytest.mark.parametrize("limit", [1 << 20, 13], ids=["unsplit", "split"])
def test_the_attention_ignores_a_sliding_window_keyword(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    """Extra keywords an architecture passes, such as its window, are accepted and ignored."""
    module, query, key, value, mask = _attention_arguments(seq_len=6)
    expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(module, query, key, value, mask)
    attention = attention_split.split_eager_attention(modeling_gemma3)
    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", limit)

    output, weights = attention(
        module, query, key, value, mask, dropout=0.0, sliding_window=3, position_ids=None
    )

    assert torch.allclose(output, expected_output, rtol=0, atol=_SPLIT_TOLERANCE)
    assert torch.allclose(weights, expected_weights, rtol=0, atol=_SPLIT_TOLERANCE)


# --- what the factory looks up, and when -------------------------------------


def test_the_key_value_expansion_is_looked_up_at_call_time() -> None:
    """Rebinding the expansion after the attention was built still takes effect."""

    def refuse(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        raise AssertionError("the expansion bound when the attention was built was called")

    calls: list[tuple[tuple[int, ...], int]] = []

    def record(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        calls.append((tuple(hidden_states.shape), n_rep))
        return _UPSTREAM_REPEAT_KV(hidden_states, n_rep)

    modeling_module = SimpleNamespace(repeat_kv=refuse)
    attention = attention_split.split_eager_attention(modeling_module)
    module, query, key, value, mask = _attention_arguments(seq_len=6)
    expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(module, query, key, value, mask)

    modeling_module.repeat_kv = record
    output, weights = attention(module, query, key, value, mask)

    # Once for the keys, once for the values.
    assert calls == [((_BATCH, _KV_HEADS, 6, _HEAD_DIM), _HEADS // _KV_HEADS)] * 2
    assert torch.equal(output, expected_output)
    assert torch.equal(weights, expected_weights)


def test_the_expansion_of_a_real_modeling_module_is_looked_up_at_call_time() -> None:
    """A backend's own replacement of the expansion reaches an attention built before it."""
    attention = attention_split.split_eager_attention(modeling_gemma3)
    module, query, key, value, mask = _attention_arguments(seq_len=6)
    expected_output, _ = _UPSTREAM_EAGER_ATTENTION(module, query, key, value, mask)
    calls: list[int] = []

    def record(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        calls.append(n_rep)
        return decoder_patches.repeat_kv(hidden_states, n_rep)

    modeling_gemma3.repeat_kv = record
    output, _ = attention(module, query, key, value, mask)

    assert calls == [_HEADS // _KV_HEADS] * 2
    assert torch.equal(output, expected_output)


def test_a_given_row_rule_decides_the_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    """A row rule handed to the factory is asked instead of the module's limit."""
    asked: list[tuple[int, int]] = []

    def two_rows(num_queries: int, num_keys: int) -> int:
        asked.append((num_queries, num_keys))
        return 2

    # The module's limit alone would cut every row apart.
    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", 1)
    attention = attention_split.split_eager_attention(modeling_gemma3, query_rows=two_rows)
    module, query, key, value, mask = _attention_arguments(seq_len=6)
    wider = torch.cat([mask, torch.zeros(_BATCH, 1, 6, 2)], dim=-1)
    expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(module, query, key, value, mask)

    output, weights = attention(module, query, key, value, wider)

    # The keys counted are those of the states, not the mask columns.
    assert asked == [(6, 6)]
    assert _softmax_count(attention, module, query, key, value, wider) == 3
    assert torch.allclose(output, expected_output, rtol=0, atol=_SPLIT_TOLERANCE)
    assert torch.allclose(weights, expected_weights, rtol=0, atol=_SPLIT_TOLERANCE)


def test_a_given_row_rule_can_leave_the_rows_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rule granting every row takes the upstream path, whatever the module's limit says."""
    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", 1)
    attention = attention_split.split_eager_attention(
        modeling_gemma3, query_rows=lambda num_queries, num_keys: num_queries
    )
    module, query, key, value, mask = _attention_arguments(seq_len=6)
    expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(module, query, key, value, mask)

    output, weights = attention(module, query, key, value, mask)

    assert _softmax_count(attention, module, query, key, value, mask) == 1
    assert torch.equal(output, expected_output)
    assert torch.equal(weights, expected_weights)


# --- the traced graph --------------------------------------------------------


class _AttentionCaller(torch.nn.Module):
    """Calls an attention function the way an attention module of a model does."""

    def __init__(self, attention: Callable[..., tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Store the attention function and the head geometry it reads."""
        super().__init__()
        self.attention = attention
        self.head_dim = _HEAD_DIM
        self.num_key_value_groups = _HEADS // _KV_HEADS

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend from ``query`` to ``key``/``value`` under ``attention_mask``."""
        return self.attention(self, query, key, value, attention_mask)


@pytest.mark.parametrize(("limit", "ranges"), [(1 << 20, 1), (19, 2), (13, 3), (1, 6)], ids=str)
def test_the_traced_attention_is_static_and_stays_within_rank_four(
    monkeypatch: pytest.MonkeyPatch, limit: int, ranges: int
) -> None:
    """Splitting must add neither shape arithmetic nor a wider tensor to the graph."""
    monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", limit)
    # The rank-4 expansion a backend installs before tracing, so that what
    # is left in the graph is the attention's own doing.
    modeling_module = SimpleNamespace(repeat_kv=decoder_patches.repeat_kv)
    caller = _AttentionCaller(attention_split.split_eager_attention(modeling_module)).eval()
    _, query, key, value, mask = _attention_arguments(seq_len=6)
    assert mask is not None

    with torch.no_grad():
        traced = torch.jit.trace(caller, (query, key, value, mask))
        expected_output, expected_weights = _UPSTREAM_EAGER_ATTENTION(
            caller, query, key, value, mask
        )
        _, other_query, other_key, other_value, _ = _attention_arguments(seq_len=6, seed=9)
        other_mask = _row_mask(6).flip(0)
        replayed = traced(other_query, other_key, other_value, other_mask)
        replay_expected = _UPSTREAM_EAGER_ATTENTION(
            caller, other_query, other_key, other_value, other_mask
        )

    graph = traced.inlined_graph
    assert sum(1 for node in graph.nodes() if node.kind() == _SOFTMAX_NODE) == ranges
    assert _DYNAMIC_SIZE_NODE not in str(graph)
    assert max(_traced_ranks(traced)) <= _MAX_GRAPH_RANK
    output, weights = traced(query, key, value, mask)
    assert torch.allclose(output, expected_output, rtol=0, atol=_SPLIT_TOLERANCE)
    assert torch.allclose(weights, expected_weights, rtol=0, atol=_SPLIT_TOLERANCE)
    # The ranges are constants of the graph; the values of the inputs are not.
    for replayed_tensor, expected_tensor in zip(replayed, replay_expected, strict=True):
        assert torch.allclose(replayed_tensor, expected_tensor, rtol=0, atol=_SPLIT_TOLERANCE)


def test_the_module_lists_its_public_surface() -> None:
    """Every name the module declares public exists."""
    assert set(attention_split.__all__) == {
        "ATTENTION_SCORE_ELEMENT_LIMIT",
        "attention_query_rows",
        "attention_rows",
        "split_eager_attention",
        "static_size",
    }
    assert all(hasattr(attention_split, name) for name in attention_split.__all__)
