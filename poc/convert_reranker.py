"""Convert the ruri-v3-reranker-310m cross-encoder to a Core ML program.

Pipeline (see 開発資料/v0.2実装計画.md §4.1-§4.3):
HF model -> RerankerWrapper (raw logits) -> torch.jit.trace
-> ct.convert (mlprogram, fixed (B, S) int32 inputs) -> .mlpackage
-> `xcrun coremlcompiler` -> .mlmodelc -> sanity check on CPU_AND_NE.

The graph outputs raw logits; sigmoid is applied outside the graph in Python
post-processing (§2.2). The batch size B is fixed at conversion time
(v0.3実装計画.md §4.2); ``--batch 1`` reproduces the v0.2 artifacts and
their naming.

Usage:
    uv run python poc/convert_reranker.py --seq-len 512 --batch 4
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/convert_reranker.py` to import the poc package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc.common import (  # noqa: E402
    DEFAULT_RERANKER_DIR,
    load_reranker_torch_model,
    load_tokenizer,
    score_pytorch,
    sigmoid_np,
    tokenize_pairs,
)
from poc.convert_common import (  # noqa: E402
    build_versions_info,
    compile_model,
    convert_model,
    patch_eager_attention_rank4,
    patch_mask_fill_value,
    patch_rotate_half,
    resolve_output_key,
    trace_model,
)

# Short Japanese (query, document) pair used as the example input for
# torch.jit.trace (§5 T4).
TRACE_EXAMPLE_PAIR: tuple[str, str] = (
    "これは変換用のサンプル質問です。",
    "これは変換用のサンプル文書です。",
)

# Fixed sanity-check pairs (§5 T4): relevant / irrelevant / partially related.
SANITY_PAIRS: list[tuple[str, str]] = [
    # Relevant pair
    (
        "日本の首都はどこですか。",
        "東京は日本の首都であり、政治と経済の中心地として発展してきた都市である。",
    ),
    # Irrelevant pair
    (
        "日本の首都はどこですか。",
        "機械学習モデルを専用のアクセラレータ上で動かすと、消費電力を抑えつつ高い推論スループットを得られる場合がある。",
    ),
    # Partially related pair
    (
        "機械学習の推論を高速化する方法",
        "毎朝のコーヒーにはカフェインが含まれており集中力を高めてくれる。",
    ),
]

# Indices into SANITY_PAIRS used by the ordering check (§5 T4 condition (c)).
RELEVANT_PAIR_INDEX = 0
IRRELEVANT_PAIR_INDEX = 1

# Maximum tolerated |sigmoid(coreml) - sigmoid(fp32)| (§5 T4 condition (b)).
SANITY_SIGMOID_TOLERANCE = 0.02

# Filler pair for the rows that pad the last sanity batch when the number of
# sanity pairs is not a multiple of B. The empty strings encode to special
# tokens only, so the row still has a non-empty attention mask (a fully
# masked row would risk NaN, v0.3実装計画.md §4.2).
BATCH_PADDING_PAIR: tuple[str, str] = ("", "")

# Maximum tolerated |logit(row k) - logit(row 0)| between rows of a batch
# holding the same pair (batch consistency check, v0.3実装計画.md §4.2 / R1).
BATCH_CONSISTENCY_LOGIT_TOLERANCE = 0.01


class RerankerWrapper(torch.nn.Module):
    """Wraps ModernBertForSequenceClassification and exposes raw logits.

    The Core ML graph reproduces the HF forward as-is (CLS pooling +
    classification head, logits output). Sigmoid is applied outside the
    graph in Python post-processing (see v0.2実装計画.md §2.2).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the classification model.

        Args:
            model: ModernBertForSequenceClassification loaded in eval/FP32
                mode with ``config.return_dict = False``.
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute raw relevance logits.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            Raw logits, shape (B, 1).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]  # logits (B, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Convert ruri-v3-reranker-310m to a Core ML program."
    )
    parser.add_argument("--seq-len", type=int, required=True, help="Fixed sequence length S.")
    parser.add_argument("--batch", type=int, default=1, help="Fixed batch size B (default: 1).")
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_RERANKER_DIR, help="Local HF model directory."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_REPO_ROOT / "models" / "compiled",
        help="Root directory for conversion artifacts.",
    )
    parser.add_argument(
        "--attn", choices=["eager", "sdpa"], default="eager", help="Attention implementation."
    )
    parser.add_argument(
        "--target",
        choices=["macos13", "macos15"],
        default="macos13",
        help="Minimum deployment target.",
    )
    parser.add_argument(
        "--precision", choices=["fp16", "fp32"], default="fp16", help="Compute precision."
    )
    parser.add_argument(
        "--mask-fill-value",
        type=float,
        default=None,
        help=(
            "Experimental (§4.8 C2-3): replace the attention mask fill value "
            "(torch.finfo.min) with this value, e.g. -30000. Disabled by default."
        ),
    )
    return parser.parse_args(argv)


def build_stem(args: argparse.Namespace) -> str:
    """Build the artifact base name, e.g. ``s512_b4_eager_macos13`` (§4.1).

    ``--batch 1`` yields the same name as the v0.2 artifacts.
    """
    stem = f"s{args.seq_len}_b{args.batch}_{args.attn}_{args.target}"
    if args.precision == "fp32":
        stem += "_fp32"
    if args.mask_fill_value is not None:
        # Non-default masking is an experiment; never overwrite the baseline artifact.
        stem += "_maskfill"
    return stem


def _fill_batch(rows: np.ndarray, padding_row: np.ndarray, batch_size: int) -> np.ndarray:
    """Pad a partial input chunk up to ``batch_size`` rows.

    Args:
        rows: Chunk of tokenized rows, shape (n, S) with ``n <= batch_size``.
        padding_row: Filler row, shape (1, S).
        batch_size: Fixed batch size B expected by the model.

    Returns:
        Array of shape (batch_size, S); ``rows`` itself when already full.
    """
    missing = batch_size - rows.shape[0]
    if missing <= 0:
        return rows
    return np.concatenate([rows, np.repeat(padding_row, missing, axis=0)], axis=0)


def _predict_rows(
    compiled: ct.models.CompiledMLModel,
    tokens: dict[str, np.ndarray],
    padding: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[str, np.ndarray]:
    """Score all tokenized rows with a model whose batch size is fixed to B.

    Rows are grouped into chunks of ``batch_size``; a trailing partial chunk
    is padded with :data:`BATCH_PADDING_PAIR` rows whose outputs are dropped.

    Args:
        compiled: Loaded compiled model.
        tokens: Tokenized rows, each value of shape (N, S).
        padding: Tokenized filler row, each value of shape (1, S).
        batch_size: Fixed batch size B of the model.

    Returns:
        Tuple of the resolved output key and the raw logits, shape (N,).

    Raises:
        ValueError: If ``tokens`` holds no rows.
    """
    n_rows = int(tokens["input_ids"].shape[0])
    if n_rows == 0:
        raise ValueError("no rows to predict")
    output_key = ""
    values: list[float] = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        prediction = compiled.predict(
            {
                key: _fill_batch(tokens[key][start:end], padding[key], batch_size)
                for key in ("input_ids", "attention_mask")
            }
        )
        output_key = output_key or resolve_output_key(prediction, "logits")
        # The graph output is (B, 1); flatten to one scalar logit per row.
        logits = np.asarray(prediction[output_key], dtype=np.float32).reshape(-1)
        values.extend(float(v) for v in logits[: end - start])
    return output_key, np.asarray(values, dtype=np.float32)


def check_batch_consistency(
    compiled: ct.models.CompiledMLModel,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    output_key: str,
) -> dict[str, Any]:
    """Verify that rows of one batch do not influence each other (R1).

    Predicts a single batch whose B rows all hold ``SANITY_PAIRS[0]`` and
    compares row 0 with every other row.

    Args:
        compiled: Loaded compiled model.
        tokenizer: Tokenizer returned by ``load_tokenizer``.
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B of the model (must be > 1).
        output_key: Output key resolved by :func:`_predict_rows`.

    Returns:
        Dict with the per-row absolute logit differences against row 0,
        their min/max, the tolerance, and the pass/fail flag.
    """
    tokens = tokenize_pairs(tokenizer, [SANITY_PAIRS[0]] * batch_size, seq_len)
    prediction = compiled.predict(
        {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
    )
    logits = np.asarray(prediction[output_key], dtype=np.float32).reshape(-1)
    # Compare row 0 against rows 1..B-1 of the very same prediction.
    abs_diff = np.abs(logits[1:] - logits[0])
    return {
        "pair_index": 0,
        "logit_per_row": [float(v) for v in logits],
        "logit_abs_diff_per_row": [float(v) for v in abs_diff],
        "logit_abs_diff_min": float(abs_diff.min()),
        "logit_abs_diff_max": float(abs_diff.max()),
        "logit_tolerance": BATCH_CONSISTENCY_LOGIT_TOLERANCE,
        "passed": bool(np.isfinite(logits).all())
        and bool(abs_diff.max() <= BATCH_CONSISTENCY_LOGIT_TOLERANCE),
    }


def run_sanity_check(
    mlmodelc_path: Path, model_dir: Path, seq_len: int, batch_size: int = 1
) -> dict[str, Any]:
    """Run the Core ML model on CPU_AND_NE and compare with the FP32 baseline.

    Checks the three conditions of §5 T4: (a) finite outputs, (b) sigmoid
    scores within :data:`SANITY_SIGMOID_TOLERANCE` of the FP32 baseline, and
    (c) the relevant pair outranks the irrelevant pair on both back ends.
    For B>1 the batch consistency check of R1 is added as condition (d).

    Args:
        mlmodelc_path: Compiled model directory.
        model_dir: Local HF model directory (for the baseline and tokenizer).
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B of the compiled model.

    Returns:
        Dict with the output key, the measured Core ML / FP32 logits and
        sigmoid scores, the batch consistency result (``None`` for B=1), and
        the pass/fail flags for (a), (b), (c) and (d).
    """
    tokenizer = load_tokenizer(model_dir)
    tokens = tokenize_pairs(tokenizer, SANITY_PAIRS, seq_len)
    padding = tokenize_pairs(tokenizer, [BATCH_PADDING_PAIR], seq_len)
    compiled = ct.models.CompiledMLModel(
        str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    output_key, coreml_logits = _predict_rows(compiled, tokens, padding, batch_size)
    consistency = (
        check_batch_consistency(compiled, tokenizer, seq_len, batch_size, output_key)
        if batch_size > 1
        else None
    )

    # FP32 reference: sdpa attention, tokenization shared via poc.common.
    baseline_model = load_reranker_torch_model(model_dir, attn="sdpa")
    fp32_logits = score_pytorch(baseline_model, tokenizer, SANITY_PAIRS, seq_len)
    del baseline_model
    gc.collect()

    coreml_scores = sigmoid_np(coreml_logits)
    fp32_scores = sigmoid_np(fp32_logits)
    # NaN propagates through the diff/comparisons below, so a non-finite
    # output can never satisfy (b) or (c) either.
    abs_diff = np.abs(coreml_scores - fp32_scores)
    finite = bool(np.isfinite(coreml_logits).all() and np.isfinite(fp32_logits).all())
    ordering_coreml = bool(
        coreml_scores[RELEVANT_PAIR_INDEX] > coreml_scores[IRRELEVANT_PAIR_INDEX]
    )
    ordering_fp32 = bool(fp32_scores[RELEVANT_PAIR_INDEX] > fp32_scores[IRRELEVANT_PAIR_INDEX])
    max_abs_diff = float(abs_diff.max())
    return {
        "output_key": output_key,
        "compute_units": "CPU_AND_NE",
        "batch_size": batch_size,
        "coreml_logits": [float(v) for v in coreml_logits],
        "fp32_logits": [float(v) for v in fp32_logits],
        "coreml_scores": [float(v) for v in coreml_scores],
        "fp32_scores": [float(v) for v in fp32_scores],
        "sigmoid_abs_diff": [float(v) for v in abs_diff],
        "sigmoid_max_abs_diff": max_abs_diff,
        "sigmoid_tolerance": SANITY_SIGMOID_TOLERANCE,
        "ordering_ok_coreml": ordering_coreml,
        "ordering_ok_fp32": ordering_fp32,
        "batch_consistency": consistency,
        "finite": finite,
        "passed": (
            finite
            and max_abs_diff <= SANITY_SIGMOID_TOLERANCE
            and ordering_coreml
            and ordering_fp32
            and (consistency is None or bool(consistency["passed"]))
        ),
    }


def build_metadata(
    args: argparse.Namespace,
    mlpackage_path: Path,
    mlmodelc_path: Path,
    timings: dict[str, float],
    sanity: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the conversion metadata record saved next to the artifacts."""
    return {
        "args": {
            "seq_len": args.seq_len,
            "batch": args.batch,
            "model_dir": str(args.model_dir),
            "out_root": str(args.out_root),
            "attn": args.attn,
            "target": args.target,
            "precision": args.precision,
            "mask_fill_value": args.mask_fill_value,
        },
        "versions": build_versions_info(),
        "patches": {
            "rotate_half_static": True,
            "eager_attention_rank4": True,
            "mask_fill_value": args.mask_fill_value is not None,
        },
        "artifacts": {"mlpackage": str(mlpackage_path), "mlmodelc": str(mlmodelc_path)},
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
        "sanity": sanity,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the full conversion pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, 1 if the sanity check failed).
    """
    args = parse_args(argv)
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.batch <= 0:
        raise SystemExit("--batch must be a positive integer")
    stem = build_stem(args)
    out_dir = args.out_root / args.model_dir.resolve().name
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage_path = out_dir / f"{stem}.mlpackage"
    mlmodelc_path = out_dir / f"{stem}.mlmodelc"
    timings: dict[str, float] = {}
    started = time.perf_counter()

    print(f"[1/7] Loading model from {args.model_dir} (attn={args.attn}, fp32)")
    step = time.perf_counter()
    tokenizer = load_tokenizer(args.model_dir)
    model = load_reranker_torch_model(args.model_dir, attn=args.attn)
    model.config.return_dict = False  # trace requires tuple outputs
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    if head_dim % 2 != 0:
        raise SystemExit(f"odd RoPE head dim ({head_dim}) is incompatible with patch_rotate_half")
    patch_rotate_half()
    # Rank-5 attention intermediates make the ANE compiler reject the model
    # at B>1 (v0.3実装計画.md §4.11). The rewrite is bit-exact and is applied
    # at every batch size so the S×B sweep compares one graph shape instead
    # of mixing rank-5 (B=1) and rank-4 (B>1) variants.
    print("      applying rank-4 eager attention rewrite")
    patch_eager_attention_rank4()
    if args.mask_fill_value is not None:
        print(f"      applying experimental mask fill value {args.mask_fill_value}")
        # The mask helper lives on the backbone, not the classification model.
        patch_mask_fill_value(model.model, args.mask_fill_value)
    timings["load"] = time.perf_counter() - step

    print("[2/7] Wrapping to expose raw logits")
    wrapper = RerankerWrapper(model).eval()

    print(f"[3/7] Tracing with a fixed ({args.batch}, {args.seq_len}) example pair")
    step = time.perf_counter()
    # The single trace pair is replicated to B rows so the traced graph
    # already carries the target batch size.
    example = tokenize_pairs(tokenizer, [TRACE_EXAMPLE_PAIR] * args.batch, args.seq_len)
    traced = trace_model(wrapper, example)
    timings["trace"] = time.perf_counter() - step

    print(f"[4/7] Converting to mlprogram ({args.precision}, {args.target})")
    step = time.perf_counter()
    mlmodel = convert_model(
        traced, args.seq_len, args.precision, args.target, "logits", batch_size=args.batch
    )
    timings["convert"] = time.perf_counter() - step
    del traced, wrapper, model
    gc.collect()

    print(f"[5/7] Saving {mlpackage_path}")
    if mlpackage_path.exists():
        shutil.rmtree(mlpackage_path)
    mlmodel.save(str(mlpackage_path))
    del mlmodel
    gc.collect()

    print(f"[6/7] Compiling to {mlmodelc_path}")
    step = time.perf_counter()
    compile_model(mlpackage_path, mlmodelc_path)
    timings["compile"] = time.perf_counter() - step

    print("[7/7] Sanity check on CPU_AND_NE")
    step = time.perf_counter()
    sanity = run_sanity_check(mlmodelc_path, args.model_dir, args.seq_len, args.batch)
    timings["sanity"] = time.perf_counter() - step
    timings["total"] = time.perf_counter() - started

    metadata = build_metadata(args, mlpackage_path, mlmodelc_path, timings, sanity)
    metadata_path = out_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"      output key      : {sanity['output_key']}")
    print(f"      finite outputs  : {sanity['finite']}")
    print(f"      logits coreml   : {[round(v, 5) for v in sanity['coreml_logits']]}")
    print(f"      logits fp32     : {[round(v, 5) for v in sanity['fp32_logits']]}")
    print(f"      scores coreml   : {[round(v, 5) for v in sanity['coreml_scores']]}")
    print(f"      scores fp32     : {[round(v, 5) for v in sanity['fp32_scores']]}")
    print(
        f"      sigmoid max |Δ| : {sanity['sigmoid_max_abs_diff']:.5f} "
        f"(tolerance {sanity['sigmoid_tolerance']})"
    )
    print(
        f"      ordering ok     : coreml={sanity['ordering_ok_coreml']} "
        f"fp32={sanity['ordering_ok_fp32']}"
    )
    consistency = sanity["batch_consistency"]
    if consistency is not None:
        print(
            f"      batch rows |Δ|  : min {consistency['logit_abs_diff_min']:.7f} / "
            f"max {consistency['logit_abs_diff_max']:.7f} "
            f"(tolerance {consistency['logit_tolerance']}, passed={consistency['passed']})"
        )
    print(f"      timings (sec)   : {metadata['timings_sec']}")
    print(f"      metadata        : {metadata_path}")
    if not sanity["passed"]:
        print("SANITY CHECK FAILED (see §4.8 C2)")
        return 1
    print(f"SANITY CHECK PASSED: {mlmodelc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
