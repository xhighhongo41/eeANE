"""Convert the ruri-v3-310m embedding model to a Core ML program.

Pipeline (see 開発資料/v0.1実装計画.md §4.1-§4.5):
HF model -> EmbeddingWrapper (in-graph mean pooling) -> torch.jit.trace
-> ct.convert (mlprogram, fixed (1, S) int32 inputs) -> .mlpackage
-> `xcrun coremlcompiler` -> .mlmodelc -> sanity check on CPU_AND_NE.

Usage:
    uv run python poc/convert_embedding.py --seq-len 512
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/convert_embedding.py` to import the poc package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc.common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    cosine_rowwise,
    encode_pytorch,
    load_tokenizer,
    load_torch_model,
    mean_pool,
    tokenize_batch,
)
from poc.convert_common import (  # noqa: E402
    build_versions_info,
    compile_model,
    convert_model,
    patch_mask_fill_value,
    patch_rotate_half,
    resolve_output_key,
    trace_model,
)

# Short Japanese sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "これは変換用のサンプル文です。"

# Fixed sanity-check sentences (short / medium / long) exercising different
# amounts of padding under the same fixed sequence length.
SANITY_TEXTS: list[str] = [
    "検索クエリ: 日本の首都はどこですか。",
    "検索文書: 東京は日本の首都であり、政治と経済の中心地として発展してきた都市である。",
    "トピック: 機械学習モデルを専用のアクセラレータ上で動かすと、消費電力を抑えつつ"
    "高い推論スループットを得られる場合がある。",
]

# Minimum cosine similarity against the PyTorch FP32 baseline (§4.5 step 6).
SANITY_COSINE_THRESHOLD = 0.99


class EmbeddingWrapper(torch.nn.Module):
    """Wraps ModernBertModel and performs masked mean pooling in-graph.

    Output matches sentence-transformers (Transformer + mean Pooling,
    no normalization) for ruri-v3-310m.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the backbone model.

        Args:
            model: ModernBertModel loaded in eval/FP32 mode with
                ``config.return_dict = False``.
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute pooled sentence embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            Pooled embeddings, shape (B, hidden_size).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs[0]  # (B, S, H)
        # Pooling formula lives in poc.common so the baseline cannot drift.
        return mean_pool(hidden, attention_mask)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(description="Convert ruri-v3-310m to a Core ML program.")
    parser.add_argument("--seq-len", type=int, required=True, help="Fixed sequence length S.")
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local HF model directory."
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
    """Build the artifact base name, e.g. ``s512_b1_eager_macos13`` (§4.4)."""
    stem = f"s{args.seq_len}_b1_{args.attn}_{args.target}"
    if args.precision == "fp32":
        stem += "_fp32"
    if args.mask_fill_value is not None:
        # Non-default masking is an experiment; never overwrite the baseline artifact.
        stem += "_maskfill"
    return stem


def run_sanity_check(mlmodelc_path: Path, model_dir: Path, seq_len: int) -> dict[str, Any]:
    """Run the Core ML model on CPU_AND_NE and compare with the FP32 baseline.

    Args:
        mlmodelc_path: Compiled model directory.
        model_dir: Local HF model directory (for the baseline and tokenizer).
        seq_len: Fixed sequence length S.

    Returns:
        Dict with the output key, per-text cosine similarities, and the
        pass/fail flags for (a) finiteness and (b) the cosine threshold.
    """
    tokenizer = load_tokenizer(model_dir)
    batch = tokenize_batch(tokenizer, SANITY_TEXTS, seq_len)
    compiled = ct.models.CompiledMLModel(
        str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    output_key: str | None = None
    rows: list[np.ndarray] = []
    for i in range(len(SANITY_TEXTS)):
        prediction = compiled.predict(
            {
                "input_ids": batch["input_ids"][i : i + 1],
                "attention_mask": batch["attention_mask"][i : i + 1],
            }
        )
        output_key = output_key or resolve_output_key(prediction, "embedding")
        rows.append(np.asarray(prediction[output_key], dtype=np.float32).reshape(-1))
    coreml_emb = np.stack(rows)

    # FP32 reference: sdpa attention, pooling shared via poc.common.mean_pool.
    baseline_model = load_torch_model(model_dir, attn="sdpa")
    baseline_emb = encode_pytorch(baseline_model, tokenizer, SANITY_TEXTS, seq_len)
    del baseline_model
    gc.collect()

    cosines = cosine_rowwise(coreml_emb, baseline_emb)
    finite = bool(np.isfinite(coreml_emb).all())
    return {
        "output_key": output_key,
        "compute_units": "CPU_AND_NE",
        "cosine_per_text": [float(c) for c in cosines],
        "cosine_min": float(cosines.min()),
        "cosine_mean": float(cosines.mean()),
        "cosine_threshold": SANITY_COSINE_THRESHOLD,
        "finite": finite,
        "passed": finite and bool(cosines.min() >= SANITY_COSINE_THRESHOLD),
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
    model = load_torch_model(args.model_dir, attn=args.attn)
    model.config.return_dict = False  # trace requires tuple outputs
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    if head_dim % 2 != 0:
        raise SystemExit(f"odd RoPE head dim ({head_dim}) is incompatible with patch_rotate_half")
    patch_rotate_half()
    if args.mask_fill_value is not None:
        print(f"      applying experimental mask fill value {args.mask_fill_value}")
        patch_mask_fill_value(model, args.mask_fill_value)
    timings["load"] = time.perf_counter() - step

    print("[2/7] Wrapping with in-graph mean pooling")
    wrapper = EmbeddingWrapper(model).eval()

    print(f"[3/7] Tracing with a fixed (1, {args.seq_len}) example input")
    step = time.perf_counter()
    example = tokenize_batch(tokenizer, [TRACE_EXAMPLE_TEXT], args.seq_len)
    traced = trace_model(wrapper, example)
    timings["trace"] = time.perf_counter() - step

    print(f"[4/7] Converting to mlprogram ({args.precision}, {args.target})")
    step = time.perf_counter()
    mlmodel = convert_model(traced, args.seq_len, args.precision, args.target, "embedding")
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
    sanity = run_sanity_check(mlmodelc_path, args.model_dir, args.seq_len)
    timings["sanity"] = time.perf_counter() - step
    timings["total"] = time.perf_counter() - started

    metadata = build_metadata(args, mlpackage_path, mlmodelc_path, timings, sanity)
    metadata_path = out_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"      output key      : {sanity['output_key']}")
    print(f"      finite outputs  : {sanity['finite']}")
    print(f"      cosine vs FP32  : {[round(c, 5) for c in sanity['cosine_per_text']]}")
    print(f"      cosine min/mean : {sanity['cosine_min']:.5f} / {sanity['cosine_mean']:.5f}")
    print(f"      timings (sec)   : {metadata['timings_sec']}")
    print(f"      metadata        : {metadata_path}")
    if not sanity["passed"]:
        print("SANITY CHECK FAILED (see §4.8 C2)")
        return 1
    print(f"SANITY CHECK PASSED: {mlmodelc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
