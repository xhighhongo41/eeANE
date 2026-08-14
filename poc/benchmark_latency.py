"""Measure Core ML inference latency and ANE op placement for ruri-v3-310m.

Pipeline (see 開発資料/v0.1実装計画.md §4.7):
compiled .mlmodelc -> CompiledMLModel (cold load + first predict timing)
-> warmup -> N timed predict() calls -> median/p90/mean/min + tokens/sec
-> optional MLComputePlan op-placement report -> optional sustained-load
mode for manual `powermetrics` observation.

Usage:
    uv run python poc/benchmark_latency.py --seq-len 512 --compute-units CPU_AND_NE
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
import transformers

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/benchmark_latency.py` to import the poc package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc.common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    load_corpus_paragraphs,
    load_tokenizer,
    tokenize_batch,
)

# Number of leading corpus paragraphs used as the round-robin input pool.
NUM_INPUT_TEXTS = 8

# Predict calls run (and discarded) before the timed warm loop begins.
NUM_WARMUP_PREDICTS = 5

# Reporting interval (seconds) for the --sustain progress log.
SUSTAIN_REPORT_INTERVAL_SEC = 5.0

# Maps the --compute-units CLI choice to the coremltools enum.
_COMPUTE_UNITS = {
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "ALL": ct.ComputeUnit.ALL,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Benchmark Core ML latency and ANE op placement for ruri-v3-310m."
    )
    parser.add_argument("--seq-len", type=int, required=True, help="Fixed sequence length S.")
    parser.add_argument(
        "--compute-units",
        choices=list(_COMPUTE_UNITS),
        default="CPU_AND_NE",
        help="Core ML compute unit selection.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local HF model directory."
    )
    parser.add_argument(
        "--mlmodelc",
        type=Path,
        default=None,
        help="Compiled model path. Defaults to the T5 naming convention "
        "(models/compiled/<model-dir-name>/s{S}_b1_eager_macos13.mlmodelc).",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of timed warm predict() calls.")
    parser.add_argument(
        "--sustain",
        type=int,
        default=0,
        help="If > 0, run a sustained predict() loop for this many seconds "
        "(for manual `powermetrics` observation) instead of the cold/warm benchmark.",
    )
    parser.add_argument(
        "--compute-plan",
        action="store_true",
        help="Report per-op compute device placement via MLComputePlan.",
    )
    return parser.parse_args(argv)


def default_mlmodelc_path(model_dir: Path, seq_len: int) -> Path:
    """Build the default compiled model path, following T5's naming scheme.

    Matches ``poc/convert_embedding.py``'s ``build_stem``/output directory
    for the default conversion settings (``attn=eager``, ``target=macos13``,
    ``precision=fp16``).

    Args:
        model_dir: Local HF model directory (only its resolved name is used).
        seq_len: Fixed sequence length S.

    Returns:
        Path to the expected ``.mlmodelc`` directory.
    """
    stem = f"s{seq_len}_b1_eager_macos13"
    return _REPO_ROOT / "models" / "compiled" / model_dir.resolve().name / f"{stem}.mlmodelc"


def prepare_inputs(model_dir: Path, seq_len: int) -> dict[str, np.ndarray]:
    """Tokenize the first ``NUM_INPUT_TEXTS`` corpus paragraphs ahead of time.

    Tokenization must happen outside the timed predict() loop so that
    latency measurements reflect Core ML inference only.

    Args:
        model_dir: Local HF model directory used to load the tokenizer.
        seq_len: Fixed sequence length S.

    Returns:
        Dict with ``input_ids``/``attention_mask``, each shape
        ``(NUM_INPUT_TEXTS, seq_len)`` int32.
    """
    tokenizer = load_tokenizer(model_dir)
    paragraphs = load_corpus_paragraphs()
    texts = paragraphs[:NUM_INPUT_TEXTS]
    if len(texts) < NUM_INPUT_TEXTS:
        raise SystemExit(f"corpus provided only {len(texts)} paragraphs, need {NUM_INPUT_TEXTS}")
    return tokenize_batch(tokenizer, texts, seq_len)


def _predict_input(model: Any, inputs: dict[str, np.ndarray], index: int) -> None:
    """Run a single predict() call on one round-robin input row.

    Args:
        model: Loaded ``CompiledMLModel``.
        inputs: Batch tensors from :func:`prepare_inputs`.
        index: Row index (already reduced modulo the pool size by the caller).
    """
    model.predict(
        {
            "input_ids": inputs["input_ids"][index : index + 1],
            "attention_mask": inputs["attention_mask"][index : index + 1],
        }
    )


def run_cold_and_warm_benchmark(
    model: Any, inputs: dict[str, np.ndarray], n: int
) -> dict[str, Any]:
    """Measure cold load/first-predict timing and warm predict() statistics.

    The model is assumed freshly constructed (its load time was already
    measured by the caller); this function performs the first predict()
    call plus the warmup and timed warm loop.

    Args:
        model: Freshly constructed ``CompiledMLModel``.
        inputs: Round-robin input pool from :func:`prepare_inputs`.
        n: Number of timed warm predict() calls.

    Returns:
        Dict with ``first_predict_sec`` and the warm statistics.
    """
    pool_size = inputs["input_ids"].shape[0]
    counter = 0

    # First predict() after construction: the "cold" inference cost.
    step = time.perf_counter()
    _predict_input(model, inputs, counter % pool_size)
    first_predict_sec = time.perf_counter() - step
    counter += 1

    # Warmup predicts (discarded), so the timed loop sees a settled model.
    for _ in range(NUM_WARMUP_PREDICTS):
        _predict_input(model, inputs, counter % pool_size)
        counter += 1

    # Timed warm loop.
    warm_times: list[float] = []
    for _ in range(n):
        step = time.perf_counter()
        _predict_input(model, inputs, counter % pool_size)
        warm_times.append(time.perf_counter() - step)
        counter += 1

    return {
        "first_predict_sec": first_predict_sec,
        "warm_times_sec": warm_times,
    }


def summarize_warm_times(warm_times: list[float], seq_len: int) -> dict[str, float]:
    """Compute median/p90/mean/min latency and tokens/sec from warm timings.

    Args:
        warm_times: Per-call predict() durations in seconds.
        seq_len: Fixed sequence length S, used for the tokens/sec estimate.

    Returns:
        Dict of summary statistics (seconds unless noted).
    """
    median_sec = statistics.median(warm_times)
    return {
        "n": len(warm_times),
        "median_sec": median_sec,
        "p90_sec": float(np.percentile(warm_times, 90)),
        "mean_sec": statistics.mean(warm_times),
        "min_sec": min(warm_times),
        "tokens_per_sec": seq_len / median_sec,
    }


def _collect_program_operations(block: Any) -> list[Any]:
    """Recursively collect all operations from an MLProgram block.

    Descends into nested blocks (e.g. control-flow ops) so no op is missed,
    even though ruri-v3-310m is not expected to contain any.

    Args:
        block: An ``MLModelStructureProgramBlock``.

    Returns:
        Flat list of ``MLModelStructureProgramOperation`` instances.
    """
    operations: list[Any] = []
    for op in block.operations:
        operations.append(op)
        for nested_block in op.blocks:
            operations.extend(_collect_program_operations(nested_block))
    return operations


def compute_plan_report(mlmodelc_path: Path, compute_units: str) -> dict[str, Any]:
    """Summarize per-op compute device placement via MLComputePlan.

    Uses the ``coremltools.models.compute_plan.MLComputePlan`` API (verified
    against the installed coremltools 9.0 source at
    ``coremltools/models/compute_plan.py``). If the API is unavailable or
    raises at runtime, the failure is reported as a skip rather than
    propagated, per §4.7.

    Args:
        mlmodelc_path: Compiled model directory.
        compute_units: One of the ``--compute-units`` CLI choices.

    Returns:
        Dict describing op counts per device, or ``{"skipped": True, ...}``.
    """
    try:
        from coremltools.models.compute_device import (
            MLCPUComputeDevice,
            MLGPUComputeDevice,
            MLNeuralEngineComputeDevice,
        )
        from coremltools.models.compute_plan import MLComputePlan

        plan = MLComputePlan.load_from_path(
            str(mlmodelc_path), compute_units=_COMPUTE_UNITS[compute_units]
        )
        program = plan.model_structure.program
        if program is None:
            raise RuntimeError("compiled model has no MLProgram structure")

        operations = _collect_program_operations(program.functions["main"].block)
        device_counts = {"NE": 0, "GPU": 0, "CPU": 0, "unspecified": 0}
        for op in operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                # Typically `const` ops: no dispatch decision is made for them.
                device_counts["unspecified"] += 1
                continue
            device = usage.preferred_compute_device
            if isinstance(device, MLNeuralEngineComputeDevice):
                device_counts["NE"] += 1
            elif isinstance(device, MLGPUComputeDevice):
                device_counts["GPU"] += 1
            elif isinstance(device, MLCPUComputeDevice):
                device_counts["CPU"] += 1
            else:
                device_counts["unspecified"] += 1

        dispatched = device_counts["NE"] + device_counts["GPU"] + device_counts["CPU"]
        ne_placement_pct = 100.0 * device_counts["NE"] / dispatched if dispatched > 0 else 0.0
        return {
            "skipped": False,
            "total_ops": len(operations),
            "device_counts": device_counts,
            "ne_placement_pct": ne_placement_pct,
        }
    except Exception as exc:  # noqa: BLE001 - any failure degrades to a recorded skip (§4.7)
        print(f"WARNING: MLComputePlan reporting skipped: {exc}")
        return {"skipped": True, "reason": str(exc)}


def run_sustain_loop(
    model: Any, inputs: dict[str, np.ndarray], sustain_seconds: int
) -> dict[str, Any]:
    """Run predict() continuously for manual `powermetrics` ANE observation.

    Args:
        model: Loaded ``CompiledMLModel``.
        inputs: Round-robin input pool from :func:`prepare_inputs`.
        sustain_seconds: Duration to keep looping, in seconds.

    Returns:
        Dict with the elapsed time and total predict() call count.
    """
    print("Run 'sudo powermetrics --samplers ane_power -i 1000' in another terminal now.")
    pool_size = inputs["input_ids"].shape[0]
    counter = 0
    next_report_at = SUSTAIN_REPORT_INTERVAL_SEC
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= sustain_seconds:
            break
        _predict_input(model, inputs, counter % pool_size)
        counter += 1
        elapsed = time.perf_counter() - start
        if elapsed >= next_report_at:
            print(f"  elapsed={elapsed:.1f}s loops={counter}")
            next_report_at += SUSTAIN_REPORT_INTERVAL_SEC
    total_elapsed = time.perf_counter() - start
    print(f"Sustain loop finished: elapsed={total_elapsed:.1f}s loops={counter}")
    return {
        "sustain_seconds": sustain_seconds,
        "elapsed_sec": total_elapsed,
        "total_predicts": counter,
    }


def build_environment_info(mlmodelc_path: Path, compute_units: str) -> dict[str, Any]:
    """Assemble environment/version metadata recorded alongside measurements."""
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver(),
        "compute_units": compute_units,
        "mlmodelc_path": str(mlmodelc_path),
    }


def result_path(seq_len: int, compute_units: str, sustain: bool) -> Path:
    """Build the results JSON output path per §4.7 naming convention."""
    results_dir = _REPO_ROOT / "poc" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = "latency_sustain" if sustain else "latency"
    return results_dir / f"{prefix}_s{seq_len}_{compute_units}.json"


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary of the benchmark result to stdout."""
    print(f"mode            : {result['mode']}")
    print(f"compute_units   : {result['environment']['compute_units']}")
    print(f"mlmodelc        : {result['environment']['mlmodelc_path']}")
    if result["mode"] == "sustain":
        sustain = result["sustain"]
        print(f"load_sec        : {result['load_sec']:.4f}")
        print(f"sustain_seconds : {sustain['sustain_seconds']}")
        print(f"total_predicts  : {sustain['total_predicts']}")
        return
    print(f"load_sec        : {result['cold']['load_sec']:.4f}")
    print(f"first_predict   : {result['cold']['first_predict_sec']:.4f}")
    print(f"cold_total_sec  : {result['cold']['cold_total_sec']:.4f}")
    warm = result["warm"]
    print(f"median_sec      : {warm['median_sec']:.4f}")
    print(f"p90_sec         : {warm['p90_sec']:.4f}")
    print(f"mean_sec        : {warm['mean_sec']:.4f}")
    print(f"min_sec         : {warm['min_sec']:.4f}")
    print(f"tokens_per_sec  : {warm['tokens_per_sec']:.2f}")
    plan = result["compute_plan"]
    if plan.get("skipped"):
        print(f"compute_plan    : skipped ({plan['reason']})")
    elif plan:
        print(
            f"compute_plan    : {plan['device_counts']} "
            f"(NE placement {plan['ne_placement_pct']:.1f}% of {plan['total_ops']} ops)"
        )
    else:
        print("compute_plan    : not requested (--compute-plan)")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark for a single (seq_len, compute_units) combination.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (always 0; this script is not a pass/fail gate).
    """
    args = parse_args(argv)
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    mlmodelc_path = args.mlmodelc or default_mlmodelc_path(args.model_dir, args.seq_len)
    if not mlmodelc_path.exists():
        raise SystemExit(f"compiled model not found: {mlmodelc_path}")

    print(f"Preparing {NUM_INPUT_TEXTS} tokenized inputs (S={args.seq_len})")
    inputs = prepare_inputs(args.model_dir, args.seq_len)

    print(f"Cold-loading {mlmodelc_path} (compute_units={args.compute_units})")
    step = time.perf_counter()
    # This is the process's only CompiledMLModel construction, per §4.7
    # ("one combination per process") so this timing reflects a true cold load.
    model = ct.models.CompiledMLModel(
        str(mlmodelc_path), compute_units=_COMPUTE_UNITS[args.compute_units]
    )
    load_sec = time.perf_counter() - step

    environment = build_environment_info(mlmodelc_path, args.compute_units)
    result: dict[str, Any] = {
        "seq_len": args.seq_len,
        "environment": environment,
    }

    if args.sustain > 0:
        if args.compute_plan:
            print("NOTE: --compute-plan is ignored in --sustain mode.")
        result["mode"] = "sustain"
        result["load_sec"] = load_sec
        result["sustain"] = run_sustain_loop(model, inputs, args.sustain)
    else:
        result["mode"] = "benchmark"
        print(
            f"Running cold first-predict + {NUM_WARMUP_PREDICTS} warmup + {args.n} timed predicts"
        )
        cold_warm = run_cold_and_warm_benchmark(model, inputs, args.n)
        warm = summarize_warm_times(cold_warm["warm_times_sec"], args.seq_len)
        result["cold"] = {
            "load_sec": load_sec,
            "first_predict_sec": cold_warm["first_predict_sec"],
            "cold_total_sec": load_sec + cold_warm["first_predict_sec"],
        }
        result["warm"] = warm
        result["warm"]["warm_times_sec"] = cold_warm["warm_times_sec"]
        if args.compute_plan:
            print("Computing MLComputePlan op placement report")
            result["compute_plan"] = compute_plan_report(mlmodelc_path, args.compute_units)
        else:
            result["compute_plan"] = {}

    out_path = result_path(args.seq_len, args.compute_units, sustain=args.sustain > 0)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print_summary(result)
    print(f"results         : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
