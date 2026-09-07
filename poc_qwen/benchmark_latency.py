"""Measure Core ML inference latency and Neural Engine op placement.

This script does not convert or verify a model; it only loads an already
compiled ``.mlmodelc`` artifact and measures how fast it runs and where its
operations land. Three independent measurements are available:

* Cold load and warm inference timing: how long it takes to construct a
  ``CompiledMLModel`` and run the first ``predict()`` call, followed by
  median/p90/mean/min latency over many warm calls.
* Compute unit placement (``--compute-plan``): what fraction of the
  program's operations the coremltools compute plan API assigns to the
  Neural Engine, versus the GPU or CPU, using the ``MLComputePlan`` API.
* A sustained inference loop (``--sustain``): keeps calling ``predict()``
  for a fixed duration so a human can observe Neural Engine power draw
  with an external measurement tool in another terminal.

Usage:
    uv run python poc_qwen/benchmark_latency.py --mlmodelc models/foo.mlmodelc --seq-len 512
    uv run python -m poc_qwen.benchmark_latency --mlmodelc models/foo.mlmodelc --seq-len 512 \\
        --batch 4 --compute-plan
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc_qwen/benchmark_latency.py` to import the poc_qwen package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc_qwen import common  # noqa: E402

# Maps the --compute-units CLI choice to the coremltools enum. Both
# single-accelerator options matter for interpreting a result: "cpu_only"
# separates the accelerator's contribution from the graph itself, and
# "cpu_and_gpu" gives the comparison against the accelerator a serving
# stack would otherwise use on this hardware.
_COMPUTE_UNITS: dict[str, ct.ComputeUnit] = {
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "all": ct.ComputeUnit.ALL,
}

# How many of the most common non-Neural-Engine op types are reported by
# compute_plan_report; which ops keep falling off the Neural Engine is the
# central question this script exists to answer.
TOP_NON_NE_OP_TYPES = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Benchmark Core ML latency and Neural Engine op placement for a "
        "compiled .mlmodelc artifact."
    )
    parser.add_argument(
        "--model-id",
        default=common.DEFAULT_MODEL_ID,
        help="HuggingFace model id or local directory used only to tokenize the "
        "benchmark input texts; it is not loaded as a PyTorch model.",
    )
    parser.add_argument(
        "--mlmodelc",
        type=Path,
        required=True,
        help="Path to the compiled Core ML model (.mlmodelc) to benchmark.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Fixed sequence length S the compiled model was converted for.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Fixed batch size B the compiled model was converted for.",
    )
    parser.add_argument(
        "--compute-units",
        choices=list(_COMPUTE_UNITS),
        default="cpu_and_ne",
        help="Core ML compute unit selection passed to CompiledMLModel.",
    )
    parser.add_argument(
        "--n", type=int, default=50, help="Number of timed warm predict() calls to measure."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of predict() calls to run and discard before timing starts.",
    )
    parser.add_argument(
        "--compute-plan",
        action="store_true",
        help="Also report per-operation compute unit placement (Neural Engine, GPU, CPU) "
        "using the coremltools MLComputePlan API.",
    )
    parser.add_argument(
        "--sustain",
        type=float,
        default=0.0,
        help="If > 0, run predict() in a loop for this many seconds instead of the "
        "cold/warm timing measurement. This mode exists so a human can observe Neural "
        "Engine power draw with an external measurement tool in a separate terminal "
        "while inference runs continuously.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to "
        "poc_qwen/results/latency_s{seq_len}_b{batch}_{compute_units}.json.",
    )
    return parser.parse_args(argv)


def default_output_path(seq_len: int, batch: int, compute_units: str) -> Path:
    """Build the default results JSON path from the benchmarked configuration.

    Args:
        seq_len: Fixed sequence length S.
        batch: Fixed batch size B.
        compute_units: One of the ``--compute-units`` CLI choices.

    Returns:
        Path under ``poc_qwen/results/`` encoding the configuration in its
        filename, so sweeps over different configurations never collide.
    """
    return _REPO_ROOT / "poc_qwen" / "results" / f"latency_s{seq_len}_b{batch}_{compute_units}.json"


def build_input_pool(tokenizer: Any, seq_len: int, batch: int) -> dict[str, np.ndarray]:
    """Tokenize a round-robin pool of query-prompted input batches.

    Feeding the exact same array to every predict() call risks measuring an
    internal cache instead of genuine inference cost, so the pool cycles
    through several different texts instead. The base texts are every
    sentence in :func:`common.sanity_text_sets` (9 sentences across the
    project's en/ja/zh fixtures); they are repeated cyclically until the
    pool holds a whole number of ``batch``-sized rows, so plain slicing
    always yields a full batch, and (when ``batch`` does not evenly divide
    the number of base texts) so multiple distinct batches exist to rotate
    through.

    Args:
        tokenizer: Tokenizer from :func:`common.load_tokenizer`.
        seq_len: Fixed sequence length used for tokenization.
        batch: Fixed batch size; the returned pool's row count is always a
            positive multiple of this value.

    Returns:
        Dict with ``input_ids``/``attention_mask``, each of shape
        ``(num_rows, seq_len)`` int32, where ``num_rows`` is a multiple of
        ``batch``.
    """
    base_texts = [text for _, texts in common.sanity_text_sets() for text in texts]
    total_rows = max(len(base_texts), batch)
    remainder = total_rows % batch
    if remainder:
        total_rows += batch - remainder
    cycled_texts = [base_texts[i % len(base_texts)] for i in range(total_rows)]
    return common.tokenize_batch(tokenizer, cycled_texts, seq_len, prompt=common.QUERY_PROMPT)


def _predict_batch(model: Any, pool: dict[str, np.ndarray], batch_index: int, batch: int) -> None:
    """Run a single predict() call on one batch-sized slice of the input pool.

    Args:
        model: Loaded ``CompiledMLModel``.
        pool: Input pool from :func:`build_input_pool`.
        batch_index: Index of the batch within the pool (already reduced
            modulo the number of batches by the caller).
        batch: Batch size B; selects the ``[batch_index*B : (batch_index+1)*B]``
            row slice from ``pool``.
    """
    start = batch_index * batch
    end = start + batch
    model.predict(
        {
            "input_ids": pool["input_ids"][start:end],
            "attention_mask": pool["attention_mask"][start:end],
        }
    )


def measure_cold_load(
    mlmodelc_path: Path, compute_unit: ct.ComputeUnit, pool: dict[str, np.ndarray], batch: int
) -> tuple[Any, dict[str, float]]:
    """Time construction of a CompiledMLModel and its first predict() call.

    This must be the process's only ``CompiledMLModel`` construction for
    this ``.mlmodelc``, so the timing reflects a genuinely cold load rather
    than a framework-level cache from an earlier construction.

    Args:
        mlmodelc_path: Compiled model directory to load.
        compute_unit: Compute unit enum member to load the model with.
        pool: Input pool from :func:`build_input_pool`, used for the first
            predict() call.
        batch: Fixed batch size B.

    Returns:
        Tuple of the constructed model and a dict with ``construct_sec``,
        ``first_predict_sec``, and their sum ``total_sec``.
    """
    step = time.perf_counter()
    model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units=compute_unit)
    construct_sec = time.perf_counter() - step

    step = time.perf_counter()
    _predict_batch(model, pool, 0, batch)
    first_predict_sec = time.perf_counter() - step

    return model, {
        "construct_sec": construct_sec,
        "first_predict_sec": first_predict_sec,
        "total_sec": construct_sec + first_predict_sec,
    }


def run_warm_benchmark(
    model: Any, pool: dict[str, np.ndarray], batch: int, n: int, warmup: int
) -> list[float]:
    """Run warmup predict() calls, then time ``n`` further predict() calls.

    Args:
        model: Loaded ``CompiledMLModel`` (its cold load was already timed
            by the caller).
        pool: Input pool from :func:`build_input_pool`.
        batch: Fixed batch size B.
        n: Number of timed predict() calls.
        warmup: Number of predict() calls to run and discard first.

    Returns:
        Per-call predict() durations in seconds, one per timed call.
    """
    num_batches = pool["input_ids"].shape[0] // batch
    counter = 0
    for _ in range(warmup):
        _predict_batch(model, pool, counter % num_batches, batch)
        counter += 1

    warm_times: list[float] = []
    for _ in range(n):
        step = time.perf_counter()
        _predict_batch(model, pool, counter % num_batches, batch)
        warm_times.append(time.perf_counter() - step)
        counter += 1
    return warm_times


def summarize_warm_times(warm_times: list[float], seq_len: int, batch: int) -> dict[str, float]:
    """Compute median/p90/mean/min latency and tokens/sec from warm timings.

    Args:
        warm_times: Per-call predict() durations in seconds.
        seq_len: Fixed sequence length S, used for the tokens/sec estimate.
        batch: Fixed batch size B; each predict() call processes
            ``seq_len * batch`` padded tokens, so ``tokens_per_sec`` reports
            padded (not effective) throughput.

    Returns:
        Dict of summary statistics (seconds unless noted).
    """
    warm_array = np.asarray(warm_times, dtype=np.float64)
    median_sec = float(np.percentile(warm_array, 50))
    return {
        "n": len(warm_times),
        "median_sec": median_sec,
        "p90_sec": float(np.percentile(warm_array, 90)),
        "mean_sec": float(np.mean(warm_array)),
        "min_sec": float(np.min(warm_array)),
        "tokens_per_sec": seq_len * batch / median_sec,
    }


def _collect_program_operations(block: Any) -> list[Any]:
    """Recursively collect all operations from an MLProgram block.

    Descends into nested blocks (control-flow ops carry their own nested
    blocks) so no operation is missed.

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


def compute_plan_report(mlmodelc_path: Path, compute_unit: ct.ComputeUnit) -> dict[str, Any]:
    """Summarize per-operation compute unit placement via MLComputePlan.

    Uses ``coremltools.models.compute_plan.MLComputePlan`` together with
    ``coremltools.models.compute_device``. Any failure (missing API,
    unsupported model type, or any other exception) degrades to a recorded
    "unavailable" status instead of propagating, since a placement report
    failing should not stop the rest of the benchmark.

    Args:
        mlmodelc_path: Compiled model directory.
        compute_unit: Compute unit enum member the plan is evaluated for.

    Returns:
        On success, a dict with ``status: "ok"``, ``total_ops``,
        ``device_counts`` (operation count per device, including
        ``"unspecified"`` for operations with no dispatch decision, such as
        constants), ``ne_placement_pct`` (Neural Engine operations as a
        percentage of operations that *were* dispatched to some device),
        and ``top_non_ne_op_types`` (the most common op types among
        operations dispatched to the GPU or CPU, i.e. away from the Neural
        Engine, sorted by count, longest list :data:`TOP_NON_NE_OP_TYPES`).
        On failure, a dict with ``status: "unavailable"`` and ``reason``.
    """
    try:
        from coremltools.models.compute_device import (
            MLCPUComputeDevice,
            MLGPUComputeDevice,
            MLNeuralEngineComputeDevice,
        )
        from coremltools.models.compute_plan import MLComputePlan

        plan = MLComputePlan.load_from_path(str(mlmodelc_path), compute_units=compute_unit)
        program = plan.model_structure.program
        if program is None:
            raise RuntimeError("compiled model has no MLProgram structure")

        operations = _collect_program_operations(program.functions["main"].block)
        device_counts = {"neural_engine": 0, "gpu": 0, "cpu": 0, "unspecified": 0}
        non_ne_op_types: Counter[str] = Counter()
        for op in operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                # Typically `const` ops: no dispatch decision is made for them,
                # so they are excluded from the ne_placement_pct denominator.
                device_counts["unspecified"] += 1
                continue
            device = usage.preferred_compute_device
            if isinstance(device, MLNeuralEngineComputeDevice):
                device_counts["neural_engine"] += 1
            elif isinstance(device, MLGPUComputeDevice):
                device_counts["gpu"] += 1
                non_ne_op_types[op.operator_name] += 1
            elif isinstance(device, MLCPUComputeDevice):
                device_counts["cpu"] += 1
                non_ne_op_types[op.operator_name] += 1
            else:
                device_counts["unspecified"] += 1

        dispatched = device_counts["neural_engine"] + device_counts["gpu"] + device_counts["cpu"]
        ne_placement_pct = (
            100.0 * device_counts["neural_engine"] / dispatched if dispatched > 0 else 0.0
        )
        top_non_ne_op_types = [
            {"op_type": op_type, "count": count}
            for op_type, count in non_ne_op_types.most_common(TOP_NON_NE_OP_TYPES)
        ]
        return {
            "status": "ok",
            "total_ops": len(operations),
            "device_counts": device_counts,
            "ne_placement_pct": ne_placement_pct,
            "top_non_ne_op_types": top_non_ne_op_types,
        }
    except Exception as exc:
        # Any failure here (API unavailable, unsupported model type, ...)
        # degrades to a recorded status so the rest of the benchmark still runs.
        return {"status": "unavailable", "reason": str(exc)}


def run_sustain_loop(
    model: Any, pool: dict[str, np.ndarray], batch: int, sustain_seconds: float
) -> dict[str, float]:
    """Run predict() continuously so a human can observe power draw externally.

    Args:
        model: Loaded ``CompiledMLModel``.
        pool: Input pool from :func:`build_input_pool`.
        batch: Fixed batch size B.
        sustain_seconds: Duration to keep looping, in seconds.

    Returns:
        Dict with the requested and actual elapsed duration, the number of
        predict() calls made, and their mean latency.
    """
    num_batches = pool["input_ids"].shape[0] // batch
    counter = 0
    start = time.perf_counter()
    while time.perf_counter() - start < sustain_seconds:
        _predict_batch(model, pool, counter % num_batches, batch)
        counter += 1
    elapsed_sec = time.perf_counter() - start
    return {
        "requested_sustain_sec": sustain_seconds,
        "elapsed_sec": elapsed_sec,
        "iterations": counter,
        "mean_latency_sec": elapsed_sec / counter if counter > 0 else 0.0,
    }


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary of the benchmark result to stdout."""
    args = result["args"]
    print()
    print(f"mlmodelc        : {args['mlmodelc']}")
    print(f"compute_units   : {args['compute_units']}")
    print(f"seq_len / batch : {args['seq_len']} / {args['batch']}")
    print(f"mode            : {result['mode']}")

    cold = result["cold"]
    print(f"construct_sec   : {cold['construct_sec']:.4f}")
    print(f"first_predict   : {cold['first_predict_sec']:.4f}")
    print(f"cold_total_sec  : {cold['total_sec']:.4f}")

    if result["mode"] == "sustain":
        sustain = result["sustain"]
        print(
            f"sustain_sec     : {sustain['elapsed_sec']:.2f} "
            f"(requested {sustain['requested_sustain_sec']:.2f})"
        )
        print(f"iterations      : {sustain['iterations']}")
        print(f"mean_latency    : {sustain['mean_latency_sec']:.4f}")
    else:
        warm = result["warm"]
        print(f"median_sec      : {warm['median_sec']:.4f}")
        print(f"p90_sec         : {warm['p90_sec']:.4f}")
        print(f"mean_sec        : {warm['mean_sec']:.4f}")
        print(f"min_sec         : {warm['min_sec']:.4f}")
        print(f"tokens_per_sec  : {warm['tokens_per_sec']:.2f}")

    plan = result["compute_plan"]
    status = plan.get("status")
    if status == "ok":
        print(
            f"ne_placement_pct: {plan['ne_placement_pct']:.1f}% "
            f"of {plan['total_ops']} ops (device_counts={plan['device_counts']})"
        )
        if plan["ne_placement_pct"] < common.NE_PLACEMENT_WARN_THRESHOLD:
            print(
                f"WARNING: Neural Engine placement {plan['ne_placement_pct']:.1f}% is below "
                f"the {common.NE_PLACEMENT_WARN_THRESHOLD:.1f}% threshold."
            )
        if plan["top_non_ne_op_types"]:
            print("Top op types away from the Neural Engine (GPU or CPU):")
            for entry in plan["top_non_ne_op_types"]:
                print(f"  {entry['op_type']:<20s} {entry['count']}")
    elif status == "unavailable":
        print(f"compute_plan    : unavailable ({plan['reason']})")
    else:
        print("compute_plan    : not requested (pass --compute-plan)")


def main(argv: list[str] | None = None) -> int:
    """Run the latency and Neural Engine placement benchmark.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (always 0; low Neural Engine placement is a
        measurement result to report, not a failure to signal).
    """
    args = parse_args(argv)
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.batch <= 0:
        raise SystemExit("--batch must be a positive integer")
    if args.sustain < 0:
        raise SystemExit("--sustain must not be negative")
    if not args.mlmodelc.exists():
        raise SystemExit(f"compiled model not found: {args.mlmodelc}")

    compute_unit = _COMPUTE_UNITS[args.compute_units]
    output_path = args.output or default_output_path(args.seq_len, args.batch, args.compute_units)

    print(
        f"Tokenizing input pool (model_id={args.model_id}, seq_len={args.seq_len}, "
        f"batch={args.batch})"
    )
    model_dir = common.resolve_model_dir(args.model_id)
    tokenizer = common.load_tokenizer(model_dir)
    pool = build_input_pool(tokenizer, args.seq_len, args.batch)

    result: dict[str, Any] = {
        "args": {
            "model_id": args.model_id,
            "mlmodelc": str(args.mlmodelc),
            "seq_len": args.seq_len,
            "batch": args.batch,
            "compute_units": args.compute_units,
            "n": args.n,
            "warmup": args.warmup,
            "compute_plan": args.compute_plan,
            "sustain": args.sustain,
            "output": str(output_path),
        }
    }

    print(f"Cold-loading {args.mlmodelc} (compute_units={args.compute_units})")
    model, cold = measure_cold_load(args.mlmodelc, compute_unit, pool, args.batch)
    result["cold"] = cold

    if args.compute_plan:
        print("Computing MLComputePlan operation placement report")
        result["compute_plan"] = compute_plan_report(args.mlmodelc, compute_unit)
    else:
        result["compute_plan"] = {"status": "not_requested"}

    if args.sustain > 0:
        result["mode"] = "sustain"
        print(f"Running sustained inference for {args.sustain:.1f}s")
        result["sustain"] = run_sustain_loop(model, pool, args.batch, args.sustain)
    else:
        result["mode"] = "benchmark"
        print(f"Running {args.warmup} warmup + {args.n} timed predict() calls")
        warm_times = run_warm_benchmark(model, pool, args.batch, args.n, args.warmup)
        result["warm"] = summarize_warm_times(warm_times, args.seq_len, args.batch)

    print_summary(result)
    common.write_result_json(output_path, result)
    print(f"\nresults written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
