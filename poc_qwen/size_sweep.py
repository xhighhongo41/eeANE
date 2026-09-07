"""Sweep synthetic Qwen3-shaped configs to find the Neural Engine size boundary.

The real Qwen3 embedding checkpoints large enough to answer "how big can this
still go" are impractical to obtain and convert on the machine this project
was developed on: the 4B and 8B checkpoints alone are several gigabytes of
weights to download, on top of the memory the conversion itself needs. This
script sidesteps the download and the memory cost of loading real weights by
building a ``Qwen3Model`` directly from a ``Qwen3Config`` -- so the weights
are randomly initialized rather than downloaded -- and pushing that model
through the exact same trace/convert/compile pipeline the real embedding
model goes through in :mod:`poc_qwen.convert_embedding`.

Methodology and its limits (read before trusting a result from this script):

* Where a Core ML operation gets dispatched (Neural Engine, GPU, or CPU) and
  how long it takes to run depend on the *shape* of the compute graph --
  operator types, tensor shapes, and the numeric-range assumptions the fp16
  conversion patches in :mod:`poc_qwen.patches` are built around -- not on
  the numeric *values* stored in the weights. A randomly initialized model
  therefore produces the same graph, the same op placement, and the same
  latency a real checkpoint of the same shape would, without needing the
  real weights at all.
* What this method cannot measure is embedding *quality*: there is no
  meaningful reference to compare a random-weight model's output against, so
  this script never runs the cosine-similarity sanity check that
  :mod:`poc_qwen.convert_embedding` does. It only checks that a single
  ``predict()`` call returns finite numbers, which catches outright fp16
  numeric blow-ups (for example an overflow newly triggered by a wider
  hidden size) without claiming anything about accuracy.
* ``vocab_size`` is held fixed at the real model's value across every config
  in this sweep (see :data:`VOCAB_SIZE`) because the embedding table's size
  can itself affect conversion and placement, and the goal here is to
  isolate the effect of the *transformer backbone* growing, not the
  vocabulary.

Each config is run in its own subprocess (this module re-invokes itself in
``--worker`` mode via ``sys.executable``): a large config can exhaust memory
partway through conversion, and even a config that converts cleanly could
otherwise leave Core ML's compiler cache or process memory in a different
state than a fresh process would have. One config failing -- including one
whose worker subprocess is killed outright -- does not stop the sweep.

Usage:
    uv run python poc_qwen/size_sweep.py
    uv run python poc_qwen/size_sweep.py --configs a_0.6b,b_1.1b --n 10
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc_qwen/size_sweep.py` to import the poc_qwen package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc_qwen import common  # noqa: E402

# Random seed for both `torch.manual_seed` (weight initialization) and the
# NumPy generator that builds the dummy inputs, so a sweep run is
# reproducible even though the weights themselves carry no meaning.
RANDOM_SEED = 0

# Shape parameters shared by every config in the sweep, fixed at the real
# Qwen3-Embedding-0.6B/4B values. Only hidden_size, num_attention_heads,
# intermediate_size and num_hidden_layers vary between configs; see
# :data:`CONFIGS`.
HEAD_DIM = 128
NUM_KEY_VALUE_HEADS = 8
VOCAB_SIZE = 151669
MAX_POSITION_EMBEDDINGS = 32768
ROPE_THETA = 1_000_000.0
RMS_NORM_EPS = 1e-6
TIE_WORD_EMBEDDINGS = True

# Conversion precision and deployment target used for every config; the
# sweep is only about model size, so these are not exposed as CLI options.
PRECISION = "fp16"
TARGET = "macos13"

# Fraction of the sequence length treated as left padding in the dummy
# input, matching the left-padded convention the "last-left" pooling this
# sweep uses requires (see poc_qwen.convert_embedding.LastTokenEmbeddingWrapper).
_LEFT_PAD_FRACTION = 0.25

# Number of distinct random batches cycled through during the warm latency
# measurement, mirroring the intent of
# poc_qwen.benchmark_latency.build_input_pool (never feed predict() the
# exact same array on every call) without needing a tokenizer.
_POOL_CYCLES = 4

# Warm-up predict() calls discarded before the warm latency measurement.
_WARMUP_CALLS = 5

# Trailing characters of a crashed worker's stdout/stderr kept in the sweep
# result, so a result file records enough to diagnose a crash without
# growing unbounded for a worker that printed a lot before dying.
_TAIL_CHARS = 4000

DEFAULT_WORK_DIR = _REPO_ROOT / "models" / "compiled" / "qwen3-size-sweep"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "poc_qwen" / "results" / "size_sweep.json"


@dataclass(frozen=True)
class SizeConfig:
    """The backbone shape parameters that vary across the sweep."""

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_hidden_layers: int


# Configs to sweep, in ascending size order. "a_0.6b" is the control: its
# shape is identical to the real Qwen3-Embedding-0.6B, whose measured
# results are hardcoded in REFERENCE_0_6B below for comparison. "e_4b" is
# identical to the real Qwen3-Embedding-4B shape. The keys are display
# labels only; the actual parameter count is measured, not assumed.
CONFIGS: dict[str, SizeConfig] = {
    "a_0.6b": SizeConfig(
        hidden_size=1024, num_attention_heads=16, intermediate_size=3072, num_hidden_layers=28
    ),
    # The three configurations below bracket the point where placement was
    # first observed to collapse: 0.6b lands almost entirely on the Neural
    # Engine while 1.1b lands entirely on the CPU, so these fill that gap in
    # roughly even steps of compiled weight size.
    #
    # Grouped-query attention requires the query head count to be an exact
    # multiple of the key/value head count, so the head count cannot be
    # varied freely to hit an intermediate size. These configurations keep
    # it at the same value the smallest configuration uses and widen only
    # the hidden and feed-forward dimensions, which is enough to sweep the
    # size range and leaves the attention structure untouched between them.
    "a2_0.7b": SizeConfig(
        hidden_size=1152, num_attention_heads=16, intermediate_size=3456, num_hidden_layers=28
    ),
    "a3_0.8b": SizeConfig(
        hidden_size=1280, num_attention_heads=16, intermediate_size=3840, num_hidden_layers=28
    ),
    "a4_1.0b": SizeConfig(
        hidden_size=1408, num_attention_heads=16, intermediate_size=4224, num_hidden_layers=28
    ),
    # These two straddle a compiled weight size of 2 GiB. Placement was
    # found to hold at 1.78 GB and to be lost entirely at 2.36 GB, and a
    # two-gigabyte limit is a plausible shape for such a cliff, so these
    # sit just below and just above it to test that reading.
    "a5_1.0b": SizeConfig(
        hidden_size=1472, num_attention_heads=16, intermediate_size=4416, num_hidden_layers=28
    ),
    "a6_1.1b": SizeConfig(
        hidden_size=1536, num_attention_heads=16, intermediate_size=4608, num_hidden_layers=28
    ),
    "b_1.1b": SizeConfig(
        hidden_size=1536, num_attention_heads=24, intermediate_size=4608, num_hidden_layers=28
    ),
    "c_1.7b": SizeConfig(
        hidden_size=2048, num_attention_heads=32, intermediate_size=6144, num_hidden_layers=28
    ),
    "d_2.1b": SizeConfig(
        hidden_size=2048, num_attention_heads=32, intermediate_size=6144, num_hidden_layers=36
    ),
    "e_4b": SizeConfig(
        hidden_size=2560, num_attention_heads=32, intermediate_size=9728, num_hidden_layers=36
    ),
}

# Measured on the real Qwen3-Embedding-0.6B model (seq_len=128, batch=1,
# fp16, macOS13 target): the control config "a_0.6b" shares its exact
# shape, so its measurement here is meant to be compared against these
# numbers -- if they diverge substantially, something about the synthetic
# pipeline no longer matches how the real model was measured, and the rest
# of the sweep's results should not be trusted either.
REFERENCE_0_6B: dict[str, float] = {
    "param_count": 595_776_512,
    "ne_placement_pct": 99.7,
    "warm_median_ms": 20.3,
    "trace_sec": 2.3,
    "convert_sec": 32.6,
    "compile_sec": 0.5,
    "artifact_size_gb": 1.1,
}


class _StageError(RuntimeError):
    """An error raised while running one named pipeline stage.

    Wrapping every stage's exceptions this way lets the worker report which
    of the seven named stages (build/trace/convert/compile/load/predict/
    benchmark) failed, instead of a bare traceback the caller would have to
    parse.
    """

    def __init__(self, stage: str, original: BaseException) -> None:
        """Record the failing stage and the original exception.

        Args:
            stage: Name of the pipeline stage that failed.
            original: The exception raised inside that stage.
        """
        super().__init__(f"[{stage}] {original}")
        self.stage = stage
        self.original = original


@contextmanager
def _stage_guard(stage: str):
    """Re-raise any exception from a pipeline stage wrapped in :class:`_StageError`.

    Args:
        stage: Name of the pipeline stage this context covers.

    Yields:
        Nothing; the block runs under normal exception propagation, except
        that any :class:`Exception` is re-raised as a :class:`_StageError`
        carrying ``stage``.
    """
    try:
        yield
    except Exception as exc:
        raise _StageError(stage, exc) from exc


def _make_synthetic_batch(
    vocab_size: int, seq_len: int, batch_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Build one batch of random token ids with a left-padded mask.

    The values carry no language and are never compared against a
    reference (see the module docstring): random ids with a fixed,
    non-degenerate left-padding pattern are enough to exercise the same
    graph a real tokenized batch would, which is all this sweep needs.

    Args:
        vocab_size: Exclusive upper bound for the random token ids.
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B.
        rng: Random generator; the caller controls the seed for
            reproducibility.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(batch_size, seq_len)`` and dtype ``np.int32``.
    """
    input_ids = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
    pad_len = max(1, int(seq_len * _LEFT_PAD_FRACTION))
    attention_mask[:, :pad_len] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_dummy_pool(
    vocab_size: int, seq_len: int, batch_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Build a small pool of distinct random batches for the warm latency measurement.

    Mirrors the intent of ``poc_qwen.benchmark_latency.build_input_pool``
    (cycle through several distinct inputs so ``predict()`` cannot take a
    shortcut on a literally repeated array) without a tokenizer, since a
    synthetic model has no real vocabulary to tokenize text with.

    Args:
        vocab_size: Exclusive upper bound for the random token ids.
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B.
        rng: Random generator shared with the caller for reproducibility.

    Returns:
        Dict with ``input_ids``/``attention_mask``, each of shape
        ``(_POOL_CYCLES * batch_size, seq_len)``.
    """
    batches = [
        _make_synthetic_batch(vocab_size, seq_len, batch_size, rng) for _ in range(_POOL_CYCLES)
    ]
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in ("input_ids", "attention_mask")
    }


def _dir_size_bytes(path: Path) -> int:
    """Sum the size of every file under ``path``, recursively.

    A Core ML artifact (``.mlpackage`` or ``.mlmodelc``) is a directory of
    many files, so ``path.stat().st_size`` on the directory itself would
    not report its real size.

    Args:
        path: File or directory to measure.

    Returns:
        Total size in bytes; 0 if ``path`` does not exist.
    """
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _run_worker(config_key: str, args: argparse.Namespace) -> dict[str, Any]:
    """Build, convert, and benchmark one synthetic config end to end.

    Runs entirely in the current process; this is only ever called from
    ``--worker`` mode, which the parent sweep runs as a dedicated
    subprocess per config (see the module docstring for why). Every stage
    is wrapped in :func:`_stage_guard`, so an expected failure (a shape
    that does not trace, a conversion Core ML rejects, ...) comes back as
    a normal ``status: "failed"`` result rather than a traceback; a crash
    the process cannot catch at all (killed for memory) instead shows up
    to the parent as an abnormal subprocess exit.

    Args:
        config_key: Key into :data:`CONFIGS`.
        args: Parsed command line arguments (uses ``seq_len``, ``batch``,
            ``n`` and ``work_dir``).

    Returns:
        A JSON-serializable result dict; see the module's result schema
        as produced here (``status``, ``param_count``, ``convert_ok``,
        ``load_ok``, ``compute_plan``, ``warm``, timings, ...).
    """
    import coremltools as ct
    import torch
    from transformers import Qwen3Config, Qwen3Model

    from poc_qwen import benchmark_latency as bl
    from poc_qwen import convert, patches
    from poc_qwen.convert_embedding import OUTPUT_NAME, LastTokenEmbeddingWrapper

    config = CONFIGS[config_key]
    seq_len = args.seq_len
    batch_size = args.batch
    work_dir: Path = args.work_dir
    mlpackage_path = work_dir / f"{config_key}.mlpackage"
    mlmodelc_path = work_dir / f"{config_key}.mlmodelc"

    result: dict[str, Any] = {
        "status": "ok",
        "config_key": config_key,
        "config": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "head_dim": HEAD_DIM,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "vocab_size": VOCAB_SIZE,
        },
        "seq_len": seq_len,
        "batch": batch_size,
        "convert_ok": False,
        "load_ok": False,
    }
    timings: dict[str, float] = {}
    convert_peak_rss_bytes: int | None = None
    compiled_model = None
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    try:
        rng = np.random.default_rng(RANDOM_SEED)

        step = time.perf_counter()
        with _stage_guard("build"):
            torch.manual_seed(RANDOM_SEED)
            hf_config = Qwen3Config(
                vocab_size=VOCAB_SIZE,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=NUM_KEY_VALUE_HEADS,
                head_dim=HEAD_DIM,
                max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                rope_theta=ROPE_THETA,
                rms_norm_eps=RMS_NORM_EPS,
                tie_word_embeddings=TIE_WORD_EMBEDDINGS,
                # sdpa attention never reaches the patched repeat_kv, so
                # eager attention is mandatory for a convertible graph.
                attn_implementation="eager",
            )
            model = Qwen3Model(hf_config).eval()
            model.config.use_cache = False
            model.config.return_dict = False
            param_count = int(sum(p.numel() for p in model.parameters()))
            patches.apply_patches(model)
        result["param_count"] = param_count
        timings["build"] = time.perf_counter() - step

        step = time.perf_counter()
        with _stage_guard("trace"):
            wrapper = LastTokenEmbeddingWrapper(model).eval()
            example = _make_synthetic_batch(VOCAB_SIZE, seq_len, batch_size, rng)
            traced = convert.trace_model(wrapper, example)
        timings["trace"] = time.perf_counter() - step

        step = time.perf_counter()
        with _stage_guard("convert"):
            mlmodel = convert.convert_model(
                traced, seq_len, PRECISION, TARGET, OUTPUT_NAME, batch_size=batch_size
            )
            output_key = convert.resolve_output_key(mlmodel, OUTPUT_NAME)
            # The FP32 backbone and the traced graph are no longer needed
            # once the Core ML program exists; releasing them before
            # saving keeps peak memory from stacking both representations
            # on top of each other, which matters most for the largest
            # configs this sweep exists to probe.
            del traced, wrapper, model
            gc.collect()
            work_dir.mkdir(parents=True, exist_ok=True)
            if mlpackage_path.exists():
                shutil.rmtree(mlpackage_path)
            mlmodel.save(str(mlpackage_path))
            del mlmodel
            gc.collect()
        timings["convert"] = time.perf_counter() - step

        step = time.perf_counter()
        with _stage_guard("compile"):
            convert.compile_model(mlpackage_path, mlmodelc_path)
        timings["compile"] = time.perf_counter() - step
        result["convert_ok"] = True
        # ru_maxrss only ever grows within a process, so this is the peak
        # RSS reached at any point since the worker started; in practice
        # it is dominated by the convert stage, the memory-heaviest one.
        # On macOS (unlike Linux) ru_maxrss is reported in bytes.
        convert_peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["artifact_size_bytes"] = _dir_size_bytes(mlmodelc_path)

        step = time.perf_counter()
        with _stage_guard("load"):
            compiled_model = ct.models.CompiledMLModel(
                str(mlmodelc_path), compute_units=compute_unit
            )
        timings["load"] = time.perf_counter() - step
        result["load_ok"] = True

        step = time.perf_counter()
        with _stage_guard("predict"):
            probe = _make_synthetic_batch(VOCAB_SIZE, seq_len, batch_size, rng)
            prediction = compiled_model.predict(probe)
            values = np.asarray(prediction[output_key], dtype=np.float32)
            predict_finite = bool(np.isfinite(values).all())
        timings["predict"] = time.perf_counter() - step
        result["predict_finite"] = predict_finite

        step = time.perf_counter()
        with _stage_guard("benchmark"):
            compute_plan = bl.compute_plan_report(mlmodelc_path, compute_unit)
            pool = _build_dummy_pool(VOCAB_SIZE, seq_len, batch_size, rng)
            warm_times = bl.run_warm_benchmark(
                compiled_model, pool, batch_size, args.n, _WARMUP_CALLS
            )
            warm = bl.summarize_warm_times(warm_times, seq_len, batch_size)
        timings["benchmark"] = time.perf_counter() - step
        result["compute_plan"] = compute_plan
        result["warm"] = warm
    except _StageError as exc:
        result["status"] = "failed"
        result["stage"] = exc.stage
        result["error"] = str(exc.original)
    finally:
        result["timings_sec"] = {key: round(value, 3) for key, value in timings.items()}
        if convert_peak_rss_bytes is not None:
            result["convert_peak_rss_bytes"] = convert_peak_rss_bytes
        del compiled_model
        gc.collect()
        if not args.keep_artifacts:
            # Both artifacts are removed together once measurement has
            # finished (successfully or not): a synthetic model's weights
            # carry no reusable value, and the largest configs' artifacts
            # would otherwise consume several gigabytes of disk each.
            shutil.rmtree(mlpackage_path, ignore_errors=True)
            shutil.rmtree(mlmodelc_path, ignore_errors=True)

    return result


def _worker_main(args: argparse.Namespace) -> int:
    """Run exactly one config and write its result JSON.

    This is the entry point ``--worker`` mode dispatches to. A failure
    inside :func:`_run_worker` itself is already turned into a stage-
    tagged result; this wrapper only guards against something going wrong
    before any stage guard could run (for example an import error), so the
    parent still receives a JSON file describing what happened whenever
    the process is able to run Python code at all.

    Args:
        args: Parsed command line arguments; must have ``worker`` and
            ``worker_output`` set.

    Returns:
        Always 0: a failed config is a valid measurement, not a process
        failure.

    Raises:
        SystemExit: If ``--worker`` names an unknown config or
            ``--worker-output`` is missing.
    """
    if args.worker not in CONFIGS:
        raise SystemExit(
            f"unknown --worker config: {args.worker!r} (expected one of {list(CONFIGS)})"
        )
    if args.worker_output is None:
        raise SystemExit("--worker requires --worker-output")

    try:
        result = _run_worker(args.worker, args)
    except Exception as exc:
        # Something failed outside of any _stage_guard block (most likely
        # an import error before the pipeline itself even started); "build"
        # is the closest attributable stage for a best-effort record.
        result = {
            "status": "failed",
            "stage": "build",
            "error": f"unexpected worker failure: {exc}",
            "config_key": args.worker,
        }

    common.write_result_json(args.worker_output, result)
    return 0


def _tail(text: str) -> str:
    """Keep only the trailing :data:`_TAIL_CHARS` characters of process output."""
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


def _run_config_subprocess(config_key: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one config in a fresh ``--worker`` subprocess and return its result.

    Args:
        config_key: Key into :data:`CONFIGS`.
        args: Parsed sweep-mode (parent) arguments.

    Returns:
        The worker's JSON result on a clean exit, or a synthesized
        ``status: "failed"`` record with ``stage: "killed_or_crashed"``
        when the subprocess timed out, exited abnormally, or produced no
        usable result file -- most commonly because the OS killed it for
        memory, which leaves no Python-level exception for the worker
        itself to report.
    """
    worker_output = args.work_dir / f"{config_key}.worker_result.json"
    worker_output.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "poc_qwen.size_sweep",
        "--worker",
        config_key,
        "--worker-output",
        str(worker_output),
        "--seq-len",
        str(args.seq_len),
        "--batch",
        str(args.batch),
        "--n",
        str(args.n),
        "--work-dir",
        str(args.work_dir),
    ]
    if args.keep_artifacts:
        command.append("--keep-artifacts")

    try:
        completed = subprocess.run(
            command, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=args.timeout
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "config_key": config_key,
            "stage": "killed_or_crashed",
            "error": f"worker timed out after {args.timeout:.0f}s",
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
        }

    if worker_output.exists():
        try:
            return json.loads(worker_output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass  # Falls through to the crash report below.

    return {
        "status": "failed",
        "config_key": config_key,
        "stage": "killed_or_crashed",
        "error": (
            f"worker exited with code {completed.returncode} and produced no usable "
            f"result file ({worker_output})"
        ),
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _resolve_config_keys(raw: str | None) -> list[str]:
    """Parse ``--configs`` into an ordered, deduplicated list of config keys.

    Args:
        raw: Comma-separated config keys, or ``None`` for every config in
            :data:`CONFIGS`, in its declared order.

    Returns:
        Config keys in the order they should run.

    Raises:
        SystemExit: If any requested key is not in :data:`CONFIGS`.
    """
    if raw is None:
        return list(CONFIGS)
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    unknown = [key for key in keys if key not in CONFIGS]
    if unknown:
        raise SystemExit(
            f"unknown --configs entries: {unknown} (expected a subset of {list(CONFIGS)})"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _fmt(value: float | int | None, spec: str = ".2f") -> str:
    """Format a number with ``spec``, or ``"-"`` when it is ``None``."""
    return "-" if value is None else format(value, spec)


def _bytes_to_gb(num_bytes: float | int | None) -> float | None:
    """Convert a byte count to gibibytes, passing ``None`` through unchanged."""
    return None if num_bytes is None else num_bytes / (1024**3)


def _sec_to_ms(seconds: float | int | None) -> float | None:
    """Convert seconds to milliseconds, passing ``None`` through unchanged."""
    return None if seconds is None else seconds * 1000.0


def _ne_placement_pct(result: dict[str, Any]) -> float | None:
    """Extract the Neural Engine placement percentage from a worker result, if present."""
    plan = result.get("compute_plan") or {}
    return plan.get("ne_placement_pct") if plan.get("status") == "ok" else None


def _warm_median_sec(result: dict[str, Any]) -> float | None:
    """Extract the warm median predict() latency from a worker result, if present."""
    warm = result.get("warm") or {}
    return warm.get("median_sec")


def print_table(results: dict[str, dict[str, Any]], config_keys: list[str]) -> None:
    """Print the sweep results as a Markdown table.

    Args:
        results: Worker results keyed by config key.
        config_keys: Config keys to print, in the order they ran.
    """
    headers = [
        "config",
        "params",
        "hidden",
        "layers",
        "converted",
        "loaded",
        "NE %",
        "warm median (ms)",
        "convert peak mem (GB)",
        "artifact (GB)",
        "failed stage",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for key in config_keys:
        result = results.get(key, {})
        cfg = CONFIGS[key]
        row = [
            key,
            _fmt(result.get("param_count"), ","),
            str(cfg.hidden_size),
            str(cfg.num_hidden_layers),
            "yes" if result.get("convert_ok") else "no",
            "yes" if result.get("load_ok") else "no",
            _fmt(_ne_placement_pct(result), ".1f"),
            _fmt(_sec_to_ms(_warm_median_sec(result)), ".1f"),
            _fmt(_bytes_to_gb(result.get("convert_peak_rss_bytes")), ".2f"),
            _fmt(_bytes_to_gb(result.get("artifact_size_bytes")), ".2f"),
            result.get("stage", "-") if result.get("status") == "failed" else "-",
        ]
        print("| " + " | ".join(row) + " |")


def print_reference_comparison(results: dict[str, dict[str, Any]]) -> None:
    """Print the "a_0.6b" synthetic measurement next to the real 0.6B baseline.

    "a_0.6b" uses the exact shape of the real Qwen3-Embedding-0.6B model
    (see :data:`CONFIGS` and :data:`REFERENCE_0_6B`), so this comparison is
    the sweep's built-in sanity check: if the synthetic numbers do not
    roughly agree with the real measurement, something about the synthetic
    pipeline diverges from how the real model was measured, and the rest of
    the sweep's results should not be trusted either.

    Args:
        results: Worker results keyed by config key.
    """
    print()
    print("Control check: a_0.6b (synthetic) vs. the real Qwen3-Embedding-0.6B measurement")
    result = results.get("a_0.6b")
    if result is None:
        print("  a_0.6b was not part of this sweep; no control check available.")
        return

    timings = result.get("timings_sec") or {}
    rows = [
        (
            "param count",
            _fmt(result.get("param_count"), ","),
            format(REFERENCE_0_6B["param_count"], ","),
        ),
        (
            "NE placement %",
            _fmt(_ne_placement_pct(result), ".1f"),
            f"{REFERENCE_0_6B['ne_placement_pct']:.1f}",
        ),
        (
            "warm median (ms)",
            _fmt(_sec_to_ms(_warm_median_sec(result)), ".1f"),
            f"{REFERENCE_0_6B['warm_median_ms']:.1f}",
        ),
        ("trace (s)", _fmt(timings.get("trace"), ".1f"), f"{REFERENCE_0_6B['trace_sec']:.1f}"),
        (
            "convert (s)",
            _fmt(timings.get("convert"), ".1f"),
            f"{REFERENCE_0_6B['convert_sec']:.1f}",
        ),
        (
            "compile (s)",
            _fmt(timings.get("compile"), ".1f"),
            f"{REFERENCE_0_6B['compile_sec']:.1f}",
        ),
        (
            "artifact size (GB)",
            _fmt(_bytes_to_gb(result.get("artifact_size_bytes")), ".2f"),
            f"{REFERENCE_0_6B['artifact_size_gb']:.1f}",
        ),
    ]
    print(f"  {'metric':<20s} {'synthetic':>15s} {'real 0.6B':>15s}")
    for name, synthetic, real in rows:
        print(f"  {name:<20s} {synthetic:>15s} {real:>15s}")


def _print_progress_line(key: str, result: dict[str, Any], elapsed_sec: float) -> None:
    """Print a one-line progress summary for a config that just finished.

    Args:
        key: Config key that finished.
        result: Its worker result.
        elapsed_sec: Wall-clock time the subprocess took.
    """
    if result.get("status") == "ok":
        params = result.get("param_count")
        ne_pct = _ne_placement_pct(result)
        median_ms = _sec_to_ms(_warm_median_sec(result))
        print(
            f"[{key}] done in {elapsed_sec:.1f}s: OK "
            f"(params={_fmt(params, ',')}, NE={_fmt(ne_pct, '.1f')}%, "
            f"warm median={_fmt(median_ms, '.1f')}ms)"
        )
    else:
        stage = result.get("stage", "unknown")
        error = result.get("error", "")
        print(f"[{key}] done in {elapsed_sec:.1f}s: FAILED at stage '{stage}': {error}")


def _run_sweep(args: argparse.Namespace) -> int:
    """Run every requested config as a subprocess and report the results.

    Args:
        args: Parsed sweep-mode (parent) arguments.

    Returns:
        Always 0: a per-config failure is a recorded measurement, not a
        reason to fail the whole sweep.
    """
    config_keys = _resolve_config_keys(args.configs)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for key in config_keys:
        print(f"[{key}] starting (seq_len={args.seq_len}, batch={args.batch})")
        start = time.perf_counter()
        result = _run_config_subprocess(key, args)
        elapsed = time.perf_counter() - start
        results[key] = result
        _print_progress_line(key, result, elapsed)

    payload = {
        "configs_run": config_keys,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "n": args.n,
        "timeout": args.timeout,
        "results": results,
    }
    common.write_result_json(args.output, payload)
    print(f"\nResults written to {args.output}")
    print()
    print_table(results, config_keys)
    print_reference_comparison(results)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments shared by sweep (parent) and worker mode.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sweep synthetic, randomly-initialized Qwen3-shaped configs through the Core ML "
            "conversion pipeline to find where Neural Engine placement or conversion breaks "
            "down as model size grows. See the module docstring for the methodology and its "
            "limits."
        )
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated config keys to run (default: all of " + ", ".join(CONFIGS) + ").",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128, help="Fixed sequence length S (default: %(default)s)."
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Fixed batch size B (default: %(default)s)."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of timed warm predict() calls per config (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Per-config wall-clock timeout in seconds for the worker subprocess "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Scratch directory each worker writes its .mlpackage/.mlmodelc artifacts to "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path the collected sweep result JSON is written to (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep each config's .mlpackage/.mlmodelc instead of deleting them once measured.",
    )
    parser.add_argument(
        "--worker",
        default=None,
        metavar="CONFIG_KEY",
        help="(internal) Run only this single config as a worker subprocess and exit; used by "
        "the sweep to isolate each config in its own process. Do not pass this directly.",
    )
    parser.add_argument(
        "--worker-output",
        type=Path,
        default=None,
        help="(internal) Result JSON path written by --worker mode.",
    )
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> None:
    """Reject option combinations that cannot produce a usable run.

    Args:
        args: Parsed command line arguments.

    Raises:
        SystemExit: With an explanatory message for any invalid option.
    """
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.batch <= 0:
        raise SystemExit("--batch must be a positive integer")
    if args.n <= 0:
        raise SystemExit("--n must be a positive integer")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")


def main(argv: list[str] | None = None) -> int:
    """Dispatch to worker mode or sweep (parent) mode.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    _validate(args)
    if args.worker is not None:
        return _worker_main(args)
    return _run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
