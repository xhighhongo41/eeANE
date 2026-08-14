"""Sweep the S x B latency matrix for ruri-v3-310m / ruri-v3-reranker-310m.

Pipeline (see 開発資料/v0.3実装計画.md §4.5):
for each (S, B) in seq_lens x batches (S outer, B inner, strictly
sequential): convert the model if the compiled artifact is missing (via
``poc/convert_embedding.py`` or ``poc/convert_reranker.py``) -> run
``poc/benchmark_latency.py --compute-plan`` in its own subprocess (one
combination per process, per v0.3実装計画.md §2.5) -> read the result JSON
it wrote and add a summary row -> pause before the next configuration.
Per-configuration failures (conversion or benchmark) are recorded and do not
stop the sweep. Results are written to an aggregate JSON plus a Markdown
table printed to stdout.

Usage:
    uv run python poc/run_sweep.py
    uv run python poc/run_sweep.py --model reranker --seq-lens 512 --batches 1,4,8
    uv run python poc/run_sweep.py --seq-lens 128 --batches 1,2 --skip-convert
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/run_sweep.py` to import the poc package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc.benchmark_latency import default_mlmodelc_path, result_path  # noqa: E402
from poc.common import DEFAULT_MODEL_DIR, DEFAULT_RERANKER_DIR  # noqa: E402

# Which --model choice maps to which default local HF model directory
# (mirrors poc/benchmark_latency.py's _DEFAULT_MODEL_DIRS).
_DEFAULT_MODEL_DIRS: dict[str, Path] = {
    "embedding": DEFAULT_MODEL_DIR,
    "reranker": DEFAULT_RERANKER_DIR,
}

# Which --model choice maps to which conversion script.
_CONVERT_SCRIPTS: dict[str, str] = {
    "embedding": "poc/convert_embedding.py",
    "reranker": "poc/convert_reranker.py",
}

# Conversion subprocess timeout, in seconds (v0.3実装計画.md §4.5: 15 minutes).
CONVERT_TIMEOUT_SEC = 900

# Maximum number of stderr characters kept when recording a subprocess failure.
STDERR_TAIL_CHARS = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Sweep the S x B latency matrix by driving convert_*.py "
        "and benchmark_latency.py once per configuration."
    )
    parser.add_argument(
        "--model",
        choices=list(_DEFAULT_MODEL_DIRS),
        default="embedding",
        help="Which PoC model to sweep (default: embedding).",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="128,256,512,1024",
        help="Comma-separated list of fixed sequence lengths S.",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of fixed batch sizes B.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Assume every configuration is already converted; only benchmark.",
    )
    parser.add_argument(
        "--pause-sec",
        type=float,
        default=10.0,
        help="Cooldown pause between configurations, in seconds (default: 10).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of timed warm predict() calls, passed through to "
        "benchmark_latency.py (default: 50).",
    )
    return parser.parse_args(argv)


def parse_positive_int_list(raw: str, flag: str) -> list[int]:
    """Parse a comma-separated list of positive integers.

    Args:
        raw: Raw comma-separated string, e.g. ``"128,256,512"``.
        flag: Originating CLI flag name, used in error messages.

    Returns:
        Parsed positive integers, in the order they appear in ``raw``.

    Raises:
        SystemExit: If any element is empty, not an integer, or <= 0.
    """
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise SystemExit(f"{flag}: empty element in comma-separated list ({raw!r})")
        try:
            value = int(token)
        except ValueError:
            raise SystemExit(f"{flag}: not an integer: {token!r}") from None
        if value <= 0:
            raise SystemExit(f"{flag}: must be a positive integer, got {value}")
        values.append(value)
    return values


def _run_subprocess(cmd: list[str], timeout: float | None) -> dict[str, Any]:
    """Run one subprocess, capturing stdout/stderr, without raising on failure.

    Args:
        cmd: Full argv, including the interpreter (``sys.executable``).
        timeout: Optional timeout in seconds (``None`` for no limit).

    Returns:
        Dict with ``"success"`` (bool) and, on failure, ``"error"`` (str)
        holding a short description plus up to :data:`STDERR_TAIL_CHARS`
        characters of stderr.
    """
    try:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Partial output may still be available on the exception itself.
        tail = (exc.stderr or "")[-STDERR_TAIL_CHARS:]
        return {"success": False, "error": f"timed out after {timeout}s\n{tail}"}
    except Exception as exc:  # noqa: BLE001 - any subprocess failure is recorded, not fatal
        return {"success": False, "error": f"raised {type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        tail = proc.stderr[-STDERR_TAIL_CHARS:]
        return {"success": False, "error": f"exited with code {proc.returncode}\n{tail}"}
    return {"success": True}


def _error_highlight(error: str) -> str:
    """Pick the single most informative line out of a recorded error string.

    ``error`` strings built by :func:`_run_subprocess` lead with a generic
    prefix (e.g. ``"exited with code 1"``) followed by a captured stderr
    tail; the actual failure reason is typically the last non-empty line of
    that tail (e.g. a ``SystemExit`` message), so it is preferred over the
    generic prefix for concise console/table display.

    Args:
        error: Full error string recorded on a subprocess failure.

    Returns:
        The last non-empty line of ``error``, or ``""`` if there is none.
    """
    non_empty_lines = [line for line in error.splitlines() if line.strip()]
    return non_empty_lines[-1] if non_empty_lines else ""


def convert_config(model: str, seq_len: int, batch: int) -> dict[str, Any]:
    """Run the conversion subprocess for one (S, B) configuration.

    Args:
        model: One of ``"embedding"`` / ``"reranker"``.
        seq_len: Fixed sequence length S.
        batch: Fixed batch size B.

    Returns:
        See :func:`_run_subprocess`.
    """
    cmd = [
        sys.executable,
        _CONVERT_SCRIPTS[model],
        "--seq-len",
        str(seq_len),
        "--batch",
        str(batch),
    ]
    print(f"  [convert] {' '.join(cmd)}")
    step = time.perf_counter()
    outcome = _run_subprocess(cmd, timeout=CONVERT_TIMEOUT_SEC)
    elapsed = time.perf_counter() - step
    if outcome["success"]:
        print(f"  [convert] done ({elapsed:.1f}s)")
    else:
        print(f"  [convert] FAILED ({elapsed:.1f}s): {_error_highlight(outcome['error'])}")
    return outcome


def benchmark_config(model: str, seq_len: int, batch: int, n: int) -> dict[str, Any]:
    """Run the benchmark subprocess for one (S, B) configuration.

    Args:
        model: One of ``"embedding"`` / ``"reranker"``.
        seq_len: Fixed sequence length S.
        batch: Fixed batch size B.
        n: Number of timed warm predict() calls (``--n`` passthrough).

    Returns:
        See :func:`_run_subprocess`.
    """
    cmd = [
        sys.executable,
        "poc/benchmark_latency.py",
        "--model",
        model,
        "--seq-len",
        str(seq_len),
        "--batch",
        str(batch),
        "--compute-plan",
        "--n",
        str(n),
    ]
    print(f"  [bench]   {' '.join(cmd)}")
    step = time.perf_counter()
    # No timeout here: only conversion has an explicit 15-minute budget
    # (v0.3実装計画.md §4.5); benchmark_latency.py's own --n bounds its runtime.
    outcome = _run_subprocess(cmd, timeout=None)
    elapsed = time.perf_counter() - step
    if outcome["success"]:
        print(f"  [bench]   done ({elapsed:.1f}s)")
    else:
        print(f"  [bench]   FAILED ({elapsed:.1f}s): {_error_highlight(outcome['error'])}")
    return outcome


def load_result_summary(model: str, seq_len: int, batch: int) -> dict[str, Any]:
    """Load and summarize the ``benchmark_latency.py`` result JSON for one config.

    Args:
        model: One of ``"embedding"`` / ``"reranker"``.
        seq_len: Fixed sequence length S.
        batch: Fixed batch size B.

    Returns:
        Summary dict with S, B, warm median/p90 seconds, padded tokens/sec,
        NE placement percentage (``None`` if the compute-plan report was
        skipped), load seconds and peak RSS bytes.
    """
    path = result_path(seq_len, "CPU_AND_NE", sustain=False, model=model, batch=batch)
    data = json.loads(path.read_text(encoding="utf-8"))
    plan = data.get("compute_plan") or {}
    ne_placement_pct = None if plan.get("skipped", True) else plan.get("ne_placement_pct")
    return {
        "seq_len": seq_len,
        "batch": batch,
        "median_sec": data["warm"]["median_sec"],
        "p90_sec": data["warm"]["p90_sec"],
        "tokens_per_sec": data["warm"]["tokens_per_sec"],
        "ne_placement_pct": ne_placement_pct,
        "load_sec": data["cold"]["load_sec"],
        "peak_rss_bytes": data["peak_rss_bytes"],
    }


def _pause(pause_sec: float, index: int, total: int) -> None:
    """Sleep between configurations, skipping the pause after the last one.

    Args:
        pause_sec: Requested pause, in seconds (no-op if <= 0).
        index: 1-based index of the configuration just finished.
        total: Total number of configurations in the sweep.
    """
    if pause_sec > 0 and index < total:
        print(f"  pausing {pause_sec:.1f}s before the next configuration")
        time.sleep(pause_sec)


def run_sweep(
    model: str,
    seq_lens: list[int],
    batches: list[int],
    skip_convert: bool,
    pause_sec: float,
    n: int,
) -> dict[str, Any]:
    """Run the S x B sweep sequentially (S outer, B inner) and collect results.

    Args:
        model: One of ``"embedding"`` / ``"reranker"``.
        seq_lens: Sequence lengths S, in the order to sweep (outer loop).
        batches: Batch sizes B, in the order to sweep (inner loop).
        skip_convert: If ``True``, never invoke the conversion subprocess;
            missing artifacts are left for ``benchmark_latency.py`` to reject.
        pause_sec: Cooldown pause between configurations, in seconds.
        n: Number of timed warm predict() calls, passed through per config.

    Returns:
        Dict with ``"configs"`` (execution record per configuration, in run
        order, each holding ``seq_len``/``batch``/``success`` and, on
        failure, ``stage``/``error``) and ``"summaries"`` (summary per
        successful configuration, from :func:`load_result_summary`).
    """
    model_dir = _DEFAULT_MODEL_DIRS[model]
    configs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    total = len(seq_lens) * len(batches)
    index = 0

    for seq_len in seq_lens:
        for batch in batches:
            index += 1
            print(f"[{index}/{total}] S={seq_len} B={batch}")
            record: dict[str, Any] = {"seq_len": seq_len, "batch": batch}

            mlmodelc_path = default_mlmodelc_path(model_dir, seq_len, batch)
            if not mlmodelc_path.exists() and not skip_convert:
                convert_result = convert_config(model, seq_len, batch)
                if not convert_result["success"]:
                    record["success"] = False
                    record["stage"] = "convert"
                    record["error"] = convert_result["error"]
                    configs.append(record)
                    _pause(pause_sec, index, total)
                    continue

            bench_result = benchmark_config(model, seq_len, batch, n)
            if not bench_result["success"]:
                record["success"] = False
                record["stage"] = "benchmark"
                record["error"] = bench_result["error"]
                configs.append(record)
                _pause(pause_sec, index, total)
                continue

            summary = load_result_summary(model, seq_len, batch)
            summaries.append(summary)
            record["success"] = True
            configs.append(record)
            ne_pct = summary["ne_placement_pct"]
            ne_str = f"{ne_pct:.1f}%" if ne_pct is not None else "n/a"
            print(
                f"  OK median={summary['median_sec'] * 1000:.2f}ms "
                f"tokens/s={summary['tokens_per_sec']:.1f} NE%={ne_str}"
            )
            _pause(pause_sec, index, total)

    return {"configs": configs, "summaries": summaries}


def render_markdown_table(
    seq_lens: list[int],
    batches: list[int],
    configs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    """Render the S x B results as a Markdown table plus a failure list.

    Args:
        seq_lens: Sequence lengths S (rows), in the order to display.
        batches: Batch sizes B (columns), in the order to display.
        configs: Per-configuration execution records from :func:`run_sweep`.
        summaries: Per-configuration summaries from :func:`run_sweep`.

    Returns:
        Markdown text: one table (rows=S, columns=B; each cell holds
        ``"{median_ms} / {tokens_per_sec} / {NE%}"`` or ``"FAIL"``) followed
        by a blank line and a failure list (or "Failures: none").
    """
    summary_by_key = {(s["seq_len"], s["batch"]): s for s in summaries}
    failures = [c for c in configs if not c["success"]]

    header = "| S \\ B | " + " | ".join(str(b) for b in batches) + " |"
    separator = "|---|" + "|".join(["---"] * len(batches)) + "|"
    lines = [header, separator]
    for seq_len in seq_lens:
        cells = []
        for batch in batches:
            summary = summary_by_key.get((seq_len, batch))
            if summary is None:
                cells.append("FAIL")
                continue
            ne_pct = summary["ne_placement_pct"]
            ne_str = f"{ne_pct:.1f}%" if ne_pct is not None else "n/a"
            cells.append(
                f"{summary['median_sec'] * 1000:.2f}ms / "
                f"{summary['tokens_per_sec']:.1f} tok/s / {ne_str}"
            )
        lines.append(f"| {seq_len} | " + " | ".join(cells) + " |")

    lines.append("")
    if failures:
        lines.append("Failures:")
        for record in failures:
            highlight = _error_highlight(record["error"])
            lines.append(
                f"- S={record['seq_len']} B={record['batch']} "
                f"(stage={record['stage']}): {highlight}"
            )
    else:
        lines.append("Failures: none")
    return "\n".join(lines)


def build_sweep_record(
    args: argparse.Namespace,
    seq_lens: list[int],
    batches: list[int],
    configs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    """Assemble the aggregate sweep JSON record saved to disk.

    Args:
        args: Parsed CLI arguments.
        seq_lens: Sequence lengths swept, in run order.
        batches: Batch sizes swept, in run order.
        configs: Per-configuration execution records from :func:`run_sweep`.
        summaries: Per-configuration summaries from :func:`run_sweep`.
        started_at: ISO timestamp for the start of the sweep.
        finished_at: ISO timestamp for the end of the sweep.

    Returns:
        Dict ready for ``json.dumps``.
    """
    return {
        "args": {
            "model": args.model,
            "seq_lens": seq_lens,
            "batches": batches,
            "skip_convert": args.skip_convert,
            "pause_sec": args.pause_sec,
            "n": args.n,
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "configs": configs,
        "summaries": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the S x B sweep and write the aggregate JSON plus a Markdown table.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code, always 0: per-configuration failures are recorded
        rather than fatal (v0.3実装計画.md §4.5), so the sweep itself never
        fails outright.
    """
    args = parse_args(argv)
    seq_lens = parse_positive_int_list(args.seq_lens, "--seq-lens")
    batches = parse_positive_int_list(args.batches, "--batches")
    if args.n <= 0:
        raise SystemExit("--n must be a positive integer")
    if args.pause_sec < 0:
        raise SystemExit("--pause-sec must be >= 0")

    print(
        f"Sweeping model={args.model} seq_lens={seq_lens} batches={batches} "
        f"skip_convert={args.skip_convert} n={args.n} pause_sec={args.pause_sec}"
    )
    started_at = datetime.now().isoformat()
    result = run_sweep(args.model, seq_lens, batches, args.skip_convert, args.pause_sec, args.n)
    finished_at = datetime.now().isoformat()

    sweep_record = build_sweep_record(
        args, seq_lens, batches, result["configs"], result["summaries"], started_at, finished_at
    )
    out_dir = _REPO_ROOT / "poc" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_{args.model}_{datetime.now():%Y%m%d}.json"
    out_path.write_text(json.dumps(sweep_record, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(render_markdown_table(seq_lens, batches, result["configs"], result["summaries"]))
    print()
    print(f"summary : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
