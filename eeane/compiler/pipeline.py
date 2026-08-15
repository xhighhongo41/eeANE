"""``eeane compile`` driver.

One invocation compiles one model into one or more sequence-length
buckets: the source is resolved (local directory or Hub id), a backend and
model kind are dispatched from ``config.json``, the tokenizer is frozen and
verified, and then -- with the PyTorch model loaded and patched exactly
once -- every bucket is traced, converted, compiled to ``.mlmodelc`` and
described by a metadata JSON file. Once every bucket is done, the
per-bucket self-checks are aggregated into the model-level
``model_info.json`` (``recommended_buckets``, ``calibration``,
``embedding_dim``), which is what :mod:`eeane.config` reads back to
auto-resolve a ``[[models]]`` entry that only names an ``id``.

Everything is written under the cache root (``--out-dir``, default
``~/.cache/eeane``); the input model directory is strictly read-only.

The decisions *about* the outputs (cache layout, artifact naming, bucket
defaults, reuse of existing artifacts, the self-check aggregation, the
generated ``[[models]]`` snippet) live in :mod:`eeane.compiler.artifacts`;
the Core ML steps themselves live in :mod:`eeane.compiler.conversion`.

The self-check (accuracy sanity, ANE placement, latency) is not part of
this module either: :data:`SelfcheckFn` is the hook
:mod:`eeane.compiler.selfcheck` plugs into. When it is unavailable --
and whenever ``--skip-selfcheck`` is given -- every variant records
``selfcheck: {"status": "skipped", ...}`` instead.

Progress messages go to stderr so that stdout carries only the
``[[models]]`` TOML snippet, which stays pipeable into a config file.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeane import __version__
from eeane.compiler import conversion, sources
from eeane.compiler.artifacts import (
    CACHE_SUBDIR,
    METADATA_FORMAT_VERSION,
    MODEL_INFO_FILENAME,
    MODEL_INFO_FORMAT_VERSION,
    SELFCHECK_STATUS_FAILED,
    SELFCHECK_STATUS_SKIPPED,
    TOKENIZER_FILENAME,
    CompileError,
    VariantPlan,
    aggregate_calibration,
    build_config_snippet,
    discover_variants,
    ensure_writable_directory,
    model_cache_name,
    model_identifier,
    needs_conversion,
    resolve_buckets,
    resolve_out_root,
    variant_stem,
    write_config_snippet,
    write_json_record,
)
from eeane.compiler.dispatch import DispatchError, resolve_dispatch
from eeane.compiler.tokenizer_freeze import (
    TokenizerFreezeError,
    freeze_tokenizer,
    verify_frozen_tokenizer,
)

# Reasons recorded when no self-check result is produced.
SELFCHECK_REASON_OPTION = "--skip-selfcheck was given"
SELFCHECK_REASON_UNAVAILABLE = "no self-check implementation was provided"

# Building block of the long tokenizer-verification input. The gate inputs
# must be self-contained (a user's machine has no repository test data),
# so the long case is generated from this sentence rather than read from a
# corpus file.
_VERIFICATION_LONG_UNIT = "これはトークナイザ凍結検証用の長い日本語の文章です。"

# Degenerate tokenizer-verification inputs: empty, whitespace-only and
# single-character strings (both ASCII and Japanese).
_VERIFICATION_BOUNDARY_TEXTS: tuple[str, ...] = ("", " ", "a", "あ")


class SelfcheckFailedError(CompileError):
    """Raised when a variant's self-check reported ``status='failed'``."""


@dataclass(frozen=True)
class SelfcheckContext:
    """Everything a self-check implementation needs for one variant.

    This is the stable hand-off between the pipeline and
    :mod:`eeane.compiler.selfcheck`: the pipeline converts and compiles,
    the self-check measures the compiled artifact and returns a JSON-
    serializable dict that is stored under the metadata's ``selfcheck``
    key.

    Attributes:
        backend: Loaded compile backend instance (e.g.
            ``ModernBertBackend``), for fixtures and reference outputs.
        model_dir: Read-only HuggingFace-format model directory.
        kind: ``"embedding"`` or ``"reranker"``.
        seq_len: Fixed sequence length S of this variant.
        batch_size: Fixed batch size B of this variant.
        output_name: Core ML graph output name requested at conversion.
        mlmodelc_path: Compiled artifact to measure.
        tokenizer_path: Frozen ``tokenizer.json`` of the model.
    """

    backend: Any
    model_dir: Path
    kind: str
    seq_len: int
    batch_size: int
    output_name: str
    mlmodelc_path: Path
    tokenizer_path: Path


# A self-check implementation: given one compiled variant, return a
# JSON-serializable report. A report whose ``status`` is ``"failed"``
# fails the compile.
SelfcheckFn = Callable[[SelfcheckContext], dict[str, Any]]


@dataclass(frozen=True)
class _CompileContext:
    """Per-invocation state shared by every variant conversion.

    Attributes:
        args: Parsed ``eeane compile`` arguments.
        model_dir: Resolved (read-only) model directory.
        model_id: Model id used in the config snippet and model_info.json.
        kind: Resolved model kind.
        output_name: Core ML graph output name for ``kind``.
        batch_size: Fixed batch size B.
        model_root: ``<out-dir>/compiled/<model-name>`` directory.
        tokenizer_path: Frozen ``tokenizer.json`` inside ``model_root``.
        versions: Version block recorded in every metadata file.
        recorded_args: Resolved argument block recorded in metadata.
        backend: Loaded compile backend instance.
        selfcheck_fn: Self-check hook, or ``None`` when unavailable.
    """

    args: argparse.Namespace
    model_dir: Path
    model_id: str
    kind: str
    output_name: str
    batch_size: int
    model_root: Path
    tokenizer_path: Path
    versions: dict[str, str]
    recorded_args: dict[str, Any]
    backend: Any
    selfcheck_fn: SelfcheckFn | None


def run(args: argparse.Namespace, selfcheck_fn: SelfcheckFn | None = None) -> int:
    """Run ``eeane compile`` end to end and return a process exit code.

    Every failure mode a user can act on (unknown source, unsupported
    architecture, unwritable cache, tokenizer mismatch, a conversion that
    blew up, a failed self-check) is turned into a single
    ``eeane compile: <reason>`` line on stderr rather than a traceback.

    Args:
        args: Parsed ``compile`` arguments (see
            :func:`eeane.cli.build_parser`).
        selfcheck_fn: Optional self-check implementation. ``None``
            records ``selfcheck.status = "skipped"`` for every variant.

    Returns:
        ``0`` on success, ``1`` on any reported failure.
    """
    try:
        return _run(args, selfcheck_fn)
    except (
        CompileError,
        DispatchError,
        sources.SourceError,
        TokenizerFreezeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"eeane compile: {exc}", file=sys.stderr)
        return 1


def verification_inputs(
    backend: Any, kind: str, buckets: Sequence[int]
) -> tuple[list[str], list[tuple[str, str]]]:
    """Build the self-contained tokenizer verification inputs.

    The frozen-tokenizer gate must run on any user's machine, so the
    inputs are the backend's own fixtures plus generated boundary cases
    (empty, whitespace-only, single character, and a text far longer than
    the largest bucket) -- never repository test data.

    Args:
        backend: Loaded compile backend instance.
        kind: Resolved model kind.
        buckets: Bucket lengths that will be verified.

    Returns:
        Tuple of (single-sequence texts, ``(query, document)`` pairs),
        both free of duplicates. The pair list is empty for an embedding
        model, which never encodes pairs at request time.
    """
    long_text = _long_verification_text(max(buckets) if buckets else 1)
    texts: list[str] = [*_VERIFICATION_BOUNDARY_TEXTS, long_text]
    pairs: list[tuple[str, str]] = []

    sanity_inputs = list(backend.sanity_spec(kind).inputs)
    if kind == "reranker":
        sanity_pairs = [(query, document) for query, document in sanity_inputs]
        trace_pair = tuple(backend.trace_example(kind))
        pairs = [*sanity_pairs, trace_pair, ("", ""), ("a", long_text)]
        # The single-sequence path is verified too: both halves of every
        # pair are legitimate standalone inputs.
        for query, document in pairs:
            texts += [query, document]
    else:
        texts += sanity_inputs
        texts.append(backend.trace_example(kind))

    return list(dict.fromkeys(texts)), list(dict.fromkeys(pairs))


def _run(args: argparse.Namespace, selfcheck_fn: SelfcheckFn | None) -> int:
    """Execute the compile pipeline (see :func:`run` for the contract).

    Args:
        args: Parsed ``compile`` arguments.
        selfcheck_fn: Optional self-check implementation.

    Returns:
        ``0`` on success.

    Raises:
        CompileError: On any reported failure; :func:`run` renders it.
    """
    batch_size = _validate_batch(args.batch)

    _progress(f"[1/6] Resolving source '{args.source}'")
    model_dir = sources.resolve_source(args.source)
    dispatch = resolve_dispatch(model_dir, args.kind)
    backend = dispatch.load_backend()
    kind = dispatch.kind
    output_name = backend.output_name(kind)
    buckets = resolve_buckets(args.buckets, kind)
    _progress(f"      model directory : {model_dir}")
    _progress(f"      architecture    : {dispatch.architecture} -> {dispatch.backend_name}")
    buckets = _apply_max_seq_len(backend, model_dir, buckets, explicit=args.buckets is not None)
    _progress(f"      kind / buckets  : {kind} / {', '.join(str(b) for b in buckets)}")

    out_root = resolve_out_root(args.out_dir)
    model_root = out_root / CACHE_SUBDIR / model_cache_name(args.source, model_dir)
    ensure_writable_directory(model_root)
    _progress(f"      output directory: {model_root}")

    # The snippet and model_info.json describe the cache as a whole, not
    # just this invocation: same-family buckets compiled by earlier runs
    # stay listed (and the tokenizer is verified for them too).
    existing = discover_variants(
        model_root,
        batch_size=batch_size,
        attn=args.attn,
        target=args.target,
        precision=args.precision,
    )
    previous_buckets = sorted(set(existing) - set(buckets))
    if previous_buckets:
        _progress(
            "      previously compiled bucket(s) kept: "
            + ", ".join(str(bucket) for bucket in previous_buckets)
        )
    cache_buckets = sorted(set(existing) | set(buckets))

    context = _CompileContext(
        args=args,
        model_dir=model_dir,
        model_id=model_identifier(args.source, model_dir),
        kind=kind,
        output_name=output_name,
        batch_size=batch_size,
        model_root=model_root,
        tokenizer_path=model_root / TOKENIZER_FILENAME,
        versions=conversion.build_versions_info(),
        recorded_args=_recorded_args(args, kind, buckets, out_root, batch_size),
        backend=backend,
        selfcheck_fn=selfcheck_fn,
    )

    # The tokenizer is frozen and verified on every run: it costs seconds
    # and it is what guarantees the served artifacts and the tokenizer
    # agree.
    freeze_info, freeze_report = _freeze_and_verify(context, cache_buckets)

    plans = _plan_variants(context, buckets)
    run_reports = _convert_variants(context, plans)

    cache_artifacts = dict(existing)
    cache_artifacts.update({plan.seq_len: plan.mlmodelc_path for plan in plans})

    # Aggregated across the whole cache, not just this invocation: adding
    # one bucket must re-derive recommended_buckets/embedding_dim from
    # every same-family bucket, not overwrite them with this run's alone.
    calibration, recommended_buckets, embedding_dim = aggregate_calibration(
        context.kind, cache_artifacts, run_reports
    )

    write_json_record(
        context.model_root / MODEL_INFO_FILENAME,
        _build_model_info(
            context,
            cache_artifacts,
            freeze_info,
            freeze_report,
            calibration,
            recommended_buckets,
            embedding_dim,
        ),
    )

    # A non-default --out-dir must be told to the server too, or it will
    # resolve this id against its own default cache root and find nothing.
    cache_root_hint = out_root if out_root != resolve_out_root(None) else None
    snippet = build_config_snippet(
        model_id=context.model_id,
        kind=context.kind,
        tokenizer_path=context.tokenizer_path,
        artifacts=cache_artifacts,
        cache_root_hint=cache_root_hint,
    )
    _progress(_calibration_summary(cache_artifacts, recommended_buckets, calibration))
    _progress("[6/6] Done.")
    if args.emit_config is not None:
        write_config_snippet(args.emit_config, snippet)
        _progress(f"      config snippet written to {args.emit_config}")
    _progress("      add the following to your eeane.toml:")
    print(snippet, end="")
    return 0


def _calibration_summary(
    cache_artifacts: Mapping[int, Path],
    recommended_buckets: Sequence[int],
    calibration: Mapping[str, Any],
) -> str:
    """Build the one-line calibration summary printed at the end of a run.

    Args:
        cache_artifacts: Every same-family bucket now present in the cache.
        recommended_buckets: Buckets :func:`aggregate_calibration`
            recommends loading, as recorded in ``model_info.json``.
        calibration: The ``calibration`` record :func:`aggregate_calibration`
            built, consulted for the excluded buckets' recorded status.

    Returns:
        A progress line naming the recommended buckets, plus the excluded
        ones and their reason when the cache holds any.
    """
    recommended = ", ".join(str(bucket) for bucket in recommended_buckets)
    line = f"      calibration : recommended buckets: {recommended}"
    dropped = sorted(set(cache_artifacts) - set(recommended_buckets))
    if dropped:
        buckets = calibration.get("buckets", {})
        reasons = ", ".join(
            f"{bucket} [{buckets.get(str(bucket), {}).get('status')}]" for bucket in dropped
        )
        line += f" (dropped: {reasons})"
    return line


def _apply_max_seq_len(
    backend: Any, model_dir: Path, buckets: Sequence[int], *, explicit: bool
) -> list[int]:
    """Reject or drop buckets the model cannot process.

    A bucket longer than the model's effective maximum sequence length
    would either fail during conversion or silently produce a graph the
    model was never trained for. An explicit ``--buckets`` value is the
    user's decision, so a violation is reported; the kind-specific
    defaults are not, so they are clipped and the clipping is announced.

    Args:
        backend: Loaded compile backend instance.
        model_dir: Read-only model directory the limit is read from.
        buckets: Resolved bucket lengths, ascending.
        explicit: Whether ``buckets`` came from an explicit ``--buckets``
            value rather than from the kind defaults.

    Returns:
        The buckets to compile: ``buckets`` unchanged when the model has
        no known limit or every bucket fits, else the ones that fit.

    Raises:
        CompileError: If an explicitly requested bucket exceeds the limit,
            or if clipping would leave nothing to compile.
    """
    limit = backend.max_seq_len(model_dir)
    if limit is None:
        # The backend cannot determine a limit; nothing to enforce.
        return list(buckets)

    too_long = [bucket for bucket in buckets if bucket > limit]
    if not too_long:
        return list(buckets)

    offending = ", ".join(str(bucket) for bucket in too_long)
    if explicit:
        raise CompileError(
            f"--buckets asks for {offending}, but the model's maximum sequence length "
            f"is {limit}; request buckets of at most {limit}"
        )

    kept = [bucket for bucket in buckets if bucket <= limit]
    for bucket in too_long:
        _progress(
            f"      bucket {bucket} dropped: exceeds the model's max sequence length ({limit})"
        )
    if not kept:
        raise CompileError(
            f"every default bucket ({offending}) exceeds the model's maximum sequence "
            f"length ({limit}); rerun with --buckets set to at most {limit}"
        )
    return kept


def _convert_variants(
    context: _CompileContext, plans: Sequence[VariantPlan]
) -> dict[int, dict[str, Any]]:
    """Convert every planned variant, loading the model at most once.

    Args:
        context: Per-invocation state.
        plans: Variant plans in ascending bucket order.

    Returns:
        This invocation's own self-check reports, keyed by bucket --
        empty for a bucket that was skipped (an up-to-date artifact was
        reused, so nothing ran) and for the run as a whole when nothing
        needed conversion. Fed into :func:`eeane.compiler.artifacts.
        aggregate_calibration` together with the buckets read back from
        disk.

    Raises:
        CompileError: If loading or any conversion fails.
        SelfcheckFailedError: If a variant's self-check failed.
    """
    pending = [plan for plan in plans if plan.convert]
    _progress(
        f"[4/6] Checking existing artifacts ({len(pending)} of {len(plans)} bucket(s) "
        "need conversion)"
    )
    for plan in plans:
        if not plan.convert:
            _progress(
                f"      s{plan.seq_len}: up-to-date artifact found, skipping "
                "(use --force to reconvert)"
            )
    if not pending:
        _progress("      nothing to convert; the model is not loaded")
        return {}

    _progress(f"      loading the model in FP32 (attn={context.args.attn}) and applying patches")
    step = time.perf_counter()
    try:
        loaded = context.backend.load(context.model_dir, context.kind, attn=context.args.attn)
        patches = context.backend.apply_patches(loaded)
    except Exception as exc:
        raise CompileError(f"failed to load the model from '{context.model_dir}': {exc}") from exc
    if not isinstance(patches, dict):
        raise CompileError(
            f"{context.backend.name}.apply_patches() returned "
            f"{type(patches).__name__}, expected a dict"
        )
    load_seconds = time.perf_counter() - step
    _progress(f"      loaded in {load_seconds:.1f}s")

    run_reports: dict[int, dict[str, Any]] = {}
    try:
        for plan in pending:
            run_reports[plan.seq_len] = _convert_variant(
                context, plan, loaded, load_seconds, patches
            )
    finally:
        # ~1.2 GB of FP32 weights for a 310M model: dropping the handle
        # releases the model and its tokenizer as soon as the last bucket
        # is done (or the first one failed).
        del loaded
        gc.collect()
    return run_reports


def _convert_variant(
    context: _CompileContext,
    plan: VariantPlan,
    loaded: Any,
    load_seconds: float,
    patches: Mapping[str, Any],
) -> dict[str, Any]:
    """Trace, convert, compile and describe one bucket.

    Args:
        context: Per-invocation state.
        plan: The variant to build.
        loaded: Handle of the loaded and patched model (the backend's
            ``LoadedModel``).
        load_seconds: Shared model-loading time, recorded in the metadata
            of every variant of this run.
        patches: This run's own ``backend.apply_patches()`` return value,
            recorded verbatim in the variant's metadata.

    Returns:
        This variant's self-check report (possibly ``status="skipped"``),
        so the caller can feed it into the cache-wide calibration without
        reading the metadata file straight back.

    Raises:
        CompileError: If any conversion step fails.
        SelfcheckFailedError: If the self-check reported a failure.
    """
    started = time.perf_counter()
    timings: dict[str, float] = {"load": load_seconds}

    _progress(
        f"[5/6] s{plan.seq_len}: tracing with a fixed "
        f"({context.batch_size}, {plan.seq_len}) example"
    )
    try:
        wrapper = context.backend.wrap(loaded)
        # The single trace example is replicated to B rows so the traced
        # graph already carries the target batch size (PoC behaviour).
        example = context.backend.tokenize(
            loaded,
            [context.backend.trace_example(context.kind)] * context.batch_size,
            plan.seq_len,
        )
        step = time.perf_counter()
        traced = conversion.trace_model(wrapper, example)
        timings["trace"] = time.perf_counter() - step

        _progress(
            f"      s{plan.seq_len}: converting to mlprogram "
            f"({context.args.precision}, {context.args.target})"
        )
        step = time.perf_counter()
        mlmodel = conversion.convert_model(
            traced,
            plan.seq_len,
            context.args.precision,
            context.args.target,
            context.output_name,
            batch_size=context.batch_size,
        )
        timings["convert"] = time.perf_counter() - step
        del traced, wrapper
        gc.collect()

        if plan.mlpackage_path.exists():
            shutil.rmtree(plan.mlpackage_path)
        mlmodel.save(str(plan.mlpackage_path))
        del mlmodel
        gc.collect()

        _progress(f"      s{plan.seq_len}: compiling to {plan.mlmodelc_path.name}")
        step = time.perf_counter()
        conversion.compile_model(plan.mlpackage_path, plan.mlmodelc_path)
        timings["compile"] = time.perf_counter() - step
    except CompileError:
        raise
    except Exception as exc:
        # A half-written intermediate would only confuse the next run (and
        # waste hundreds of MB); the previous .mlmodelc, if any, is left
        # untouched because compile_model replaces it only on success.
        shutil.rmtree(plan.mlpackage_path, ignore_errors=True)
        raise CompileError(
            f"failed to convert bucket {plan.seq_len} of '{context.model_dir}': {exc}"
        ) from exc

    artifacts = {"mlmodelc": str(plan.mlmodelc_path)}
    if context.args.keep_mlpackage:
        artifacts["mlpackage"] = str(plan.mlpackage_path)
    else:
        # Hundreds of MB per bucket; the .mlmodelc is what gets served.
        shutil.rmtree(plan.mlpackage_path, ignore_errors=True)

    selfcheck = _run_selfcheck(context, plan)
    timings["total"] = time.perf_counter() - started + load_seconds
    write_json_record(
        plan.metadata_path,
        _build_metadata(context, plan, timings, artifacts, patches, selfcheck),
    )
    _progress(
        f"      s{plan.seq_len}: done in {timings['total']:.1f}s -> {plan.mlmodelc_path.name}"
    )

    if selfcheck.get("status") == SELFCHECK_STATUS_FAILED:
        raise SelfcheckFailedError(
            f"the self-check of bucket {plan.seq_len} failed; see {plan.metadata_path}"
        )
    return selfcheck


def _plan_variants(context: _CompileContext, buckets: Sequence[int]) -> list[VariantPlan]:
    """Resolve every bucket's paths and decide what has to be converted.

    Args:
        context: Per-invocation state.
        buckets: Ascending bucket lengths.

    Returns:
        One :class:`~eeane.compiler.artifacts.VariantPlan` per bucket, in
        the given order.
    """
    plans: list[VariantPlan] = []
    for seq_len in buckets:
        stem = variant_stem(
            seq_len,
            context.batch_size,
            context.args.attn,
            context.args.target,
            context.args.precision,
        )
        mlmodelc_path = context.model_root / f"{stem}.mlmodelc"
        metadata_path = context.model_root / f"{stem}.json"
        plans.append(
            VariantPlan(
                seq_len=seq_len,
                stem=stem,
                mlpackage_path=context.model_root / f"{stem}.mlpackage",
                mlmodelc_path=mlmodelc_path,
                metadata_path=metadata_path,
                convert=needs_conversion(
                    mlmodelc_path,
                    metadata_path,
                    context.versions,
                    force=bool(context.args.force),
                ),
            )
        )
    return plans


def _freeze_and_verify(
    context: _CompileContext, buckets: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the tokenizer and prove it reproduces ``AutoTokenizer``.

    Args:
        context: Per-invocation state.
        buckets: Bucket lengths to verify at.

    Returns:
        Tuple of the freeze information and the verification report.

    Raises:
        CompileError: If freezing failed for an unexpected reason.
        TokenizerFreezeError: If the frozen tokenizer disagrees with
            ``AutoTokenizer`` (the compile gate).
    """
    _progress(f"[2/6] Freezing the tokenizer to {context.tokenizer_path}")
    try:
        freeze_info = freeze_tokenizer(context.model_dir, context.tokenizer_path)
    except TokenizerFreezeError:
        raise
    except Exception as exc:
        raise CompileError(
            f"failed to freeze the tokenizer of '{context.model_dir}': {exc}"
        ) from exc
    _progress(f"      frozen from {freeze_info['tokenizer_class']}")

    texts, pairs = verification_inputs(context.backend, context.kind, buckets)
    _progress(
        f"[3/6] Verifying the frozen tokenizer against AutoTokenizer "
        f"({len(texts)} text(s), {len(pairs)} pair(s) x {len(buckets)} bucket(s))"
    )
    report = verify_frozen_tokenizer(
        context.model_dir, context.tokenizer_path, texts, pairs, list(buckets)
    )
    _progress("      tokenizer verification passed")
    return freeze_info, report


def _run_selfcheck(context: _CompileContext, plan: VariantPlan) -> dict[str, Any]:
    """Run the self-check hook for one variant, or record why it was skipped.

    Args:
        context: Per-invocation state.
        plan: The variant that was just compiled.

    Returns:
        The self-check report to store in the variant metadata.

    Raises:
        CompileError: If the hook raised, or returned something that is
            not a report dict.
    """
    if context.args.skip_selfcheck:
        _progress(f"      s{plan.seq_len}: self-check skipped ({SELFCHECK_REASON_OPTION})")
        return {"status": SELFCHECK_STATUS_SKIPPED, "reason": SELFCHECK_REASON_OPTION}
    if context.selfcheck_fn is None:
        _progress(f"      s{plan.seq_len}: self-check skipped ({SELFCHECK_REASON_UNAVAILABLE})")
        return {"status": SELFCHECK_STATUS_SKIPPED, "reason": SELFCHECK_REASON_UNAVAILABLE}

    _progress(f"      s{plan.seq_len}: running the self-check")
    try:
        report = context.selfcheck_fn(
            SelfcheckContext(
                backend=context.backend,
                model_dir=context.model_dir,
                kind=context.kind,
                seq_len=plan.seq_len,
                batch_size=context.batch_size,
                output_name=context.output_name,
                mlmodelc_path=plan.mlmodelc_path,
                tokenizer_path=context.tokenizer_path,
            )
        )
    except Exception as exc:
        raise CompileError(f"the self-check of bucket {plan.seq_len} raised: {exc}") from exc
    if not isinstance(report, dict):
        raise CompileError(
            f"the self-check of bucket {plan.seq_len} returned {type(report).__name__}, "
            "expected a report dict"
        )
    return report


def _build_metadata(
    context: _CompileContext,
    plan: VariantPlan,
    timings: Mapping[str, float],
    artifacts: Mapping[str, str],
    patches: Mapping[str, Any],
    selfcheck: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one variant's metadata record.

    Args:
        context: Per-invocation state.
        plan: The variant the record describes.
        timings: Per-step durations in seconds.
        artifacts: Produced artifact paths.
        patches: The backend's own ``apply_patches()`` return value for
            this run (empty for an architecture that needs none), so this
            records what was actually applied rather than an assumption
            tied to one architecture.
        selfcheck: Self-check report (possibly ``status="skipped"``).

    Returns:
        A JSON-serializable metadata record.
    """
    return {
        "format_version": METADATA_FORMAT_VERSION,
        "source": {"requested": context.args.source, "resolved": str(context.model_dir)},
        "variant": {
            "stem": plan.stem,
            "seq_len": plan.seq_len,
            "batch_size": context.batch_size,
            "kind": context.kind,
            "output_name": context.output_name,
        },
        "args": dict(context.recorded_args),
        "versions": dict(context.versions),
        "patches": dict(patches),
        "timings_sec": {key: round(value, 3) for key, value in timings.items()},
        "artifacts": dict(artifacts),
        "selfcheck": dict(selfcheck),
    }


def _build_model_info(
    context: _CompileContext,
    artifacts: Mapping[int, Path],
    freeze_info: Mapping[str, Any],
    freeze_report: Mapping[str, Any],
    calibration: Mapping[str, Any],
    recommended_buckets: Sequence[int],
    embedding_dim: int | None,
) -> dict[str, Any]:
    """Assemble the model-level ``model_info.json`` record.

    Args:
        context: Per-invocation state.
        artifacts: Bucket -> ``.mlmodelc`` path of every same-family
            variant now present in the cache (this run's plus the ones
            kept from earlier runs).
        freeze_info: Result of ``freeze_tokenizer``.
        freeze_report: Result of ``verify_frozen_tokenizer``.
        calibration: Result of :func:`eeane.compiler.artifacts.
            aggregate_calibration`'s first element: the cache-wide
            self-check summary.
        recommended_buckets: That call's second element: the ascending
            buckets ``eeane.config`` should load by default.
        embedding_dim: That call's third element: the shared embedding
            width, or ``None`` for a reranker or an unmeasured cache.

    Returns:
        A JSON-serializable summary; the input ``eeane.config``'s cache
        auto-resolution reads, hence the ``format_version``.
    """
    return {
        "format_version": MODEL_INFO_FORMAT_VERSION,
        "id": context.model_id,
        "kind": context.kind,
        "output_name": context.output_name,
        "buckets": sorted(artifacts),
        "tokenizer": TOKENIZER_FILENAME,
        "tokenizer_freeze": {
            "verified": bool(freeze_report.get("passed")),
            "tokenizer_class": freeze_info.get("tokenizer_class"),
            "pad_id": freeze_info.get("pad_id"),
            "pad_token": freeze_info.get("pad_token"),
            "padding_direction": freeze_info.get("padding_direction"),
            "buckets": list(freeze_report.get("buckets", [])),
            "n_texts": freeze_report.get("n_texts"),
            "n_pairs": freeze_report.get("n_pairs"),
            "n_comparisons": freeze_report.get("n_comparisons"),
        },
        "artifacts": {str(seq_len): path.name for seq_len, path in sorted(artifacts.items())},
        "embedding_dim": embedding_dim,
        "recommended_buckets": list(recommended_buckets),
        "calibration": dict(calibration),
        "eeane_version": __version__,
    }


def _recorded_args(
    args: argparse.Namespace,
    kind: str,
    buckets: Sequence[int],
    out_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Build the resolved-argument block stored in every metadata file.

    Args:
        args: Parsed ``compile`` arguments.
        kind: Resolved model kind (``args.kind`` may be ``"auto"``).
        buckets: Resolved bucket lengths.
        out_root: Resolved cache root.
        batch_size: Validated batch size.

    Returns:
        A JSON-serializable dict of the effective settings.
    """
    return {
        "source": args.source,
        "kind": args.kind,
        "resolved_kind": kind,
        "buckets": list(buckets),
        "out_dir": str(out_root),
        "batch": batch_size,
        "precision": args.precision,
        "target": args.target,
        "attn": args.attn,
        "keep_mlpackage": bool(args.keep_mlpackage),
        "skip_selfcheck": bool(args.skip_selfcheck),
        "force": bool(args.force),
    }


def _validate_batch(batch: int) -> int:
    """Validate ``--batch`` before anything is loaded or written.

    Args:
        batch: Raw ``--batch`` value.

    Returns:
        The validated batch size.

    Raises:
        CompileError: If it is not a positive integer.
    """
    try:
        batch_size = int(batch)
    except (TypeError, ValueError) as exc:
        raise CompileError(f"--batch must be a positive integer (got {batch!r})") from exc
    if batch_size <= 0:
        raise CompileError(f"--batch must be a positive integer (got {batch_size})")
    return batch_size


def _long_verification_text(max_bucket: int) -> str:
    """Build a text that comfortably exceeds ``max_bucket`` tokens.

    Args:
        max_bucket: Largest bucket length that will be verified.

    Returns:
        A generated Japanese text (no repository data involved).
    """
    return _VERIFICATION_LONG_UNIT * max(4, max_bucket // 4)


def _progress(message: str) -> None:
    """Print a progress line to stderr.

    stdout is reserved for the ``[[models]]`` snippet so that
    ``eeane compile ... > snippet.toml`` stays useful.

    Args:
        message: Line to print.
    """
    print(message, file=sys.stderr, flush=True)
