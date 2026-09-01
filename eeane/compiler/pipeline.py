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
auto-resolve a ``[[models]]`` entry that only names an ``id``. That
record always describes the batch-1 artifacts, which is what serving is
built on, and names the batched ones separately when the cache holds
any -- so runs of either batch size complete one record between them
rather than overwriting each other's half of it.

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
from eeane.compiler.backends.common import (
    dense_record,
    read_dense_modules,
    read_pooling_mode,
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

# Batch size a server predicts one input at a time with. It is what
# ``model_info.json`` records as the model's artifacts -- and what its
# calibration is aggregated over -- whatever batch size a run compiled,
# since that is the family serving is built on.
SERVING_BATCH_SIZE = 1

# Batch size whose artifacts are recorded alongside them, for the buckets
# one was compiled for, so a server can predict several inputs of one
# request together where that is configured.
BATCHED_BATCH_SIZE = 2

# Key the batched artifacts are recorded under, as a table keyed by batch
# size (a string, JSON object keys being strings) so a record can describe
# more than one of them later on.
BATCH_ARTIFACTS_RECORD_KEY = "batch_artifacts"

# Building block of the long tokenizer-verification input. The gate inputs
# must be self-contained (a user's machine has no repository test data),
# so the long case is generated from this sentence rather than read from a
# corpus file.
_VERIFICATION_LONG_UNIT = "これはトークナイザ凍結検証用の長い日本語の文章です。"

# Degenerate tokenizer-verification inputs: empty, whitespace-only and
# single-character strings (both ASCII and Japanese).
_VERIFICATION_BOUNDARY_TEXTS: tuple[str, ...] = ("", " ", "a", "あ")

# Headline metric of one sanity language set, per model kind, as recorded
# by a self-check report. The progress line below reads whichever of them
# the report carries, so it describes an embedding and a reranker variant
# without knowing which one it was handed.
_SANITY_SET_METRIC_KEYS: tuple[str, ...] = ("cosine_min", "sigmoid_max_abs_diff")


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
        pooling: Pooling mode (``"mean"`` or ``"cls"``) the embedding
            model's sentence-transformers declaration selects, resolved
            once via :func:`_declared_pooling`. ``None`` for a reranker
            (whose pooling belongs to the classification head, not a
            declaration) or when the declaration could not be read; the
            backend's own ``load()`` is what raises on that for an
            embedding model, this field simply has nothing to record.
        dense: Dense projections the embedding model declares after its
            pooling, resolved once via :func:`_declared_dense`; ``None``
            when it declares none, for a reranker, or when the
            declaration could not be read.
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
    pooling: str | None
    dense: tuple[dict[str, Any], ...] | None
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

    Every language set of the sanity fixtures is taken in, not just one:
    the self-check may accept a variant on any of them, and this gate
    compares token sequences, which is language-agnostic -- so covering
    all of them only widens it.

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

    sanity_inputs = list(backend.sanity_spec(kind).all_inputs)
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
    model_dir = sources.resolve_source(args.source, allow_pickle=args.allow_pickle)
    dispatch = resolve_dispatch(model_dir, args.kind)
    backend = dispatch.load_backend()
    kind = dispatch.kind
    output_name = backend.output_name(kind)
    buckets = resolve_buckets(args.buckets, kind)
    _progress(f"      model directory : {model_dir}")
    _progress(f"      architecture    : {dispatch.architecture} -> {dispatch.backend_name}")
    buckets = _apply_max_seq_len(backend, model_dir, buckets, explicit=args.buckets is not None)
    _progress(f"      kind / buckets  : {kind} / {', '.join(str(b) for b in buckets)}")
    pooling = _declared_pooling(kind, model_dir)
    if pooling is not None:
        _progress(f"      pooling         : {pooling}")
    dense = _declared_dense(kind, model_dir)
    if dense is not None:
        _progress(f"      dense           : {_dense_summary(dense)}")

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
        pooling=pooling,
        dense=dense,
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

    # model_info.json describes every batch size the cache holds, whatever
    # this run compiled, so runs of different batch sizes complete the
    # same record instead of overwriting each other's half of it.
    families = _artifact_families(context, batch_size, cache_artifacts)
    served_artifacts = families[SERVING_BATCH_SIZE]
    batched_artifacts = families[BATCHED_BATCH_SIZE]
    if not served_artifacts:
        _progress(
            f"      WARNING: no batch-{SERVING_BATCH_SIZE} artifact was found for this model, "
            "so the recorded artifact table is empty and the model cannot be served; "
            f"serving requires batch-{SERVING_BATCH_SIZE} artifacts -- compile them first "
            f"with '--batch {SERVING_BATCH_SIZE}'"
        )

    # Aggregated across the whole cache, not just this invocation: adding
    # one bucket must re-derive recommended_buckets/embedding_dim from
    # every served bucket, not overwrite them with this run's alone. This
    # run's own reports only describe the family it compiled, so a run of
    # another batch size contributes none of them.
    calibration, recommended_buckets, embedding_dim = aggregate_calibration(
        context.kind,
        served_artifacts,
        run_reports if batch_size == SERVING_BATCH_SIZE else {},
    )

    write_json_record(
        context.model_root / MODEL_INFO_FILENAME,
        _build_model_info(
            context,
            served_artifacts,
            freeze_info,
            freeze_report,
            calibration,
            recommended_buckets,
            embedding_dim,
            batched_artifacts,
        ),
    )

    # A non-default --out-dir must be told to the server too, or it will
    # resolve this id against its own default cache root and find nothing.
    cache_root_hint = out_root if out_root != resolve_out_root(None) else None
    snippet = build_config_snippet(
        model_id=context.model_id,
        kind=context.kind,
        tokenizer_path=context.tokenizer_path,
        artifacts=served_artifacts,
        batch_artifacts=batched_artifacts,
        cache_root_hint=cache_root_hint,
    )
    _progress(_calibration_summary(served_artifacts, recommended_buckets, calibration))
    _progress("[6/6] Done.")
    if args.emit_config is not None:
        write_config_snippet(args.emit_config, snippet)
        _progress(f"      config snippet written to {args.emit_config}")
    _progress("      add the following to your eeane.toml:")
    print(snippet, end="")
    return 0


def _artifact_families(
    context: _CompileContext, batch_size: int, cache_artifacts: Mapping[int, Path]
) -> dict[int, dict[int, Path]]:
    """Collect the artifacts of every batch size the record describes.

    This run's own family is known already (it holds what was just
    compiled, plus what earlier runs left); the other one is read back
    from the cache directory, so a record written by a run of either
    batch size describes both instead of overwriting the other's half.

    Args:
        context: Per-invocation state.
        batch_size: Batch size this invocation compiled for.
        cache_artifacts: This run's family: every one of its buckets now
            present in the cache.

    Returns:
        Bucket -> ``.mlmodelc`` path for :data:`SERVING_BATCH_SIZE` and
        :data:`BATCHED_BATCH_SIZE` (empty for a batch size the cache holds
        nothing for), plus this run's own family when it compiled neither.
    """
    families = {batch_size: dict(cache_artifacts)}
    for recorded in (SERVING_BATCH_SIZE, BATCHED_BATCH_SIZE):
        if recorded in families:
            continue
        families[recorded] = discover_variants(
            context.model_root,
            batch_size=recorded,
            attn=context.args.attn,
            target=context.args.target,
            precision=context.args.precision,
        )
    return families


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


def _declared_pooling(kind: str, model_dir: Path) -> str | None:
    """Best-effort read of the pooling mode an embedding model declares.

    This is for *recording and comparing* what a run compiled, not an
    authoritative gate: an embedding backend's own ``load()`` is what
    raises a full explanation when the declaration is missing or
    unreadable, and that error must reach the user exactly as before.
    This helper must never fail the run just because it could not
    resolve the mode, so it reports ``None`` for anything it cannot
    determine rather than raising.

    Args:
        kind: Resolved model kind.
        model_dir: Read-only model directory.

    Returns:
        ``"mean"`` or ``"cls"`` for an embedding model whose
        ``1_Pooling/config.json`` declares exactly one supported mode.
        ``None`` for a reranker (its pooling belongs to the model's own
        classification head, not a sentence-transformers declaration) or
        when the declaration cannot be read.
    """
    if kind == "reranker":
        return None
    try:
        return read_pooling_mode(model_dir)
    except ValueError:
        return None


def _declared_dense(kind: str, model_dir: Path) -> tuple[dict[str, Any], ...] | None:
    """Best-effort read of the Dense projection an embedding model declares.

    Like :func:`_declared_pooling`, this is for *recording and comparing*
    what a run compiled rather than an authoritative gate: an embedding
    backend's own ``load()`` is what refuses a module chain it cannot
    reproduce, with a full explanation, and that error must reach the user
    unchanged.

    Args:
        kind: Resolved model kind.
        model_dir: Read-only model directory.

    Returns:
        One record per declared projection stage, or ``None`` for a model
        that declares none, for a reranker (whose score comes from its own
        classification head, never from a sentence-transformers chain), or
        when the declaration cannot be read.
    """
    if kind == "reranker":
        return None
    try:
        return dense_record(read_dense_modules(model_dir))
    except ValueError:
        return None


def _dense_summary(dense: Sequence[Mapping[str, Any]]) -> str:
    """Summarize the declared projection for the progress log.

    Args:
        dense: Records of the declared stages, in application order.

    Returns:
        A line like ``"384 -> 1024 (identity)"``, with the stages of a
        multi-stage projection separated by commas.
    """
    return ", ".join(
        f"{stage.get('in')} -> {stage.get('out')} ({stage.get('activation')})" for stage in dense
    )


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
                    pooling=context.pooling,
                    dense=context.dense,
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
    sets_line = _sanity_sets_line(report)
    if sets_line is not None:
        _progress(f"      s{plan.seq_len}: sanity : {sets_line}")
    return report


def _sanity_sets_line(report: Mapping[str, Any]) -> str | None:
    """Summarize a self-check report's per-language sanity sets in one line.

    The accuracy sanity is evaluated once per language set and the variant
    is accepted as soon as one of them clears the threshold, so the number
    the self-check's own summary prints is the *best* set's. This line
    adds what that summary cannot show: which set that was, and how close
    the others came -- the two facts a reader needs to tell a genuinely
    accurate variant from one that only one language happened to carry.

    Args:
        report: A self-check report, as stored under the variant
            metadata's ``selfcheck`` key. Any shape is tolerated: the hook
            is pluggable, and a run must not fail over its progress line.

    Returns:
        A line like ``"pass (best=en 0.99961; ja 0.98923, zh 0.99120)"``,
        or ``None`` when the report carries no per-set measurements at all
        (a skipped self-check, or one that failed before measuring).
    """
    sanity = report.get("sanity")
    if not isinstance(sanity, dict):
        return None
    sets = sanity.get("sets")
    best = sanity.get("best_set")
    if not isinstance(sets, dict) or not isinstance(best, str) or best not in sets:
        return None
    best_report = sets[best]
    if not isinstance(best_report, Mapping):
        return None
    metric_key = next((key for key in _SANITY_SET_METRIC_KEYS if key in best_report), None)
    if metric_key is None:
        return None

    verdict = "pass" if sanity.get("passed") else "fail"
    line = f"{verdict} (best={best} {_sanity_metric(best_report, metric_key)}"
    others = [language for language in sets if language != best]
    if others:
        line += "; " + ", ".join(
            f"{language} {_sanity_metric(sets[language], metric_key)}" for language in others
        )
    return f"{line})"


def _sanity_metric(set_report: Any, metric_key: str) -> str:
    """Format one sanity set's headline metric for the progress line.

    Args:
        set_report: That set's entry in the report's ``sets`` table.
        metric_key: Key the metric is recorded under.

    Returns:
        The value to five decimals, or ``"n/a"`` when the set did not
        record a number under ``metric_key``.
    """
    value = set_report.get(metric_key) if isinstance(set_report, Mapping) else None
    return f"{value:.5f}" if isinstance(value, int | float) else "n/a"


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
        A JSON-serializable metadata record. Its ``variant.pooling`` and
        ``variant.dense`` are ``None`` (JSON ``null``) for a reranker, or
        when the embedding model's declaration could not be read; see
        :attr:`_CompileContext.pooling` and :attr:`_CompileContext.dense`.
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
            "pooling": context.pooling,
            "dense": context.dense,
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
    batch_artifacts: Mapping[int, Path] | None = None,
) -> dict[str, Any]:
    """Assemble the model-level ``model_info.json`` record.

    Args:
        context: Per-invocation state.
        artifacts: Bucket -> ``.mlmodelc`` path of every
            :data:`SERVING_BATCH_SIZE` variant now present in the cache
            (this run's, when it compiled that family, plus the ones kept
            from earlier runs). This is the table serving is built on.
        freeze_info: Result of ``freeze_tokenizer``.
        freeze_report: Result of ``verify_frozen_tokenizer``.
        calibration: Result of :func:`eeane.compiler.artifacts.
            aggregate_calibration`'s first element: the cache-wide
            self-check summary.
        recommended_buckets: That call's second element: the ascending
            buckets ``eeane.config`` should load by default.
        embedding_dim: That call's third element: the shared embedding
            width, or ``None`` for a reranker or an unmeasured cache.
        batch_artifacts: Bucket -> ``.mlmodelc`` path of every
            :data:`BATCHED_BATCH_SIZE` variant in the cache. Recorded
            under :data:`BATCH_ARTIFACTS_RECORD_KEY` when there is any,
            and left out of the record entirely otherwise, so a cache
            without them reads exactly as it did before.

    Returns:
        A JSON-serializable summary; the input ``eeane.config``'s cache
        auto-resolution reads, hence the ``format_version``. The batched
        table is an addition a reader that does not know it simply
        ignores, so it needs no new ``format_version``. ``pooling`` and
        ``dense`` are two more such additions: both are ``None`` (JSON
        ``null``) for a reranker, or when the embedding model's
        declaration could not be read.
    """
    record: dict[str, Any] = {
        "format_version": MODEL_INFO_FORMAT_VERSION,
        "id": context.model_id,
        "kind": context.kind,
        "pooling": context.pooling,
        "dense": context.dense,
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
    if batch_artifacts:
        record[BATCH_ARTIFACTS_RECORD_KEY] = {
            str(BATCHED_BATCH_SIZE): {
                str(seq_len): path.name for seq_len, path in sorted(batch_artifacts.items())
            }
        }
    return record


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
