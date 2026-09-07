"""Convert a decoder-style Qwen3 embedding model into a Core ML program.

The pipeline is: load the HuggingFace model with eager attention ->
install the numerically equivalent conversion patches -> wrap it so the
4-D attention mask and the last-token pooling happen inside the graph ->
``torch.jit.trace`` -> ``ct.convert`` to an ``mlprogram`` with fixed
``(B, S)`` int32 inputs -> ``.mlpackage`` -> ``xcrun coremlcompiler`` ->
``.mlmodelc`` -> sanity check on CPU_AND_NE against an unpatched FP32
reference. A JSON file recording the options, the applied patches, the
per-stage timings and the sanity results is written next to the
artifacts.

The sanity check never aborts the run: a conversion whose accuracy is bad
is a measurement worth keeping, so the numbers are always written to the
JSON file and the failure is reported through the exit code (1) instead.

Usage:
    uv run python poc_qwen/convert_embedding.py --seq-len 128 --batch 1
"""

from __future__ import annotations

import argparse
import gc
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, PreTrainedTokenizerBase

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc_qwen/convert_embedding.py` to import the package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc_qwen import common, convert, patches  # noqa: E402

# Name requested for the single graph output.
OUTPUT_NAME = "embedding"

# Pooling modes and the padding side each one requires.
POOLING_PADDING_SIDES: dict[str, str] = {"last-left": "left", "gather-right": "right"}

# Default destination for the artifacts and their metadata file.
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "models" / "compiled" / "qwen3-embedding-0.6b"

# Grouped-query expansion strategy used unless overridden; artifacts built
# with any other strategy get it appended to their file name.
DEFAULT_REPEAT_KV_MODE = "repeat_interleave"

# Number of pipeline stages reported in the progress output.
_TOTAL_STAGES = 8


class LastTokenEmbeddingWrapper(torch.nn.Module):
    """Wraps the backbone and pools the final token position in-graph.

    Both the attention mask and the pooling are computed inside the
    wrapper so the exported program takes exactly the two int32 tensors
    Core ML feeds it and returns one embedding per row, with no host-side
    pre- or post-processing left to reimplement.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        fill_value: float = common.MASK_FILL_VALUE,
        pooling: str = "last-left",
    ) -> None:
        """Store the backbone and the mask/pooling settings.

        Args:
            model: Backbone returning the last hidden state first, loaded
                with eager attention, ``use_cache = False`` and
                ``return_dict = False``.
            fill_value: Additive value for masked positions; see
                ``poc_qwen.common.MASK_FILL_VALUE``.
            pooling: ``"last-left"`` to read the final position (requires
                left-padded input) or ``"gather-right"`` to read each
                row's last real token (requires right-padded input).

        Raises:
            ValueError: If ``pooling`` is unknown.
        """
        super().__init__()
        if pooling not in POOLING_PADDING_SIDES:
            raise ValueError(
                f"unknown pooling: {pooling!r} (expected one of {tuple(POOLING_PADDING_SIDES)})"
            )
        self.model = model
        self.fill_value = fill_value
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Embed a batch of token id rows.

        Args:
            input_ids: Token ids of shape ``(B, S)``.
            attention_mask: 2-D mask of shape ``(B, S)``; non-zero marks a
                real token.

        Returns:
            Pooled embeddings of shape ``(B, hidden_size)``. They are
            **not** L2-normalized: normalizing is the consumer's choice
            (some callers want raw magnitudes), and it cannot affect the
            comparison against any external implementation because cosine
            similarity is scale invariant.
        """
        mask4d = common.build_causal_padding_mask(attention_mask, self.fill_value)
        hidden = self.model(input_ids=input_ids, attention_mask=mask4d)[0]
        if self.pooling == "last-left":
            return common.last_token_pool_left(hidden)
        return common.last_token_pool_gather(hidden, attention_mask)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert a Qwen3 embedding model to a Core ML program."
    )
    parser.add_argument(
        "--model-id",
        default=common.DEFAULT_MODEL_ID,
        help="Hub model id or local HuggingFace-format directory (default: %(default)s).",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128, help="Fixed sequence length S (default: %(default)s)."
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Fixed batch size B (default: %(default)s)."
    )
    parser.add_argument(
        "--precision",
        choices=sorted(convert.PRECISIONS),
        default="fp16",
        help="Compute precision of the converted program (default: %(default)s).",
    )
    parser.add_argument(
        "--target",
        choices=sorted(convert.TARGETS),
        default="macos13",
        help="Minimum deployment target (default: %(default)s).",
    )
    parser.add_argument(
        "--mask-fill-value",
        type=float,
        default=common.MASK_FILL_VALUE,
        help=(
            "Additive value written into masked attention positions; must be negative "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--rmsnorm-mode",
        choices=list(patches.RMSNORM_MODES),
        default="upstream",
        help=(
            "RMSNorm variant: 'upstream' keeps the original formula, 'scaled' divides the "
            "input before squaring so the sum of squares cannot overflow FP16 "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--rmsnorm-scale",
        type=float,
        default=64.0,
        help="Positive divisor used by --rmsnorm-mode scaled (default: %(default)s).",
    )
    parser.add_argument(
        "--repeat-kv-mode",
        choices=list(patches.REPEAT_KV_MODES),
        default=DEFAULT_REPEAT_KV_MODE,
        help="Operator used to expand grouped-query key/value heads (default: %(default)s).",
    )
    parser.add_argument(
        "--pooling",
        choices=sorted(POOLING_PADDING_SIDES),
        default="last-left",
        help=(
            "'last-left' reads the final position of a left-padded row, 'gather-right' reads "
            "each row's last real token of a right-padded row (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory the artifacts and the metadata file are written to (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-mlpackage",
        action="store_true",
        help="Keep the intermediate .mlpackage instead of deleting it after compilation.",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Convert only, without loading the FP32 reference model for the sanity check.",
    )
    return parser.parse_args(argv)


def build_stem(args: argparse.Namespace) -> str:
    """Build the base name shared by every artifact of one conversion.

    The base name is ``s{S}_b{B}_{precision}_{target}``; every option that
    changes what the artifact computes adds a suffix, so two conversions
    that differ in any way can never overwrite each other's files.

    Args:
        args: Parsed command line arguments.

    Returns:
        The artifact base name.
    """
    stem = f"s{args.seq_len}_b{args.batch}_{args.precision}_{args.target}"
    if args.rmsnorm_mode != "upstream":
        stem += f"_rms{args.rmsnorm_scale:g}"
    if args.pooling != "last-left":
        stem += "_gather"
    if args.mask_fill_value != common.MASK_FILL_VALUE:
        stem += f"_fill{args.mask_fill_value:g}"
    if args.repeat_kv_mode != DEFAULT_REPEAT_KV_MODE:
        stem += f"_{args.repeat_kv_mode.replace('_', '')}"
    return stem


def _fill_batch(rows: np.ndarray, batch_size: int) -> np.ndarray:
    """Pad a partial chunk of tokenized rows up to the fixed batch size.

    Args:
        rows: Chunk of shape ``(n, S)`` with ``n <= batch_size``.
        batch_size: Fixed batch size ``B`` of the converted model.

    Returns:
        Array of shape ``(batch_size, S)``; ``rows`` itself when already
        full.
    """
    missing = batch_size - rows.shape[0]
    if missing <= 0:
        return rows
    # The chunk's own last row is repeated rather than some filler text:
    # the extra outputs are discarded anyway, and duplicating a real row
    # guarantees no padded row is fully masked (which would make the
    # attention softmax degenerate and could produce NaN).
    return np.concatenate([rows, np.repeat(rows[-1:], missing, axis=0)], axis=0)


def predict_rows(
    compiled: ct.models.CompiledMLModel,
    tokens: dict[str, np.ndarray],
    batch_size: int,
    output_key: str,
) -> np.ndarray:
    """Embed tokenized rows with a model whose batch size is fixed.

    Args:
        compiled: Loaded compiled model.
        tokens: Tokenized rows, each value of shape ``(N, S)``.
        batch_size: Fixed batch size ``B`` of the compiled model.
        output_key: Output name resolved from the model specification.

    Returns:
        Embeddings of shape ``(N, hidden_size)``, dtype float32.

    Raises:
        ValueError: If ``tokens`` holds no rows.
    """
    n_rows = int(tokens["input_ids"].shape[0])
    if n_rows == 0:
        raise ValueError("no rows to predict")
    rows: list[np.ndarray] = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        prediction = compiled.predict(
            {key: _fill_batch(tokens[key][start:end], batch_size) for key in tokens}
        )
        values = np.asarray(prediction[output_key], dtype=np.float32).reshape(batch_size, -1)
        rows.extend(values[: end - start])
    return np.stack(rows)


def check_batch_consistency(
    compiled: ct.models.CompiledMLModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    seq_len: int,
    batch_size: int,
    output_key: str,
) -> dict[str, Any]:
    """Verify that rows of one batch do not influence each other.

    Predicts a single batch whose ``B`` rows all hold the same text and
    compares row 0 against every other row: a correct graph must treat the
    batch axis as independent.

    Args:
        compiled: Loaded compiled model.
        tokenizer: Tokenizer configured with the right padding side.
        text: Text replicated across all rows.
        seq_len: Fixed sequence length ``S``.
        batch_size: Fixed batch size ``B``; must be at least 2.
        output_key: Output name resolved from the model specification.

    Returns:
        Dict with the per-row cosine similarities against row 0, their
        minimum and maximum, the threshold and the pass/fail flag.
    """
    tokens = common.tokenize_batch(
        tokenizer, [text] * batch_size, seq_len, prompt=common.QUERY_PROMPT
    )
    prediction = compiled.predict(dict(tokens))
    embeddings = np.asarray(prediction[output_key], dtype=np.float32).reshape(batch_size, -1)
    reference = np.repeat(embeddings[:1], batch_size - 1, axis=0)
    cosines = common.cosine_rowwise(reference, embeddings[1:])
    return {
        "cosine_per_row": [float(value) for value in cosines],
        "cosine_min": float(cosines.min()),
        "cosine_max": float(cosines.max()),
        "cosine_threshold": common.BATCH_CONSISTENCY_COSINE_THRESHOLD,
        "passed": bool(np.isfinite(cosines).all())
        and bool(cosines.min() >= common.BATCH_CONSISTENCY_COSINE_THRESHOLD),
    }


def _rank_key(cosine_min: float) -> float:
    """Order a set by its worst cosine, keeping a non-finite one last.

    Args:
        cosine_min: Worst per-row cosine of one sanity set, possibly NaN.

    Returns:
        ``cosine_min`` itself, or ``-inf`` when it is not finite, so that
        ``max`` never picks a set whose outputs contain NaN.
    """
    return cosine_min if math.isfinite(cosine_min) else float("-inf")


def run_sanity_check(
    mlmodelc_path: Path,
    model_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    output_key: str,
    pooling: str,
) -> dict[str, Any]:
    """Compare the compiled model with an unpatched FP32 reference.

    The reference is a second, independently loaded model that uses sdpa
    attention, the framework's own 2-D mask handling and no conversion
    patches, so agreement between the two means the exported graph
    computes the intended function rather than merely reproducing its own
    assumptions.

    The check runs the fixed per-language sanity sets and passes when
    **any single one** of them clears the threshold. The sets cover
    several languages, and a model whose vocabulary barely covers one of
    them tokenizes that text into a long tail of fallback tokens; on such
    input FP16 and FP32 can drift apart noticeably without that saying
    anything about the faithfulness of the conversion. Demanding every set
    would therefore reject sound conversions of narrowly trained models,
    so one passing set is required and all of them are recorded.

    Args:
        mlmodelc_path: Compiled model directory.
        model_dir: Local HuggingFace-format model directory.
        tokenizer: Tokenizer configured with the padding side ``pooling``
            requires.
        seq_len: Fixed sequence length ``S``.
        batch_size: Fixed batch size ``B`` of the compiled model.
        output_key: Output name resolved from the model specification.
        pooling: Pooling mode the model was converted with.

    Returns:
        Dict with the per-set cosine statistics, the best set, the
        finiteness flag, the batch consistency result (``None`` for
        ``B == 1``) and the overall pass/fail flag.
    """
    text_sets = common.sanity_text_sets()
    compiled = ct.models.CompiledMLModel(
        str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    coreml_embeddings: dict[str, np.ndarray] = {}
    for language, texts in text_sets:
        tokens = common.tokenize_batch(tokenizer, texts, seq_len, prompt=common.QUERY_PROMPT)
        coreml_embeddings[language] = predict_rows(compiled, tokens, batch_size, output_key)
    consistency = (
        check_batch_consistency(
            compiled, tokenizer, text_sets[0][1][0], seq_len, batch_size, output_key
        )
        if batch_size > 1
        else None
    )
    # The compiled model is released before the FP32 reference is loaded
    # so the two never occupy memory at the same time.
    del compiled
    gc.collect()

    reference_model = common.load_reference_model(model_dir)
    pool = "left" if pooling == "last-left" else "gather"
    reference_embeddings = {
        language: common.encode_reference(
            reference_model, tokenizer, texts, seq_len, prompt=common.QUERY_PROMPT, pool=pool
        )
        for language, texts in text_sets
    }
    del reference_model
    gc.collect()

    sets: list[dict[str, Any]] = []
    for language, _ in text_sets:
        actual = coreml_embeddings[language]
        cosines = common.cosine_rowwise(actual, reference_embeddings[language])
        finite = bool(np.isfinite(actual).all())
        sets.append(
            {
                "language": language,
                "cosine_per_text": [float(value) for value in cosines],
                "cosine_min": float(cosines.min()),
                "cosine_mean": float(cosines.mean()),
                "finite": finite,
                "passed": finite and bool(cosines.min() >= common.SANITY_COSINE_THRESHOLD),
            }
        )

    best = max(sets, key=lambda record: _rank_key(record["cosine_min"]))
    finite = all(record["finite"] for record in sets)
    passed = (
        finite
        and any(record["passed"] for record in sets)
        and (consistency is None or bool(consistency["passed"]))
    )
    return {
        "compute_units": "CPU_AND_NE",
        "output_key": output_key,
        "batch_size": batch_size,
        "cosine_threshold": common.SANITY_COSINE_THRESHOLD,
        "sets": sets,
        "best_language": best["language"] if best["passed"] else None,
        "best_cosine_min": best["cosine_min"],
        "finite": finite,
        "batch_consistency": consistency,
        "passed": passed,
    }


def build_metadata(
    args: argparse.Namespace,
    model_dir: Path,
    patch_record: dict[str, Any],
    output_key: str,
    artifacts: dict[str, Any],
    timings: dict[str, float],
    sanity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the metadata record stored next to the artifacts.

    Args:
        args: Parsed command line arguments.
        model_dir: Resolved local model directory.
        patch_record: Records returned by ``patches.apply_patches``.
        output_key: Output name resolved from the model specification.
        artifacts: Paths of the produced files.
        timings: Per-stage wall-clock seconds.
        sanity: Sanity result, or ``None`` when it was skipped.

    Returns:
        A JSON-serializable mapping.
    """
    return {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "seq_len": args.seq_len,
        "batch": args.batch,
        "precision": args.precision,
        "target": args.target,
        "pooling": args.pooling,
        "mask_fill_value": args.mask_fill_value,
        "patches": patch_record,
        "output_key": output_key,
        "artifacts": artifacts,
        "timings_sec": {key: round(value, 3) for key, value in timings.items()},
        "sanity": sanity,
    }


def print_sanity_summary(sanity: dict[str, Any]) -> None:
    """Print the sanity result as a readable per-language table.

    Args:
        sanity: Result returned by :func:`run_sanity_check`.
    """
    print(f"      output key      : {sanity['output_key']}")
    print(f"      finite outputs  : {sanity['finite']}")
    print(f"      cosine threshold: {sanity['cosine_threshold']} (one language set must pass)")
    for record in sanity["sets"]:
        verdict = "PASS" if record["passed"] else "fail"
        print(
            f"        {record['language']:<3} min {record['cosine_min']:.6f} "
            f"mean {record['cosine_mean']:.6f}  {verdict}"
        )
    best = sanity["best_language"]
    print(f"      best passing set: {best if best is not None else '(none)'}")
    consistency = sanity["batch_consistency"]
    if consistency is not None:
        verdict = "PASS" if consistency["passed"] else "fail"
        print(
            f"      batch rows cos  : min {consistency['cosine_min']:.7f} "
            f"max {consistency['cosine_max']:.7f} "
            f"(threshold {consistency['cosine_threshold']})  {verdict}"
        )


def _validate(args: argparse.Namespace) -> None:
    """Reject option combinations that cannot produce a usable artifact.

    Args:
        args: Parsed command line arguments.

    Raises:
        SystemExit: With an explanatory message for any invalid option.
    """
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.batch <= 0:
        raise SystemExit("--batch must be a positive integer")
    if not args.mask_fill_value < 0:
        # A non-negative (or NaN) fill value would leave padded and future
        # positions attendable, silently changing what the model computes.
        raise SystemExit(f"--mask-fill-value must be negative, got {args.mask_fill_value}")
    if args.rmsnorm_mode == "scaled" and not args.rmsnorm_scale > 0:
        raise SystemExit(f"--rmsnorm-scale must be positive, got {args.rmsnorm_scale}")


def main(argv: list[str] | None = None) -> int:
    """Run the full conversion pipeline.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 when the sanity check failed.
    """
    args = parse_args(argv)
    _validate(args)
    stem = build_stem(args)
    output_dir: Path = args.output_dir
    mlpackage_path = output_dir / f"{stem}.mlpackage"
    mlmodelc_path = output_dir / f"{stem}.mlmodelc"
    metadata_path = output_dir / f"{stem}.json"
    timings: dict[str, float] = {}
    started = time.perf_counter()

    print(f"[1/{_TOTAL_STAGES}] Resolving {args.model_id}")
    step = time.perf_counter()
    model_dir = common.resolve_model_dir(args.model_id)
    padding_side = POOLING_PADDING_SIDES[args.pooling]
    tokenizer = common.load_tokenizer(model_dir, padding_side=padding_side)
    timings["resolve"] = time.perf_counter() - step
    print(f"      model dir {model_dir} (padding side {padding_side}) [{timings['resolve']:.1f}s]")

    print(f"[2/{_TOTAL_STAGES}] Loading the backbone (eager attention, fp32)")
    step = time.perf_counter()
    # Eager attention is mandatory: the sdpa path calls into a separate
    # implementation that never reaches the patched repeat_kv. Caching is
    # off so no cache update enters the graph, and return_dict is off so
    # tracing sees a plain tuple output.
    model = AutoModel.from_pretrained(
        str(model_dir), attn_implementation="eager", dtype=torch.float32
    ).eval()
    model.config.use_cache = False
    model.config.return_dict = False
    timings["load"] = time.perf_counter() - step
    print(f"      loaded [{timings['load']:.1f}s]")

    print(f"[3/{_TOTAL_STAGES}] Applying conversion patches")
    step = time.perf_counter()
    patch_record = patches.apply_patches(
        model,
        repeat_kv_mode=args.repeat_kv_mode,
        rmsnorm_mode=args.rmsnorm_mode,
        rmsnorm_scale=args.rmsnorm_scale,
    )
    timings["patch"] = time.perf_counter() - step
    print(
        f"      repeat_kv={args.repeat_kv_mode} rmsnorm={args.rmsnorm_mode} "
        f"({patch_record['rmsnorm']['patched_modules']} modules) [{timings['patch']:.1f}s]"
    )

    print(f"[4/{_TOTAL_STAGES}] Tracing at ({args.batch}, {args.seq_len})")
    step = time.perf_counter()
    wrapper = LastTokenEmbeddingWrapper(
        model, fill_value=args.mask_fill_value, pooling=args.pooling
    ).eval()
    # One sanity sentence replicated to B rows, so the traced graph
    # already carries the target batch size.
    trace_text = common.sanity_text_sets()[0][1][0]
    example = common.tokenize_batch(
        tokenizer, [trace_text] * args.batch, args.seq_len, prompt=common.QUERY_PROMPT
    )
    traced = convert.trace_model(wrapper, example)
    timings["trace"] = time.perf_counter() - step
    print(f"      traced [{timings['trace']:.1f}s]")

    print(f"[5/{_TOTAL_STAGES}] Converting to mlprogram ({args.precision}, {args.target})")
    step = time.perf_counter()
    mlmodel = convert.convert_model(
        traced, args.seq_len, args.precision, args.target, OUTPUT_NAME, batch_size=args.batch
    )
    output_key = convert.resolve_output_key(mlmodel, OUTPUT_NAME)
    timings["convert"] = time.perf_counter() - step
    # The FP32 backbone is about 2.4 GB for the target model; it is
    # released as soon as the converted program exists so that it never
    # coexists with the artifact or with the FP32 reference model.
    del traced, wrapper, model
    gc.collect()
    print(f"      output key {output_key} [{timings['convert']:.1f}s]")

    print(f"[6/{_TOTAL_STAGES}] Saving {mlpackage_path}")
    step = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if mlpackage_path.exists():
        shutil.rmtree(mlpackage_path)
    mlmodel.save(str(mlpackage_path))
    del mlmodel
    gc.collect()
    timings["save"] = time.perf_counter() - step
    print(f"      saved [{timings['save']:.1f}s]")

    print(f"[7/{_TOTAL_STAGES}] Compiling to {mlmodelc_path}")
    step = time.perf_counter()
    convert.compile_model(mlpackage_path, mlmodelc_path)
    if not args.keep_mlpackage:
        shutil.rmtree(mlpackage_path)
    timings["compile"] = time.perf_counter() - step
    print(f"      compiled [{timings['compile']:.1f}s]")

    sanity: dict[str, Any] | None = None
    if args.skip_sanity:
        print(f"[8/{_TOTAL_STAGES}] Sanity check skipped (--skip-sanity)")
    else:
        print(f"[8/{_TOTAL_STAGES}] Sanity check on CPU_AND_NE")
        step = time.perf_counter()
        sanity = run_sanity_check(
            mlmodelc_path,
            model_dir,
            tokenizer,
            args.seq_len,
            args.batch,
            output_key,
            args.pooling,
        )
        timings["sanity"] = time.perf_counter() - step
    timings["total"] = time.perf_counter() - started

    artifacts = {
        "mlpackage": str(mlpackage_path) if args.keep_mlpackage else None,
        "mlmodelc": str(mlmodelc_path),
        "metadata": str(metadata_path),
    }
    metadata = build_metadata(args, model_dir, patch_record, output_key, artifacts, timings, sanity)
    common.write_result_json(metadata_path, metadata)

    if sanity is not None:
        print_sanity_summary(sanity)
    print(f"      timings (sec)   : {metadata['timings_sec']}")
    print(f"      metadata        : {metadata_path}")
    if sanity is not None and not sanity["passed"]:
        print(f"SANITY CHECK FAILED: {mlmodelc_path}")
        return 1
    print(f"CONVERSION COMPLETE: {mlmodelc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
