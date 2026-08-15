"""Tests for the compile pipeline (v0.6 T4, 開発資料/v0.6実装計画.md §4.1/§4.3/§4.4).

Covers ``eeane.compiler.pipeline`` (the driver) together with
``eeane.compiler.artifacts`` (the layout/naming/record decisions it
drives), in two layers:

* Unit tests for the pure decisions (cache naming, output-root
  resolution, bucket defaults, variant naming, config snippet,
  idempotent-skip) -- these run anywhere, including CI.
* One end-to-end run over a *synthetic* randomly initialised ModernBERT
  (trace -> convert -> ``xcrun coremlcompiler`` -> metadata -> snippet).
  It is skipped unless the local development model directories are present
  (the established local-only marker) and ``xcrun`` is available, so CI
  stays green and fast. Converting the real 310M models is v0.6 T7's job.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from transformers.models.modernbert import modeling_modernbert

from eeane import __version__, cli
from eeane.compiler import artifacts, pipeline
from eeane.compiler.backends import modernbert as mb
from eeane.config import ModelEntry

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The end-to-end test needs the Core ML compiler and is a local-only test
# by policy; the presence of the development model directory is the marker
# the other real-artifact modules use to detect a local machine.
_LOCAL_MACHINE = (_REPO_ROOT / "models" / "ruri-v3-310m" / "config.json").exists()
_E2E_AVAILABLE = _LOCAL_MACHINE and shutil.which("xcrun") is not None

# Sequence length of the end-to-end run: small enough to keep the whole
# conversion within a few seconds.
E2E_SEQ_LEN = 32
E2E_STEM = f"s{E2E_SEQ_LEN}_b1_eager_macos13"


@pytest.fixture(autouse=True, scope="module")
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the global ModernBert monkeypatches after this module's tests."""
    original_rotate_half = modeling_modernbert.rotate_half
    original_forward = modeling_modernbert.ModernBertAttention.forward
    yield
    modeling_modernbert.rotate_half = original_rotate_half
    modeling_modernbert.ModernBertAttention.forward = original_forward


def _versions() -> dict[str, str]:
    """Build a version block shaped like the real one, without importing torch info."""
    return {key: f"{key}-1.0" for key in artifacts.SKIP_VERSION_KEYS}


def _write_variant(
    directory: Path,
    versions: dict[str, str],
    *,
    selfcheck_status: str = "skipped",
) -> tuple[Path, Path]:
    """Create a fake compiled variant (``.mlmodelc`` + metadata) on disk.

    Args:
        directory: Directory to create the variant in.
        versions: Version block to record in the metadata.
        selfcheck_status: Value recorded under ``selfcheck.status``.

    Returns:
        Tuple of the ``.mlmodelc`` and metadata paths.
    """
    mlmodelc_path = directory / f"{E2E_STEM}.mlmodelc"
    mlmodelc_path.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"{E2E_STEM}.json"
    metadata_path.write_text(
        json.dumps({"versions": versions, "selfcheck": {"status": selfcheck_status}}),
        encoding="utf-8",
    )
    return mlmodelc_path, metadata_path


# --- cache naming ------------------------------------------------------------


def test_model_cache_name_normalizes_a_hub_id() -> None:
    """A Hub id must become the HF-style ``org--name`` directory name."""
    name = artifacts.model_cache_name("cl-nagoya/ruri-v3-310m", Path("/hf/cache/snapshots/abc"))

    assert name == "cl-nagoya--ruri-v3-310m"


def test_model_cache_name_uses_the_directory_name_for_a_local_source(tmp_path: Path) -> None:
    """A local source must be named after its resolved directory name."""
    model_dir = tmp_path / "ruri-v3-310m"
    model_dir.mkdir()

    assert artifacts.model_cache_name(f"{model_dir}/", model_dir) == "ruri-v3-310m"


def test_model_identifier_keeps_the_hub_id_but_uses_the_name_locally(tmp_path: Path) -> None:
    """The snippet id is the Hub id for a download and the directory name locally."""
    model_dir = tmp_path / "ruri-v3-310m"
    model_dir.mkdir()

    assert artifacts.model_identifier("cl-nagoya/ruri-v3-310m", Path("/hf/x")) == (
        "cl-nagoya/ruri-v3-310m"
    )
    assert artifacts.model_identifier(str(model_dir), model_dir) == "ruri-v3-310m"


def test_model_cache_name_rejects_a_nameless_directory() -> None:
    """A filesystem root has no usable cache name and must be rejected."""
    with pytest.raises(artifacts.CompileError, match="name"):
        artifacts.model_cache_name("/", Path("/"))


# --- output root resolution --------------------------------------------------


def test_resolve_out_root_defaults_to_home_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without XDG_CACHE_HOME the default must be ``~/.cache/eeane``."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert artifacts.resolve_out_root(None, env={}) == (tmp_path / ".cache" / "eeane").resolve()


def test_resolve_out_root_respects_xdg_cache_home(tmp_path: Path) -> None:
    """XDG_CACHE_HOME must move the cache root."""
    root = artifacts.resolve_out_root(None, env={"XDG_CACHE_HOME": str(tmp_path)})

    assert root == (tmp_path / "eeane").resolve()


def test_resolve_out_root_ignores_a_relative_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relative XDG_CACHE_HOME is invalid per spec and must fall back to the home cache."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    root = artifacts.resolve_out_root(None, env={"XDG_CACHE_HOME": "relative/cache"})

    assert root == (tmp_path / ".cache" / "eeane").resolve()


def test_resolve_out_root_expands_an_explicit_out_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit --out-dir must be expanded and absolutized."""
    monkeypatch.setenv("HOME", str(tmp_path))

    assert artifacts.resolve_out_root(Path("~/artifacts")) == (tmp_path / "artifacts").resolve()


# --- bucket defaults ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("embedding", [128, 512, 1024]), ("reranker", [512, 1024])],
)
def test_resolve_buckets_defaults_per_kind(kind: str, expected: list[int]) -> None:
    """Omitting --buckets must reproduce the deployed v0.5 bucket configuration."""
    assert artifacts.resolve_buckets(None, kind) == expected


def test_resolve_buckets_sorts_and_deduplicates() -> None:
    """Explicit buckets must be compiled once each, in ascending order."""
    assert artifacts.resolve_buckets([512, 128, 512], "embedding") == [128, 512]


@pytest.mark.parametrize("buckets", [[], [0], [-1], [128, 0]])
def test_resolve_buckets_rejects_empty_or_non_positive(buckets: list[int]) -> None:
    """An empty or non-positive bucket list must be a clean compile error."""
    with pytest.raises(artifacts.CompileError, match="bucket"):
        artifacts.resolve_buckets(buckets, "embedding")


def test_resolve_buckets_rejects_an_unknown_kind() -> None:
    """A kind with no default bucket set must be reported, not silently emptied."""
    with pytest.raises(artifacts.CompileError, match="kind"):
        artifacts.resolve_buckets(None, "classifier")


# --- variant naming ----------------------------------------------------------


def test_variant_stem_matches_the_poc_naming() -> None:
    """The default variant name must stay byte-identical to the PoC artifacts."""
    assert artifacts.variant_stem(512, 1, "eager", "macos13", "fp16") == "s512_b1_eager_macos13"


def test_variant_stem_appends_fp32_only_for_fp32() -> None:
    """fp32 must get its own suffix so it cannot overwrite the fp16 baseline."""
    assert artifacts.variant_stem(128, 4, "sdpa", "macos15", "fp32") == "s128_b4_sdpa_macos15_fp32"


# --- config snippet ----------------------------------------------------------


def _parse_snippet(snippet: str) -> dict[str, Any]:
    """Parse a generated snippet and return its single ``[[models]]`` entry."""
    parsed = tomllib.loads(snippet)
    assert len(parsed["models"]) == 1
    return parsed["models"][0]


def test_config_snippet_parses_as_toml_and_lists_every_bucket(tmp_path: Path) -> None:
    """The snippet must be valid TOML naming every compiled bucket, with absolute paths."""
    compiled_artifacts = {512: tmp_path / "s512.mlmodelc", 128: tmp_path / "s128.mlmodelc"}

    snippet = artifacts.build_config_snippet(
        model_id="cl-nagoya/ruri-v3-310m",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts=compiled_artifacts,
    )

    entry = _parse_snippet(snippet)
    assert entry["id"] == "cl-nagoya/ruri-v3-310m"
    assert entry["kind"] == "embedding"
    assert entry["normalize"] is True
    assert Path(entry["tokenizer"]).is_absolute()
    assert sorted(entry["artifacts"]) == ["128", "512"]
    assert all(Path(value).is_absolute() for value in entry["artifacts"].values())


def test_config_snippet_is_accepted_by_the_config_schema(tmp_path: Path) -> None:
    """The snippet must validate against the runtime's own ModelEntry schema."""
    snippet = artifacts.build_config_snippet(
        model_id="ruri-v3-310m",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={128: tmp_path / "s128.mlmodelc"},
    )

    entry = ModelEntry(**_parse_snippet(snippet))

    assert entry.buckets == (128,)
    assert entry.output_name == "embedding"


def test_config_snippet_omits_normalize_for_a_reranker(tmp_path: Path) -> None:
    """`normalize` is embedding-only; setting it on a reranker is a config error."""
    snippet = artifacts.build_config_snippet(
        model_id="ruri-v3-reranker-310m",
        kind="reranker",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={512: tmp_path / "s512.mlmodelc"},
    )

    entry = _parse_snippet(snippet)
    assert "normalize" not in entry
    assert ModelEntry(**entry).kind == "reranker"


def test_config_snippet_absolutizes_relative_paths() -> None:
    """Relative paths must never reach the snippet: configs are read from elsewhere."""
    snippet = artifacts.build_config_snippet(
        model_id="local",
        kind="embedding",
        tokenizer_path=Path("cache/tokenizer.json"),
        artifacts={128: Path("cache/s128.mlmodelc")},
    )

    entry = _parse_snippet(snippet)
    assert Path(entry["tokenizer"]).is_absolute()
    assert Path(entry["artifacts"]["128"]).is_absolute()


def test_config_snippet_escapes_special_characters_in_paths(tmp_path: Path) -> None:
    """A quote or backslash in a path must survive the TOML round trip."""
    weird = tmp_path / "we'i\"rd\\dir" / "tokenizer.json"

    snippet = artifacts.build_config_snippet(
        model_id='odd "id"',
        kind="embedding",
        tokenizer_path=weird,
        artifacts={128: tmp_path / "a b" / "s128.mlmodelc"},
    )

    entry = _parse_snippet(snippet)
    assert entry["tokenizer"] == str(weird.resolve())
    assert entry["artifacts"]["128"] == str((tmp_path / "a b" / "s128.mlmodelc").resolve())
    assert entry["id"] == 'odd "id"'


def test_write_config_snippet_creates_parent_directories(tmp_path: Path) -> None:
    """--emit-config must work even when the destination directory does not exist."""
    destination = tmp_path / "nested" / "eeane.toml"

    artifacts.write_config_snippet(destination, "[[models]]\n")

    assert destination.read_text(encoding="utf-8") == "[[models]]\n"


def test_write_config_snippet_reports_an_unwritable_destination(tmp_path: Path) -> None:
    """A destination that cannot be written must fail with a clean compile error."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(artifacts.CompileError, match="eeane.toml"):
        artifacts.write_config_snippet(blocker / "eeane.toml", "[[models]]\n")


# --- idempotent skip ---------------------------------------------------------


def test_needs_conversion_skips_an_up_to_date_variant(tmp_path: Path) -> None:
    """Matching versions plus both artifacts present means nothing to do."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is False


def test_needs_conversion_when_a_recorded_version_differs(tmp_path: Path) -> None:
    """A different torch/transformers/... version must trigger a reconversion."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)

    current = {**versions, "coremltools": "9.1"}

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, current) is True


def test_needs_conversion_when_forced(tmp_path: Path) -> None:
    """--force must reconvert even a perfectly up-to-date variant."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions, force=True) is True


def test_needs_conversion_when_the_artifact_is_missing(tmp_path: Path) -> None:
    """Metadata without its .mlmodelc is not a reusable variant."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)
    shutil.rmtree(mlmodelc_path)

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is True


def test_needs_conversion_when_the_metadata_is_missing_or_corrupt(tmp_path: Path) -> None:
    """Missing or unparsable metadata must never be treated as up to date."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)
    metadata_path.unlink()

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is True

    metadata_path.write_text("{not json", encoding="utf-8")

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is True


def test_needs_conversion_when_the_recorded_selfcheck_failed(tmp_path: Path) -> None:
    """A variant whose self-check failed must be retried, not skipped (§4.5)."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions, selfcheck_status="failed")

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is True


# --- tokenizer verification inputs -------------------------------------------


def test_verification_inputs_are_self_contained_for_an_embedding_model() -> None:
    """The gate inputs must include the boundary cases and need no repository data."""
    backend = mb.ModernBertBackend()

    texts, pairs = pipeline.verification_inputs(backend, "embedding", [128])

    assert "" in texts  # empty input
    assert any(len(text) == 1 for text in texts)  # single character
    assert max(len(text) for text in texts) > 4 * 128  # far longer than the bucket
    assert all(isinstance(text, str) for text in texts)
    assert set(backend.sanity_inputs("embedding")).issubset(texts)
    assert pairs == []  # an embedding model never encodes pairs


def test_verification_inputs_include_pairs_for_a_reranker() -> None:
    """A reranker must be verified on pair encodings (the dynamic post_processor)."""
    backend = mb.ModernBertBackend()

    texts, pairs = pipeline.verification_inputs(backend, "reranker", [512])

    assert texts and "" in texts
    assert ("", "") in pairs
    assert all(len(pair) == 2 for pair in pairs)
    assert set(backend.sanity_inputs("reranker")).issubset(pairs)


def test_verification_long_input_scales_with_the_largest_bucket() -> None:
    """The long input must outgrow whatever the largest requested bucket is."""
    backend = mb.ModernBertBackend()

    short_texts, _ = pipeline.verification_inputs(backend, "embedding", [128])
    long_texts, _ = pipeline.verification_inputs(backend, "embedding", [128, 1024])

    assert max(len(text) for text in long_texts) > max(len(text) for text in short_texts)


# --- run(): argument and resolution failures ---------------------------------


def _compile_args(*arguments: str) -> Any:
    """Parse a ``compile`` command line into the namespace ``run`` expects."""
    return cli.build_parser().parse_args(["compile", *arguments])


def test_run_rejects_a_non_positive_batch(capsys: pytest.CaptureFixture) -> None:
    """--batch 0 must fail before anything is loaded or written."""
    exit_code = pipeline.run(_compile_args("some/path", "--batch", "0"))

    assert exit_code == 1
    assert "batch" in capsys.readouterr().err


def test_run_reports_an_unresolvable_source(capsys: pytest.CaptureFixture) -> None:
    """A source that is neither a directory nor a Hub id must exit non-zero, cleanly."""
    exit_code = pipeline.run(_compile_args("definitely/not/a/model/dir"))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "definitely/not/a/model/dir" in captured.err
    assert "Traceback" not in captured.err


def test_run_reports_a_directory_without_config_json(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A directory that is not a HuggingFace model must be reported by name."""
    source = tmp_path / "not-a-model"
    source.mkdir()

    exit_code = pipeline.run(_compile_args(str(source)))

    assert exit_code == 1
    assert "config.json" in capsys.readouterr().err


def test_run_reports_an_unusable_out_dir(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A cache root that cannot be created must fail before the model is loaded."""
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["ModernBertModel"]}), encoding="utf-8"
    )
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    exit_code = pipeline.run(_compile_args(str(source), "--out-dir", str(blocker)))

    assert exit_code == 1
    assert "director" in capsys.readouterr().err


# --- end-to-end on a synthetic ModernBERT (local only) -----------------------


def _build_synthetic_model(path: Path) -> Path:
    """Create a tiny randomly initialised ModernBERT model directory.

    The directory is a complete HuggingFace distribution-format model
    (config.json + safetensors weights + a byte-level fast tokenizer), so
    the pipeline can be driven exactly as it would be for a real model.

    Args:
        path: Directory to create (parents are created as needed).

    Returns:
        ``path``.
    """
    config = modeling_modernbert.ModernBertConfig(
        vocab_size=300,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=64,
        local_attention=8,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    modeling_modernbert.ModernBertModel(config).save_pretrained(path)

    # Byte-level vocabulary with no merges: every byte is its own token, so
    # Japanese text produces plenty of tokens without shipping a vocab file.
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())):
        vocab[character] = index + 4
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 2), ("</s>", 3)],
    )
    tokenizer.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=64,
    ).save_pretrained(path)
    return path


def _mtimes(directory: Path) -> dict[str, int]:
    """Snapshot every file's modification time under ``directory``."""
    return {
        str(path.relative_to(directory)): path.stat().st_mtime_ns
        for path in sorted(directory.rglob("*"))
    }


@pytest.fixture(scope="module")
def synthetic_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the synthetic model directory once for the end-to-end tests."""
    if not _E2E_AVAILABLE:
        pytest.skip("end-to-end conversion needs a local machine with xcrun")
    return _build_synthetic_model(tmp_path_factory.mktemp("synthetic") / "tiny-modernbert")


@pytest.fixture(scope="module")
def compiled(synthetic_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the full pipeline once and return its inputs, outputs and stdout."""
    workspace = tmp_path_factory.mktemp("compile")
    out_dir = workspace / "cache"
    emit_config = workspace / "emitted.toml"
    arguments = [
        str(synthetic_model_dir),
        "--buckets",
        str(E2E_SEQ_LEN),
        "--out-dir",
        str(out_dir),
        "--emit-config",
        str(emit_config),
    ]

    before = _mtimes(synthetic_model_dir)
    stdout = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout):
        exit_code = pipeline.run(_compile_args(*arguments))
    elapsed = time.perf_counter() - started

    return {
        "arguments": arguments,
        "exit_code": exit_code,
        "elapsed": elapsed,
        "stdout": stdout.getvalue(),
        "model_dir": synthetic_model_dir,
        "model_root": out_dir / "compiled" / synthetic_model_dir.name,
        "emit_config": emit_config,
        "mtimes_before": before,
    }


def test_e2e_produces_the_compiled_artifact_and_drops_the_mlpackage(
    compiled: dict[str, Any],
) -> None:
    """A successful run must leave a .mlmodelc, a frozen tokenizer and no .mlpackage."""
    model_root = compiled["model_root"]

    assert compiled["exit_code"] == 0
    assert (model_root / f"{E2E_STEM}.mlmodelc").is_dir()
    assert not (model_root / f"{E2E_STEM}.mlpackage").exists()
    assert (model_root / artifacts.TOKENIZER_FILENAME).is_file()
    assert not list(model_root.glob("*.compile_tmp"))


def test_e2e_writes_variant_metadata(compiled: dict[str, Any]) -> None:
    """The variant metadata must describe the source, versions, patches and timings."""
    metadata = json.loads((compiled["model_root"] / f"{E2E_STEM}.json").read_text(encoding="utf-8"))

    assert metadata["format_version"] == artifacts.METADATA_FORMAT_VERSION
    assert metadata["source"]["requested"] == str(compiled["model_dir"])
    assert Path(metadata["source"]["resolved"]) == compiled["model_dir"].resolve()
    assert metadata["args"]["buckets"] == [E2E_SEQ_LEN]
    assert metadata["versions"]["eeane"] == __version__
    assert metadata["patches"]["rotate_half_static"] is True
    assert metadata["patches"]["eager_attention_rank4"] is True
    assert {"load", "trace", "convert", "compile", "total"} <= set(metadata["timings_sec"])
    assert Path(metadata["artifacts"]["mlmodelc"]).is_dir()
    assert "mlpackage" not in metadata["artifacts"]
    assert metadata["selfcheck"]["status"] == artifacts.SELFCHECK_STATUS_SKIPPED


def test_e2e_writes_model_info(compiled: dict[str, Any]) -> None:
    """model_info.json must summarise the model for later cache resolution."""
    info = json.loads(
        (compiled["model_root"] / artifacts.MODEL_INFO_FILENAME).read_text(encoding="utf-8")
    )

    assert info["format_version"] == artifacts.MODEL_INFO_FORMAT_VERSION
    assert info["id"] == compiled["model_dir"].name
    assert info["kind"] == "embedding"
    assert info["output_name"] == "embedding"
    assert info["buckets"] == [E2E_SEQ_LEN]
    assert info["tokenizer"] == artifacts.TOKENIZER_FILENAME
    assert info["eeane_version"] == __version__
    assert info["tokenizer_freeze"]["verified"] is True
    assert info["tokenizer_freeze"]["buckets"] == [E2E_SEQ_LEN]


def test_e2e_prints_and_emits_a_usable_config_snippet(compiled: dict[str, Any]) -> None:
    """stdout must carry the snippet, and --emit-config must write the same text."""
    stdout = compiled["stdout"]
    emitted = compiled["emit_config"].read_text(encoding="utf-8")

    assert "[[models]]" in stdout
    assert emitted in stdout

    entry = ModelEntry(**_parse_snippet(emitted))
    assert entry.id == compiled["model_dir"].name
    assert entry.tokenizer == compiled["model_root"] / artifacts.TOKENIZER_FILENAME
    assert entry.artifacts[E2E_SEQ_LEN] == compiled["model_root"] / f"{E2E_STEM}.mlmodelc"
    assert entry.tokenizer.is_file()
    assert entry.artifacts[E2E_SEQ_LEN].is_dir()


def test_e2e_leaves_the_input_model_directory_untouched(compiled: dict[str, Any]) -> None:
    """The input model directory is read-only (v0.6実装計画.md §2-11)."""
    assert _mtimes(compiled["model_dir"]) == compiled["mtimes_before"]


def test_e2e_second_run_skips_the_up_to_date_variant(
    compiled: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    """Re-running without --force must reuse the artifact and say so."""
    mlmodelc_path = compiled["model_root"] / f"{E2E_STEM}.mlmodelc"
    before = mlmodelc_path.stat().st_mtime_ns

    exit_code = pipeline.run(_compile_args(*compiled["arguments"]))

    assert exit_code == 0
    assert mlmodelc_path.stat().st_mtime_ns == before
    assert "skip" in capsys.readouterr().err.lower()


def test_e2e_force_reconverts_and_keeps_the_mlpackage(compiled: dict[str, Any]) -> None:
    """--force must rebuild the artifact; --keep-mlpackage must retain the intermediate."""
    model_root = compiled["model_root"]
    mlmodelc_path = model_root / f"{E2E_STEM}.mlmodelc"
    before = mlmodelc_path.stat().st_mtime_ns

    with contextlib.redirect_stdout(io.StringIO()):
        exit_code = pipeline.run(
            _compile_args(*compiled["arguments"], "--force", "--keep-mlpackage")
        )

    assert exit_code == 0
    assert mlmodelc_path.stat().st_mtime_ns != before
    assert (model_root / f"{E2E_STEM}.mlpackage").is_dir()
    metadata = json.loads((model_root / f"{E2E_STEM}.json").read_text(encoding="utf-8"))
    assert Path(metadata["artifacts"]["mlpackage"]).is_dir()


def test_e2e_selfcheck_hook_result_is_recorded(compiled: dict[str, Any]) -> None:
    """A self-check implementation's report must land in the variant metadata."""
    contexts: list[pipeline.SelfcheckContext] = []

    def fake_selfcheck(context: pipeline.SelfcheckContext) -> dict[str, Any]:
        contexts.append(context)
        return {"status": "passed", "note": "fake"}

    with contextlib.redirect_stdout(io.StringIO()):
        exit_code = pipeline.run(
            _compile_args(*compiled["arguments"], "--force"), selfcheck_fn=fake_selfcheck
        )

    assert exit_code == 0
    assert len(contexts) == 1
    assert contexts[0].seq_len == E2E_SEQ_LEN
    assert contexts[0].kind == "embedding"
    assert contexts[0].output_name == "embedding"
    assert contexts[0].mlmodelc_path.is_dir()
    assert contexts[0].tokenizer_path.is_file()
    metadata = json.loads((compiled["model_root"] / f"{E2E_STEM}.json").read_text(encoding="utf-8"))
    assert metadata["selfcheck"] == {"status": "passed", "note": "fake"}


def test_e2e_failing_selfcheck_fails_the_compile(compiled: dict[str, Any]) -> None:
    """A failed self-check must exit non-zero and leave the variant non-skippable."""

    def failing_selfcheck(context: pipeline.SelfcheckContext) -> dict[str, Any]:
        return {"status": artifacts.SELFCHECK_STATUS_FAILED, "reason": "fake failure"}

    with contextlib.redirect_stdout(io.StringIO()):
        exit_code = pipeline.run(
            _compile_args(*compiled["arguments"], "--force"), selfcheck_fn=failing_selfcheck
        )

    assert exit_code == 1
    model_root = compiled["model_root"]
    metadata_path = model_root / f"{E2E_STEM}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["selfcheck"]["status"] == artifacts.SELFCHECK_STATUS_FAILED
    # The recorded failure must not be reusable on the next run.
    assert artifacts.needs_conversion(
        model_root / f"{E2E_STEM}.mlmodelc", metadata_path, metadata["versions"]
    )
