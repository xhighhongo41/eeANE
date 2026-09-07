"""Architecture-independent helpers shared by the compile backends.

Everything in here is plumbing that does not depend on a particular model
architecture: masked pooling, the stable sigmoid used for reranker
post-processing, fixed-shape tokenization, the FP32 PyTorch baselines,
the traceable wrapper modules, the readers for what a sentence-transformers
model directory declares (its pooling mode and the module chain that may
project the pooled vector further), and the self-check's per-language
sanity fixtures. Architecture-specific code (graph patches,
position-embedding offsets, and the fixtures a family overrides) stays in
the per-family backend modules that import from here.

The pooling helpers and the wrappers are the single source of truth for
both sides of the self-check: the module that is traced into the Core ML
graph and the PyTorch baseline it is compared against must compute the
same function, or the comparison is meaningless.

Importing this module pulls in ``torch``/``transformers``; it therefore
requires the ``[compile]`` extra and must never be imported from the
``eeane serve`` code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from eeane.runtime import PairTemplate

# Pooling modes the shared embedding helpers implement. A backend records
# the mode it detected on its LoadedModel handle, and both the wrapper
# selection and the FP32 baseline are driven by that value. Reading a mode
# and being able to compile it are two different things: this module knows
# every mode listed here, while a backend only serves the ones it offers a
# wrapper for (a causal stack cannot be pooled like an encoder, and vice
# versa), and refuses the others.
POOLING_MEAN = "mean"
POOLING_CLS = "cls"
POOLING_LASTTOKEN = "lasttoken"
POOLING_MODES: tuple[str, ...] = (POOLING_MEAN, POOLING_CLS, POOLING_LASTTOKEN)

# sentence-transformers pooling module: directory holding the pooling
# declaration of an embedding model, the file inside it, and the flags it
# can set. The declaration is not part of the HF configuration and does
# not depend on the architecture, so every backend whose embedding models
# come from sentence-transformers reads the same file the same way.
POOLING_DIRNAME = "1_Pooling"
POOLING_CONFIG_FILENAME = "config.json"
POOLING_MODE_PREFIX = "pooling_mode_"
POOLING_MODE_KEYS: dict[str, str] = {
    "pooling_mode_mean_tokens": POOLING_MEAN,
    "pooling_mode_cls_token": POOLING_CLS,
    "pooling_mode_lasttoken": POOLING_LASTTOKEN,
}

# Appended to every pooling-detection error: an embedding model whose
# pooling cannot be read must fail loudly rather than default silently,
# because the wrong pooling produces a plausible but wrong embedding.
_POOLING_REQUIREMENT = (
    "An embedding model must declare its pooling in the sentence-transformers "
    f"'{POOLING_DIRNAME}/{POOLING_CONFIG_FILENAME}' with exactly one of "
    f"{' / '.join(POOLING_MODE_KEYS)} set to true."
)

# sentence-transformers module declaration: the file listing the modules a
# model applies, in order, and the module types this backend knows. The
# list is not part of the HF configuration, so every backend whose
# embedding models come from sentence-transformers reads the same file the
# same way.
ST_MODULES_FILENAME = "modules.json"
ST_MODULE_TRANSFORMER = "sentence_transformers.models.Transformer"
ST_MODULE_POOLING = "sentence_transformers.models.Pooling"
ST_MODULE_DENSE = "sentence_transformers.models.Dense"
ST_MODULE_NORMALIZE = "sentence_transformers.models.Normalize"

# Files of one Dense module, and the keys its checkpoint stores the linear
# layer under: sentence-transformers holds that layer as ``self.linear``,
# so its state dict is prefixed accordingly. The pickle-based file is only
# ever reached for a source that was resolved with pickle weights allowed;
# resolution refuses a module directory without safetensors otherwise.
DENSE_CONFIG_FILENAME = "config.json"
DENSE_WEIGHTS_FILENAME = "model.safetensors"
DENSE_PICKLE_WEIGHTS_FILENAME = "pytorch_model.bin"
DENSE_WEIGHT_KEY = "linear.weight"
DENSE_BIAS_KEY = "linear.bias"

# Declaration fields a Dense module's own config.json must carry.
DENSE_IN_FEATURES_KEY = "in_features"
DENSE_OUT_FEATURES_KEY = "out_features"
DENSE_BIAS_FLAG_KEY = "bias"
DENSE_ACTIVATION_KEY = "activation_function"

# Activations a Dense module may declare, keyed by the class path it names
# them with, mapped to the short name a compiled variant records. Only
# these two are implemented: any other activation would change the
# embedding, so it is refused rather than approximated.
DENSE_ACTIVATION_IDENTITY = "identity"
DENSE_ACTIVATION_TANH = "tanh"
DENSE_ACTIVATIONS: dict[str, str] = {
    "torch.nn.modules.linear.Identity": DENSE_ACTIVATION_IDENTITY,
    "torch.nn.modules.activation.Tanh": DENSE_ACTIVATION_TANH,
}

# Appended to every module-chain error. A model whose chain cannot be
# reproduced must be refused before any weight is read: converting it
# anyway would produce a graph that silently leaves out a transformation
# the model's published embeddings depend on.
_MODULE_CHAIN_REQUIREMENT = (
    "This backend reproduces a sentence-transformers chain of a "
    f"'{ST_MODULE_TRANSFORMER}', one '{ST_MODULE_POOLING}', any number of "
    f"'{ST_MODULE_DENSE}' projections and an optional trailing "
    f"'{ST_MODULE_NORMALIZE}', in that order."
)

# Module-type suffixes of the two-module chain a generative reranker
# declares. They are matched by suffix, not compared in full, because the
# same two roles are published under more than one namespace (a
# ``...models.Transformer`` and a ``...base.modules.transformer.Transformer``
# are the same module); the chain check above keeps its exact match, since
# the modules it accepts carry weights this package reads by path.
ST_TRANSFORMER_SUFFIX = ".Transformer"
ST_LOGIT_SCORE_SUFFIX = ".LogitScore"

# sentence-transformers scoring module of a generative reranker: the
# directory holding the declaration, the file inside it, and the two keys
# naming the vocabulary entries whose logits carry the verdict.
LOGIT_SCORE_DIRNAME = "1_LogitScore"
LOGIT_SCORE_CONFIG_FILENAME = "config.json"
LOGIT_SCORE_TRUE_KEY = "true_token_id"
LOGIT_SCORE_FALSE_KEY = "false_token_id"

# Appended to every scoring-declaration error. A generative reranker reads
# two rows of its output projection and subtracts them; picking those rows
# by guesswork would produce a graph that scores something else entirely
# while still looking like a working model, so a declaration that cannot
# be read is refused rather than completed.
_LOGIT_SCORE_REQUIREMENT = (
    "A reranker scored from two vocabulary logits must declare them in the "
    f"sentence-transformers '{LOGIT_SCORE_DIRNAME}/{LOGIT_SCORE_CONFIG_FILENAME}' "
    f"as two different non-negative integers '{LOGIT_SCORE_TRUE_KEY}' and "
    f"'{LOGIT_SCORE_FALSE_KEY}'."
)

# Appended to every scoring module-chain error.
_SCORING_CHAIN_REQUIREMENT = (
    "This backend reproduces a sentence-transformers chain of exactly one module whose type "
    f"ends in '{ST_TRANSFORMER_SUFFIX}', followed by exactly one whose type ends in "
    f"'{ST_LOGIT_SCORE_SUFFIX}', in that order."
)

# Markers a pair template's body format writes the request's own texts
# at. They match the ones :mod:`eeane.runtime` declares, since both sides
# must render the very same body; they are substituted by plain string
# surgery, never through ``str.format``.
_QUERY_MARKER = "{query}"
_DOCUMENT_MARKER = "{document}"

# sentence-transformers model-level declaration: the file a model states
# its named prompts in, the key naming the one it applies by default, the
# table those names address, and the name to fall back on when no default
# is named. The prompt is part of the question the model was trained to
# answer, so it is read from the model rather than supplied here.
ST_CONFIG_FILENAME = "config_sentence_transformers.json"
ST_DEFAULT_PROMPT_NAME_KEY = "default_prompt_name"
ST_PROMPTS_KEY = "prompts"
ST_FALLBACK_PROMPT_NAME = "query"


# --- sanity fixtures, one set per language -----------------------------------
#
# The self-check evaluates every set and accepts a variant as soon as one
# of them clears the threshold, so these sets are what decides which
# checkpoints can be compiled at all: fixtures in a language a model has
# no vocabulary for encode to little more than unknown-token rows, whose
# FP16-vs-FP32 difference says nothing about the model yet can still miss
# the threshold. Offering English, Japanese and Chinese means a model
# covering any one of them is measured on inputs it can actually read.
#
# Every set is built the same way, so the sets stay comparable:
#
# * an embedding set holds three sentences -- short, medium and long -- so
#   one fixed sequence length exercises three different amounts of padding;
# * a reranker set holds three pairs -- relevant, irrelevant, partially
#   related -- of which the first two share their query, so only the
#   document decides which of them must score higher.
#
# Existing fixtures are never reworded: the accuracy numbers recorded for
# already-verified models were measured on these exact strings.

SANITY_LANGUAGE_EN = "en"
SANITY_LANGUAGE_JA = "ja"
SANITY_LANGUAGE_ZH = "zh"

# Position of the relevant and the irrelevant pair inside every reranker
# set, as handed to a SanitySpec: the sets share one pair ordering, so
# they share these indices too.
SANITY_RELEVANT_INDEX = 0
SANITY_IRRELEVANT_INDEX = 1

SANITY_TEXTS_EN: tuple[str, ...] = (
    "Question: how tall is the highest mountain in Japan?",
    "Document: Mount Fuji rises 3,776 metres above sea level on the border between "
    "Shizuoka and Yamanashi, and is the highest mountain in Japan.",
    "Topic: turning a large collection of documents into vectors ahead of time makes it "
    "possible to retrieve passages with a similar meaning without reading every text again.",
)

SANITY_TEXTS_JA: tuple[str, ...] = (
    "質問: 富士山の標高は何メートルですか。",
    "文書: 富士山は静岡県と山梨県にまたがる標高3776メートルの山であり、"
    "日本の最高峰として知られている。",
    "話題: 大量の文書をあらかじめベクトルに変換して保存しておくと、"
    "検索のたびに本文を読み直さずに近い意味の文書を取り出せる。",
)

SANITY_TEXTS_ZH: tuple[str, ...] = (
    "问题：长江全长大约有多少公里？",
    "文档：长江全长约6300公里，发源于青藏高原，自西向东流经中国多个省份，最终注入东海。",
    "主题：将大量文档预先转换为向量并建立索引，可以在检索时快速找到语义相近的内容，"
    "而无需逐篇重新阅读原文。",
)

SANITY_PAIRS_EN: tuple[tuple[str, str], ...] = (
    # Relevant pair
    (
        "How tall is the highest mountain in Japan?",
        "Mount Fuji rises 3,776 metres above sea level on the border between Shizuoka "
        "and Yamanashi, and is the highest mountain in Japan.",
    ),
    # Irrelevant pair
    (
        "How tall is the highest mountain in Japan?",
        "Brewing coffee with freshly ground beans is said to bring out a richer aroma "
        "than using pre-ground coffee.",
    ),
    # Partially related pair
    (
        "How does vector search work?",
        "Public libraries usually arrange the books on their shelves in alphabetical "
        "order by the author's surname.",
    ),
)

SANITY_PAIRS_JA: tuple[tuple[str, str], ...] = (
    # Relevant pair
    (
        "富士山の標高は何メートルですか。",
        "富士山は静岡県と山梨県にまたがる標高3776メートルの山であり、日本の最高峰として知られている。",
    ),
    # Irrelevant pair
    (
        "富士山の標高は何メートルですか。",
        "味噌汁の出汁は昆布と鰹節を組み合わせると香りが良くなると言われている。",
    ),
    # Partially related pair
    (
        "ベクトル検索の仕組みを知りたい。",
        "図書館では蔵書を著者名の五十音順に並べて管理している。",
    ),
)

SANITY_PAIRS_ZH: tuple[tuple[str, str], ...] = (
    # Relevant pair
    (
        "长江全长大约有多少公里？",
        "长江全长约6300公里，发源于青藏高原，自西向东流经中国多个省份，最终注入东海。",
    ),
    # Irrelevant pair
    (
        "长江全长大约有多少公里？",
        "泡茶时水温对茶叶的香气和口感有明显影响，绿茶一般适合用八十度左右的热水冲泡。",
    ),
    # Partially related pair
    (
        "向量检索是如何工作的？",
        "图书馆通常按照作者姓氏的拼音顺序整理书架上的藏书。",
    ),
)

# The sets as a backend hands them to a SanitySpec. The order is the
# evaluation order of the self-check and the tie-break between two equally
# good sets, so it is fixed here rather than derived from a mapping.
SANITY_TEXT_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (SANITY_LANGUAGE_EN, SANITY_TEXTS_EN),
    (SANITY_LANGUAGE_JA, SANITY_TEXTS_JA),
    (SANITY_LANGUAGE_ZH, SANITY_TEXTS_ZH),
)

SANITY_PAIR_SETS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (SANITY_LANGUAGE_EN, SANITY_PAIRS_EN),
    (SANITY_LANGUAGE_JA, SANITY_PAIRS_JA),
    (SANITY_LANGUAGE_ZH, SANITY_PAIRS_ZH),
)


def override_sanity_set(
    input_sets: tuple[tuple[str, tuple[Any, ...]], ...],
    language: str,
    inputs: tuple[Any, ...],
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Replace one language's fixtures, keeping every other set and the order.

    A backend whose already-verified models were measured on fixtures of
    its own keeps those for that language -- rewording them would move
    the recorded numbers -- while still offering the shared sets for the
    languages it has nothing special to say about.

    Args:
        input_sets: Sets to start from, typically :data:`SANITY_TEXT_SETS`
            or :data:`SANITY_PAIR_SETS`.
        language: Language whose inputs are replaced.
        inputs: Replacement inputs for that language.

    Returns:
        A new tuple of sets, in the order of ``input_sets``.

    Raises:
        ValueError: If ``language`` is not among ``input_sets``; silently
            returning the shared fixtures would hide the typo until a
            model was measured against the wrong ones.
    """
    if language not in {declared for declared, _ in input_sets}:
        declared = ", ".join(declared for declared, _ in input_sets)
        raise ValueError(f"cannot override the '{language}' sanity set (declared: {declared})")
    return tuple(
        (declared, inputs if declared == language else declared_inputs)
        for declared, declared_inputs in input_sets
    )


def read_pooling_mode(model_dir: Path) -> str:
    """Read the pooling mode an embedding model directory declares.

    Args:
        model_dir: Local HuggingFace-format model directory, expected to
            carry a sentence-transformers pooling module.

    Returns:
        One of :data:`POOLING_MODES`. Whether the backend compiling the
        model implements that mode is decided by the backend, not here.

    Raises:
        ValueError: If the pooling declaration is missing, unreadable,
            malformed, or does not select exactly one supported mode.
    """
    path = model_dir / POOLING_DIRNAME / POOLING_CONFIG_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot read the pooling module '{path}': {exc}. {_POOLING_REQUIREMENT}"
        ) from exc
    try:
        declaration = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"'{path}' is not valid JSON: {exc}. {_POOLING_REQUIREMENT}") from exc
    if not isinstance(declaration, dict):
        raise ValueError(f"'{path}' does not contain a JSON object. {_POOLING_REQUIREMENT}")
    # Only a literal ``true`` counts: anything else (a string, a number)
    # is a declaration this backend cannot claim to understand.
    enabled = [
        key
        for key, value in declaration.items()
        if key.startswith(POOLING_MODE_PREFIX) and value is True
    ]
    if len(enabled) != 1 or enabled[0] not in POOLING_MODE_KEYS:
        declared = ", ".join(sorted(enabled)) if enabled else "none"
        raise ValueError(
            f"'{path}' does not enable exactly one supported pooling mode "
            f"(enabled: {declared}). {_POOLING_REQUIREMENT}"
        )
    return POOLING_MODE_KEYS[enabled[0]]


@dataclass(frozen=True)
class DenseStage:
    """One Dense projection a model directory declares after its pooling.

    Attributes:
        path: Directory the module's declaration and weights live in.
        in_features: Declared input width; must match the width the
            previous stage (or the pooling) produces.
        out_features: Declared output width.
        bias: Whether the linear layer has a bias.
        activation: Short name of the declared activation, one of
            :data:`DENSE_ACTIVATION_IDENTITY` / :data:`DENSE_ACTIVATION_TANH`.
    """

    path: Path
    in_features: int
    out_features: int
    bias: bool
    activation: str

    def as_record(self) -> dict[str, Any]:
        """Describe the stage for a compiled variant's metadata.

        Returns:
            A JSON-serializable dict. It holds the declaration rather than
            the weights, which is exactly what a later run compares its
            own declaration against to decide whether the artifact still
            matches the model.
        """
        return {
            "in": self.in_features,
            "out": self.out_features,
            "bias": self.bias,
            "activation": self.activation,
        }


def read_dense_modules(model_dir: Path) -> tuple[DenseStage, ...]:
    """Read the Dense projections a sentence-transformers directory declares.

    No weight is read here: the declaration alone decides whether a model
    can be compiled at all, and finding that out first avoids loading
    gigabytes of parameters for a model that would then be refused.

    A trailing ``Normalize`` module is read past on purpose. The compiled
    graph always returns the unnormalized embedding and the server applies
    L2 normalization according to its own configuration, so baking it into
    the graph would take that choice away; the self-check is unaffected,
    since it compares cosine similarities, which normalization leaves
    unchanged.

    Args:
        model_dir: Local HuggingFace-format model directory.

    Returns:
        One :class:`DenseStage` per declared projection, in the order the
        model applies them. Empty when the directory declares no module
        chain at all (which is what most published models look like) or
        when the chain holds no Dense module.

    Raises:
        ValueError: If the declaration is unreadable or malformed, if the
            chain is not one this backend can reproduce, or if a declared
            Dense module cannot be described exactly.
    """
    modules_path = model_dir / ST_MODULES_FILENAME
    if not modules_path.is_file():
        # No declaration at all: the Transformer-plus-Pooling model this
        # backend has always assumed, with nothing to project afterwards.
        return ()
    entries = _read_module_entries(modules_path)
    _check_module_chain(modules_path, [entry[0] for entry in entries])
    stages = tuple(
        _read_dense_stage(model_dir, modules_path, declared_path)
        for module_type, declared_path in entries
        if module_type == ST_MODULE_DENSE
    )
    _check_dense_widths(stages)
    return stages


def dense_record(stages: Sequence[DenseStage]) -> tuple[dict[str, Any], ...] | None:
    """Describe declared Dense stages for a compiled variant's metadata.

    Args:
        stages: Stages returned by :func:`read_dense_modules`.

    Returns:
        One record per stage, or ``None`` when there is no stage. ``None``
        is a statement of its own -- "this model projects nothing" -- so a
        cache baked with a projection is not reused for a model that
        dropped it.
    """
    if not stages:
        return None
    return tuple(stage.as_record() for stage in stages)


def build_dense(stages: Sequence[DenseStage]) -> torch.nn.Module | None:
    """Build the FP32 projection module the declared stages describe.

    Args:
        stages: Stages returned by :func:`read_dense_modules`.

    Returns:
        An eval-mode module applying every stage in order, or ``None``
        when nothing is declared -- in which case the wrappers and the
        baseline keep computing exactly what they did before.

    Raises:
        ValueError: If a stage's weights are missing, hold something other
            than the declared linear layer, or contradict the declared
            widths.
    """
    if not stages:
        return None
    layers: list[torch.nn.Module] = []
    for stage in stages:
        layers.append(_build_dense_linear(stage))
        # An identity activation adds no layer at all: a no-op node would
        # only cost a graph operation and change nothing.
        if stage.activation == DENSE_ACTIVATION_TANH:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers).eval()


def load_dense(model_dir: Path) -> tuple[torch.nn.Module | None, tuple[dict[str, Any], ...] | None]:
    """Read and build the Dense projection a model directory declares.

    The single entry point a backend's ``load`` calls, before the backbone
    weights are read: everything that can make a model uncompilable is
    decided here.

    Args:
        model_dir: Local HuggingFace-format model directory.

    Returns:
        Tuple of the projection module and its metadata description, both
        ``None`` for a model that declares no projection.

    Raises:
        ValueError: If the declaration cannot be read or reproduced.
    """
    stages = read_dense_modules(model_dir)
    return build_dense(stages), dense_record(stages)


def _read_module_entries(
    modules_path: Path, requirement: str = _MODULE_CHAIN_REQUIREMENT
) -> list[tuple[str, Any]]:
    """Read a ``modules.json`` into ``(type, path)`` pairs, in declared order.

    Args:
        modules_path: The declaration file.
        requirement: Sentence describing the chain the caller reproduces,
            appended to every error. It differs per caller, since the
            chain an embedding model declares is not the one a generative
            reranker does.

    Returns:
        One pair per declared module. The path is returned unvalidated;
        only the modules that are actually read need one.

    Raises:
        ValueError: If the file is unreadable, is not valid JSON, or does
            not hold a list of objects naming a module type.
    """
    try:
        raw = modules_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot read the module declaration '{modules_path}': {exc}. {requirement}"
        ) from exc
    try:
        declaration = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"'{modules_path}' is not valid JSON: {exc}. {requirement}") from exc
    if not isinstance(declaration, list) or not declaration:
        raise ValueError(
            f"'{modules_path}' does not contain a non-empty list of modules. {requirement}"
        )
    entries: list[tuple[str, Any]] = []
    for index, entry in enumerate(declaration):
        module_type = entry.get("type") if isinstance(entry, dict) else None
        if not isinstance(module_type, str):
            raise ValueError(
                f"module {index} of '{modules_path}' does not name a module type. {requirement}"
            )
        entries.append((module_type, entry.get("path")))
    return entries


def _check_module_chain(modules_path: Path, module_types: Sequence[str]) -> None:
    """Validate the declared module order against the chain this backend reproduces.

    Args:
        modules_path: The declaration file, named in the error.
        module_types: Declared module types, in order.

    Raises:
        ValueError: If the chain is anything but a Transformer, one
            Pooling, any number of Dense modules and an optional trailing
            Normalize.
    """
    leading = list(module_types[:2])
    if leading != [ST_MODULE_TRANSFORMER, ST_MODULE_POOLING]:
        raise _module_chain_error(modules_path, module_types)
    rest = list(module_types[2:])
    dense_count = 0
    while dense_count < len(rest) and rest[dense_count] == ST_MODULE_DENSE:
        dense_count += 1
    trailing = rest[dense_count:]
    if trailing and trailing != [ST_MODULE_NORMALIZE]:
        raise _module_chain_error(modules_path, module_types)


def _module_chain_error(modules_path: Path, module_types: Sequence[str]) -> ValueError:
    """Build the error raised for a chain this backend cannot reproduce.

    Args:
        modules_path: The declaration file.
        module_types: Declared module types, quoted back so the reason is
            actionable without opening the file.

    Returns:
        The :class:`ValueError` to raise.
    """
    declared = " -> ".join(module_types) if module_types else "none"
    return ValueError(
        f"'{modules_path}' declares a module chain this backend cannot reproduce "
        f"(declared: {declared}). {_MODULE_CHAIN_REQUIREMENT}"
    )


def _check_dense_widths(stages: Sequence[DenseStage]) -> None:
    """Validate that consecutive projections feed into each other.

    A stage takes the previous stage's output, so declared widths that do
    not chain describe a projection no model could apply. Refusing it here
    keeps the failure a readable statement about the declaration rather
    than a shape error raised deep inside the trace.

    Args:
        stages: The declared stages, in application order.

    Raises:
        ValueError: If a stage's input width is not its predecessor's
            output width.
    """
    for previous, stage in zip(stages, stages[1:], strict=False):
        if stage.in_features != previous.out_features:
            raise ValueError(
                f"the Dense module '{stage.path}' declares in_features="
                f"{stage.in_features}, but the preceding '{previous.path}' produces "
                f"{previous.out_features} values"
            )


def _read_dense_stage(model_dir: Path, modules_path: Path, declared_path: Any) -> DenseStage:
    """Describe one declared Dense module from its own ``config.json``.

    Args:
        model_dir: Model directory the module is declared in.
        modules_path: The declaration file, named in path errors.
        declared_path: The module's ``path`` value as declared.

    Returns:
        The described stage.

    Raises:
        ValueError: If the path does not address a directory inside the
            model directory, or if the module's declaration is missing,
            malformed, or names an activation this backend cannot apply.
    """
    module_dir = _dense_module_directory(model_dir, modules_path, declared_path)
    config_path = module_dir / DENSE_CONFIG_FILENAME
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read the Dense declaration '{config_path}': {exc}") from exc
    try:
        declaration = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"the Dense declaration '{config_path}' is not valid JSON: {exc}") from exc
    if not isinstance(declaration, dict):
        raise ValueError(f"the Dense declaration '{config_path}' is not a JSON object")

    in_features = _dense_width(config_path, declaration, DENSE_IN_FEATURES_KEY)
    out_features = _dense_width(config_path, declaration, DENSE_OUT_FEATURES_KEY)
    bias = declaration.get(DENSE_BIAS_FLAG_KEY)
    if not isinstance(bias, bool):
        raise ValueError(
            f"the Dense declaration '{config_path}' does not declare "
            f"'{DENSE_BIAS_FLAG_KEY}' as a boolean (got {bias!r})"
        )
    activation = declaration.get(DENSE_ACTIVATION_KEY)
    if activation not in DENSE_ACTIVATIONS:
        supported = ", ".join(DENSE_ACTIVATIONS)
        raise ValueError(
            f"the Dense declaration '{config_path}' names the activation function "
            f"{activation!r}, which this backend cannot reproduce (supported: {supported})"
        )
    return DenseStage(
        path=module_dir,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        activation=DENSE_ACTIVATIONS[activation],
    )


def _dense_module_directory(model_dir: Path, modules_path: Path, declared_path: Any) -> Path:
    """Resolve a declared module path against the model directory.

    The model directory is the only place a declaration may address:
    reading a file it points at elsewhere on the machine is never part of
    compiling a model.

    Args:
        model_dir: Model directory the declaration belongs to.
        modules_path: The declaration file, named in the error.
        declared_path: The declared ``path`` value.

    Returns:
        The module directory, joined onto ``model_dir`` (not resolved, so
        the path reads as the user's own).

    Raises:
        ValueError: If the path is not a usable relative path inside the
            model directory.
    """
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise ValueError(
            f"'{modules_path}' declares a '{ST_MODULE_DENSE}' module without a usable "
            f"path (got {declared_path!r})"
        )
    module_dir = model_dir / declared_path
    root = model_dir.resolve()
    resolved = module_dir.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(
            f"'{modules_path}' declares the '{ST_MODULE_DENSE}' module path "
            f"{declared_path!r}, which is not inside the model directory"
        )
    return module_dir


def _dense_width(config_path: Path, declaration: dict[str, Any], key: str) -> int:
    """Read one positive width from a Dense declaration.

    Args:
        config_path: The declaration file, named in the error.
        declaration: Its parsed contents.
        key: Field to read.

    Returns:
        The declared width.

    Raises:
        ValueError: If the field is missing or is not a positive integer.
            ``bool`` is rejected explicitly: it is a subclass of ``int``,
            but a JSON ``true`` is not a width.
    """
    value = declaration.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"the Dense declaration '{config_path}' does not declare '{key}' as a "
            f"positive integer (got {value!r})"
        )
    return int(value)


def _build_dense_linear(stage: DenseStage) -> torch.nn.Linear:
    """Build one stage's linear layer from its checkpoint.

    Args:
        stage: The declared stage.

    Returns:
        An FP32 ``torch.nn.Linear`` holding the checkpoint's tensors.
        Conversion to a lower precision happens in the Core ML conversion
        step, never here.

    Raises:
        ValueError: If the checkpoint is missing, holds anything but the
            declared linear layer, or contradicts the declared widths.
    """
    state = _read_dense_weights(stage)
    expected = {DENSE_WEIGHT_KEY} | ({DENSE_BIAS_KEY} if stage.bias else set())
    if set(state) != expected:
        raise ValueError(
            f"the Dense weights in '{stage.path}' hold {sorted(state)}, but the "
            f"declaration describes exactly {sorted(expected)}"
        )
    weight = state[DENSE_WEIGHT_KEY]
    if tuple(weight.shape) != (stage.out_features, stage.in_features):
        raise ValueError(
            f"the Dense weights in '{stage.path}' have shape {tuple(weight.shape)}, "
            f"which contradicts the declared in_features={stage.in_features} / "
            f"out_features={stage.out_features}"
        )
    linear = torch.nn.Linear(stage.in_features, stage.out_features, bias=stage.bias)
    parameters = {"weight": weight.to(torch.float32)}
    if stage.bias:
        bias = state[DENSE_BIAS_KEY]
        if tuple(bias.shape) != (stage.out_features,):
            raise ValueError(
                f"the Dense bias in '{stage.path}' has shape {tuple(bias.shape)}, "
                f"which contradicts the declared out_features={stage.out_features}"
            )
        parameters["bias"] = bias.to(torch.float32)
    linear.load_state_dict(parameters)
    return linear.eval()


def _read_dense_weights(stage: DenseStage) -> dict[str, torch.Tensor]:
    """Read one stage's checkpoint tensors.

    safetensors is preferred; the pickle-based file is a fallback that a
    source is only ever resolved with when the caller opted into pickle
    weights, and it is read with ``weights_only=True`` so loading it
    executes no pickled code.

    Args:
        stage: The declared stage.

    Returns:
        The checkpoint's tensors, keyed as stored.

    Raises:
        ValueError: If neither file is present, or reading one failed.
    """
    safetensors_path = stage.path / DENSE_WEIGHTS_FILENAME
    pickle_path = stage.path / DENSE_PICKLE_WEIGHTS_FILENAME
    try:
        if safetensors_path.is_file():
            return dict(load_file(str(safetensors_path)))
        if pickle_path.is_file():
            return dict(torch.load(pickle_path, map_location="cpu", weights_only=True))
    except Exception as exc:
        raise ValueError(f"cannot read the Dense weights in '{stage.path}': {exc}") from exc
    raise ValueError(
        f"the declared Dense module '{stage.path}' holds neither "
        f"'{DENSE_WEIGHTS_FILENAME}' nor '{DENSE_PICKLE_WEIGHTS_FILENAME}'"
    )


def read_logit_score(model_dir: Path) -> tuple[int, int]:
    """Read the two vocabulary ids a generative reranker scores with.

    Such a reranker carries no scoring head: it asks a language model a
    yes/no question and reads the logits of two vocabulary entries at the
    final position. Which two those are is a property of the checkpoint,
    declared next to it, and this is the only place it is read from.

    Args:
        model_dir: Local HuggingFace-format model directory.

    Returns:
        ``(true_token_id, false_token_id)``: the id whose logit rises with
        relevance first, the id it is scored against second.

    Raises:
        ValueError: If the declaration is missing, unreadable, malformed,
            or does not name two different non-negative integer ids.
    """
    path = model_dir / LOGIT_SCORE_DIRNAME / LOGIT_SCORE_CONFIG_FILENAME
    declaration = _read_json_object(path, "the scoring module", _LOGIT_SCORE_REQUIREMENT)
    true_id = _logit_score_id(path, declaration, LOGIT_SCORE_TRUE_KEY)
    false_id = _logit_score_id(path, declaration, LOGIT_SCORE_FALSE_KEY)
    if true_id == false_id:
        # Subtracting a logit from itself is a constant zero: such a model
        # would compile and serve, and rank everything equally.
        raise ValueError(
            f"'{path}' names the same id ({true_id}) for both answers, so the score "
            f"would always be zero. {_LOGIT_SCORE_REQUIREMENT}"
        )
    return true_id, false_id


def _read_json_object(path: Path, description: str, requirement: str) -> dict[str, Any]:
    """Read one declaration file that must hold a JSON object.

    Args:
        path: File to read.
        description: What the file is, named in the error.
        requirement: Sentence describing what a valid declaration looks
            like, appended to every error so the message is actionable
            without opening the file.

    Returns:
        The parsed object.

    Raises:
        ValueError: If the file is missing, unreadable, not valid JSON, or
            does not hold a JSON object.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {description} '{path}': {exc}. {requirement}") from exc
    try:
        declaration = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"'{path}' is not valid JSON: {exc}. {requirement}") from exc
    if not isinstance(declaration, dict):
        raise ValueError(f"'{path}' does not contain a JSON object. {requirement}")
    return declaration


def _logit_score_id(path: Path, declaration: dict[str, Any], key: str) -> int:
    """Read one vocabulary id from a scoring declaration.

    Args:
        path: The declaration file, named in the error.
        declaration: Its parsed contents.
        key: Field to read.

    Returns:
        The declared id.

    Raises:
        ValueError: If the field is missing or is not a non-negative
            integer. ``bool`` is rejected explicitly: it is a subclass of
            ``int``, but a JSON ``true`` is not a vocabulary index.
    """
    value = declaration.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"'{path}' does not declare '{key}' as a non-negative integer "
            f"(got {value!r}). {_LOGIT_SCORE_REQUIREMENT}"
        )
    return int(value)


def check_scoring_module_chain(model_dir: Path) -> None:
    """Validate that ``model_dir`` declares the chain a generative reranker is.

    Args:
        model_dir: Local HuggingFace-format model directory.

    Raises:
        ValueError: If the declaration is missing, unreadable, malformed,
            or declares anything but one Transformer module followed by
            one scoring module.
    """
    modules_path = model_dir / ST_MODULES_FILENAME
    entries = _read_module_entries(modules_path, _SCORING_CHAIN_REQUIREMENT)
    module_types = [module_type for module_type, _ in entries]
    expected = (ST_TRANSFORMER_SUFFIX, ST_LOGIT_SCORE_SUFFIX)
    if len(module_types) != len(expected) or not all(
        declared.endswith(suffix) for declared, suffix in zip(module_types, expected, strict=True)
    ):
        declared = " -> ".join(module_types) if module_types else "none"
        raise ValueError(
            f"'{modules_path}' declares a module chain this backend cannot reproduce "
            f"(declared: {declared}). {_SCORING_CHAIN_REQUIREMENT}"
        )


def read_default_prompt(model_dir: Path) -> str | None:
    """Read the prompt a sentence-transformers directory applies by default.

    Args:
        model_dir: Local HuggingFace-format model directory.

    Returns:
        The prompt string the directory's default prompt name addresses,
        or the one named :data:`ST_FALLBACK_PROMPT_NAME` when it names no
        default; ``None`` when the file is absent, unreadable, malformed
        or declares no such prompt. Nothing here decides whether a model
        may be compiled without one -- that is the backend's call.
    """
    try:
        raw = (model_dir / ST_CONFIG_FILENAME).read_text(encoding="utf-8")
        declaration = json.loads(raw)
    except (OSError, ValueError):
        # Most model directories carry no such file at all, and a broken
        # one answers the question no better than a missing one; the
        # caller decides what a missing prompt means.
        return None
    if not isinstance(declaration, dict):
        return None
    prompts = declaration.get(ST_PROMPTS_KEY)
    if not isinstance(prompts, dict):
        return None
    name = declaration.get(ST_DEFAULT_PROMPT_NAME_KEY)
    if name is None:
        # Not stating a default is the ordinary case: the conventional
        # name is then the one to look up. A default that is stated but
        # is not a name, on the other hand, is a broken declaration, and
        # falling back for it would be guessing at what the model asks.
        name = ST_FALLBACK_PROMPT_NAME
    if not isinstance(name, str):
        return None
    prompt = prompts.get(name)
    return prompt if isinstance(prompt, str) else None


def tokenize_templated_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    seq_len: int,
    template: PairTemplate,
) -> dict[str, np.ndarray]:
    """Lay (query, document) pairs out into a template, as the runtime does.

    The compile-time counterpart of
    :func:`eeane.runtime.tokenize_pairs`'s template path: both must build
    byte-identical rows, or the model would be converted for a different
    input than it is served with.

    Args:
        tokenizer: Tokenizer of the model directory being compiled.
        pairs: (query, document) pairs, in order.
        seq_len: Fixed sequence length S every row is built to.
        template: Template the pairs are laid out with.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(pairs), seq_len)`` and dtype ``np.int32``.

    Raises:
        ValueError: If ``seq_len`` is not positive, if the template's own
            tokens leave no room for the pair's text, or if the tokenizer
            defines no padding id to fill the rows with.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be a positive integer (got {seq_len})")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise ValueError(
            "the tokenizer defines no pad token, so the fixed-length rows of a templated "
            "pair encoding have nothing to be filled with"
        )
    # The template's fixed halves already spell their control tokens out
    # as text, so the tokenizer must not add its own on top of them.
    prefix_ids = list(tokenizer.encode(template.prefix, add_special_tokens=False))
    suffix_ids = list(tokenizer.encode(template.suffix, add_special_tokens=False))
    fixed = len(prefix_ids) + len(suffix_ids)
    budget = seq_len - fixed
    if budget < 1:
        raise ValueError(
            f"the pair template takes {fixed} of the bucket's {seq_len} tokens, leaving no "
            "room for the query and the document; compile this model with a longer bucket"
        )

    input_ids = np.full((len(pairs), seq_len), int(pad_id), dtype=np.int32)
    attention_mask = np.zeros((len(pairs), seq_len), dtype=np.int32)
    for row, (query, document) in enumerate(pairs):
        body = _render_pair_body(template.body_format, query, document)
        # Encoded as one single string rather than as a query and a
        # document encoded apart and joined: with a sub-word vocabulary
        # the tokens covering the seam between them differ from the ones
        # either side produces alone.
        body_ids = list(tokenizer.encode(body, add_special_tokens=False))
        # Only the body is cut, and from its right end: the prefix and the
        # suffix are what make the input the question the model answers.
        ids = [*prefix_ids, *body_ids[:budget], *suffix_ids]
        # Written from position 0, so whatever is left over at the end of
        # the row is the padding: right-padded, as last-token pooling
        # assumes.
        input_ids[row, : len(ids)] = ids
        attention_mask[row, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _render_pair_body(body_format: str, query: str, document: str) -> str:
    """Write one (query, document) pair into a template's body.

    Args:
        body_format: Body format of the template being applied.
        query: Query text to write at the ``{query}`` marker.
        document: Document text to write at the ``{document}`` marker.

    Returns:
        The body with both texts written into it.
    """
    # Substituted in one left-to-right pass, and never through
    # ``str.format``: formatting would choke on (or reinterpret) any other
    # brace the texts carry, while two successive replacements would look
    # inside the text just written in, so a document reading like a marker
    # would be substituted into.
    query_at = body_format.find(_QUERY_MARKER)
    document_at = body_format.find(_DOCUMENT_MARKER)
    if query_at <= document_at:
        first, first_text = _QUERY_MARKER, query
        second, second_text = _DOCUMENT_MARKER, document
    else:
        first, first_text = _DOCUMENT_MARKER, document
        second, second_text = _QUERY_MARKER, query
    head, _, rest = body_format.partition(first)
    middle, _, tail = rest.partition(second)
    return head + first_text + middle + second_text + tail


def score_pytorch_generative(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    seq_len: int,
    template: PairTemplate,
    true_token_id: int,
    false_token_id: int,
) -> np.ndarray:
    """Compute FP32 baseline scores of a generative reranker, row by row.

    The model is asked the templated question and answers with a full
    vocabulary of logits; the score is the difference between the two
    declared entries at the row's last real position. Going through the
    checkpoint's own output projection is the point of this baseline: the
    traced graph carries two rows cut out of that projection, so a wrong
    cut shows up here as a disagreement.

    Args:
        model: Causal language model loaded by a backend's reference path,
            in eval/FP32 mode.
        tokenizer: Tokenizer for the same model directory.
        pairs: (query, document) pairs, in order.
        seq_len: Fixed sequence length used for tokenization.
        template: Template the pairs are laid out with; the very one the
            traced graph is converted for.
        true_token_id: Vocabulary id whose logit rises with relevance.
        false_token_id: Vocabulary id it is scored against.

    Returns:
        Scores of shape ``(len(pairs),)``, dtype float32: one
        ``true - false`` logit difference per pair.
    """
    batch = tokenize_templated_pairs(tokenizer, pairs, seq_len, template)
    scores = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(pairs)):
            # nn.Embedding lookup requires int64 indices; the tokenization
            # returns int32 for Core ML compatibility, so cast here.
            input_ids = torch.from_numpy(batch["input_ids"][i : i + 1]).long()
            attention_mask = torch.from_numpy(batch["attention_mask"][i : i + 1]).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]  # (1, S, V)
            # The rows are right-padded, so the model answers at the last
            # position its mask marks as real.
            last = int(attention_mask.sum().item()) - 1
            row = logits[0, max(last, 0)]
            scores[i] = float(row[true_token_id].item() - row[false_token_id].item())
    return scores


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over the sequence dimension.

    Single source of truth shared by the in-graph wrapper
    (:class:`EmbeddingWrapper`) and the PyTorch baseline
    (:func:`encode_pytorch`); changing the formula for one consumer
    without the other would make the self-check compare two different
    functions.

    Args:
        hidden: Last hidden state, shape (B, S, H).
        attention_mask: Attention mask, shape (B, S).

    Returns:
        Pooled embeddings, shape (B, H).
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, S, 1)
    summed = (hidden * mask).sum(dim=1)  # (B, H)
    count = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return summed / count


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last unpadded position of every row.

    The pooling of a causal model: only the final real token has attended
    to the whole sequence, so it is the position that carries the sentence.
    Padding is assumed to be on the right (what the shared
    :func:`tokenize_batch` produces), so that position is
    ``attention_mask.sum(dim=1) - 1``. A row with no real token at all
    would give ``-1``; the index is clamped to ``0`` so such a row reads
    the first position instead of wrapping around to the last one.

    Single source of truth shared by the in-graph wrapper of the
    decoder-style backends and the PyTorch baseline
    (:func:`encode_pytorch`); changing the formula for one consumer
    without the other would make the self-check compare two different
    functions.

    Written without turning any shape or index into a Python number: doing
    so records an ``aten::Int`` node in the traced graph, which the Core ML
    converter cannot fold into a static gather.

    Args:
        hidden: Last hidden state, shape (B, S, H).
        attention_mask: Attention mask, shape (B, S), right-padded.

    Returns:
        Pooled embeddings, shape (B, H).
    """
    lengths = (attention_mask.sum(dim=1).to(torch.long) - 1).clamp(min=0)  # (B,)
    rows = torch.arange(hidden.shape[0], device=hidden.device)  # (B,)
    return hidden[rows, lengths]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid.

    Branches on the sign of ``x`` so ``np.exp`` is only ever evaluated on
    non-positive arguments, avoiding overflow for large-magnitude inputs.

    Args:
        x: Input array (raw logits).

    Returns:
        Array of the same shape as ``x``, with values in (0, 1).
    """
    is_positive = x >= 0
    exp_neg_abs = np.exp(-np.abs(x))
    return np.where(is_positive, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase, texts: list[str], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize texts into fixed-shape int32 arrays for Core ML input.

    Args:
        tokenizer: Tokenizer of the model directory being compiled.
        texts: Input sentences (prefixes, if any, must already be applied
            by the caller).
        seq_len: Fixed sequence length used for padding/truncation.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(texts), seq_len)`` and dtype ``np.int32``.
    """
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase, pairs: list[tuple[str, str]], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize (query, document) pairs into fixed-shape int32 arrays.

    Delegates to the tokenizer's built-in pair encoding
    (``tokenizer(queries, documents, ...)``) so that the pair template of
    the model at hand is produced by the tokenizer's own post_processor
    rather than reimplemented here. ``truncation=True`` uses the
    tokenizer's default ``longest_first`` strategy across both sequences.
    Any key other than ``input_ids``/``attention_mask`` returned by the
    tokenizer (e.g. ``token_type_ids``) is discarded, since the compiled
    graph only accepts those two inputs.

    Args:
        tokenizer: Tokenizer of the model directory being compiled.
        pairs: List of (query, document) pairs.
        seq_len: Fixed sequence length used for padding/truncation.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(pairs), seq_len)`` and dtype ``np.int32``.
    """
    queries = [query for query, _ in pairs]
    documents = [document for _, document in pairs]
    encoded = tokenizer(
        queries,
        documents,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def encode_pytorch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    seq_len: int,
    pooling: str = POOLING_MEAN,
    dense: torch.nn.Module | None = None,
) -> np.ndarray:
    """Compute FP32 baseline embeddings with a batch-size-1 loop.

    Args:
        model: Embedding model loaded by a backend's ``load``.
        tokenizer: Tokenizer for the same model directory.
        texts: Input sentences.
        seq_len: Fixed sequence length used for tokenization.
        pooling: Pooling mode to apply, one of :data:`POOLING_MODES`. It
            must match the pooling the traced wrapper performs; the caller
            is a backend that already refused any mode it cannot wrap.
        dense: Projection applied to the pooled vector, as built by
            :func:`build_dense`. It must be the very projection the traced
            wrapper applies, or the two sides of the self-check would
            compute different functions.

    Returns:
        Embeddings array of shape ``(len(texts), width)``, dtype float32,
        where the width is the backbone's hidden size or, with a
        projection, the width of its last stage.

    Raises:
        ValueError: If ``pooling`` is not a supported pooling mode.
    """
    if pooling not in POOLING_MODES:
        # Refused before the first forward pass: an unknown mode is a
        # caller error, not something to discover halfway through a batch.
        raise _unsupported_pooling(pooling)
    batch = tokenize_batch(tokenizer, texts, seq_len)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(len(texts)):
            # nn.Embedding lookup requires int64 indices; tokenize_batch
            # returns int32 for Core ML compatibility, so cast here.
            input_ids = torch.from_numpy(batch["input_ids"][i : i + 1]).long()
            attention_mask = torch.from_numpy(batch["attention_mask"][i : i + 1]).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs[0]  # (1, S, H)
            pooled = _pool(hidden, attention_mask, pooling)  # (1, H)
            if dense is not None:
                pooled = dense(pooled)  # (1, W)
            rows.append(pooled.numpy().astype(np.float32).reshape(-1))
    if not rows:
        # Nothing was encoded, so no projection was evaluated either and
        # the backbone's own width is all this can report.
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.stack(rows)


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Apply one pooling mode to one batch of hidden states.

    Every supported mode is named explicitly, so a mode added to
    :data:`POOLING_MODES` without a branch here is refused rather than
    silently pooled like one of the others.

    Args:
        hidden: Last hidden state, shape (B, S, H).
        attention_mask: Attention mask, shape (B, S).
        pooling: Mode to apply, one of :data:`POOLING_MODES`.

    Returns:
        Pooled embeddings, shape (B, H).

    Raises:
        ValueError: If ``pooling`` is not a supported pooling mode.
    """
    if pooling == POOLING_MEAN:
        return mean_pool(hidden, attention_mask)
    if pooling == POOLING_CLS:
        # The first position of an encoder row, which is never padding.
        return hidden[:, 0]
    if pooling == POOLING_LASTTOKEN:
        return last_token_pool(hidden, attention_mask)
    raise _unsupported_pooling(pooling)


def _unsupported_pooling(pooling: Any) -> ValueError:
    """Build the error raised for a pooling mode this module cannot apply.

    Args:
        pooling: The rejected mode, quoted back so the reason is actionable.

    Returns:
        The :class:`ValueError` to raise.
    """
    supported = ", ".join(POOLING_MODES)
    return ValueError(f"unsupported pooling '{pooling}' (supported: {supported})")


def score_pytorch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    seq_len: int,
) -> np.ndarray:
    """Compute FP32 baseline raw reranker logits with a batch-size-1 loop.

    Args:
        model: Reranker model loaded by a backend's ``load``.
        tokenizer: Tokenizer for the same model directory.
        pairs: List of (query, document) pairs.
        seq_len: Fixed sequence length used for tokenization.

    Returns:
        Raw logits array of shape (len(pairs),), dtype float32.
    """
    batch = tokenize_pairs(tokenizer, pairs, seq_len)
    scores = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(pairs)):
            # nn.Embedding lookup requires int64 indices; tokenize_pairs
            # returns int32 for Core ML compatibility, so cast here.
            input_ids = torch.from_numpy(batch["input_ids"][i : i + 1]).long()
            attention_mask = torch.from_numpy(batch["attention_mask"][i : i + 1]).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]  # (1, 1)
            scores[i] = logits.reshape(-1)[0].item()
    return scores


class EmbeddingWrapper(torch.nn.Module):
    """Wraps a backbone model and performs masked mean pooling in-graph.

    The output matches a sentence-transformers model whose modules are a
    Transformer, mean Pooling and the declared Dense projections, without
    normalization.
    """

    def __init__(self, model: torch.nn.Module, dense: torch.nn.Module | None = None) -> None:
        """Store the backbone model and the projection to apply after pooling.

        Args:
            model: Backbone loaded in eval/FP32 mode with
                ``config.return_dict = False``.
            dense: Projection applied to the pooled vector, as built by
                :func:`build_dense`; ``None`` for a model that declares
                none, which traces to exactly the graph it always did.
        """
        super().__init__()
        self.model = model
        self.dense = dense

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute pooled (and, if declared, projected) sentence embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            Embeddings of shape (B, hidden_size), or (B, projected width)
            when a projection was given.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs[0]  # (B, S, H)
        pooled = mean_pool(hidden, attention_mask)
        if self.dense is None:
            return pooled
        return self.dense(pooled)


class ClsEmbeddingWrapper(torch.nn.Module):
    """Wraps a backbone model and takes the first token's state in-graph.

    The output matches a sentence-transformers model whose modules are a
    Transformer, CLS Pooling and the declared Dense projections, without
    normalization. The attention mask still reaches the backbone, but it
    does not take part in the pooling: the first position is never
    padding.
    """

    def __init__(self, model: torch.nn.Module, dense: torch.nn.Module | None = None) -> None:
        """Store the backbone model and the projection to apply after pooling.

        Args:
            model: Backbone loaded in eval/FP32 mode with
                ``config.return_dict = False``.
            dense: Projection applied to the pooled vector, as built by
                :func:`build_dense`; ``None`` for a model that declares
                none, which traces to exactly the graph it always did.
        """
        super().__init__()
        self.model = model
        self.dense = dense

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute CLS-pooled (and, if declared, projected) sentence embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            The first token's state of shape (B, hidden_size), or its
            projection of shape (B, projected width) when one was given.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs[0]  # (B, S, H)
        pooled = hidden[:, 0]
        if self.dense is None:
            return pooled
        return self.dense(pooled)


class RerankerWrapper(torch.nn.Module):
    """Wraps a sequence-classification model and exposes raw logits.

    The Core ML graph reproduces the HF forward as-is (the model's own
    pooling plus its classification head). Sigmoid is applied outside the
    graph, in Python post-processing.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the classification model.

        Args:
            model: Sequence-classification model loaded in eval/FP32 mode
                with ``config.return_dict = False``.
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
