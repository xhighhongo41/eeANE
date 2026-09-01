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

# Pooling modes the shared embedding helpers implement. A backend records
# the mode it detected on its LoadedModel handle, and both the wrapper
# selection and the FP32 baseline are driven by that value.
POOLING_MEAN = "mean"
POOLING_CLS = "cls"
POOLING_MODES: tuple[str, ...] = (POOLING_MEAN, POOLING_CLS)

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
        ``"mean"`` or ``"cls"``.

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


def _read_module_entries(modules_path: Path) -> list[tuple[str, Any]]:
    """Read a ``modules.json`` into ``(type, path)`` pairs, in declared order.

    Args:
        modules_path: The declaration file.

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
            f"cannot read the module declaration '{modules_path}': {exc}. "
            f"{_MODULE_CHAIN_REQUIREMENT}"
        ) from exc
    try:
        declaration = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"'{modules_path}' is not valid JSON: {exc}. {_MODULE_CHAIN_REQUIREMENT}"
        ) from exc
    if not isinstance(declaration, list) or not declaration:
        raise ValueError(
            f"'{modules_path}' does not contain a non-empty list of modules. "
            f"{_MODULE_CHAIN_REQUIREMENT}"
        )
    entries: list[tuple[str, Any]] = []
    for index, entry in enumerate(declaration):
        module_type = entry.get("type") if isinstance(entry, dict) else None
        if not isinstance(module_type, str):
            raise ValueError(
                f"module {index} of '{modules_path}' does not name a module type. "
                f"{_MODULE_CHAIN_REQUIREMENT}"
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
            must match the pooling the traced wrapper performs.
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
        supported = ", ".join(POOLING_MODES)
        raise ValueError(f"unsupported pooling '{pooling}' (supported: {supported})")
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
            if pooling == POOLING_MEAN:
                pooled = mean_pool(hidden, attention_mask)  # (1, H)
            else:
                pooled = hidden[:, 0]  # CLS token, (1, H)
            if dense is not None:
                pooled = dense(pooled)  # (1, W)
            rows.append(pooled.numpy().astype(np.float32).reshape(-1))
    if not rows:
        # Nothing was encoded, so no projection was evaluated either and
        # the backbone's own width is all this can report.
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.stack(rows)


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
