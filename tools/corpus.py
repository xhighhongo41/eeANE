"""Read-only test-data loaders shared by the verification tooling.

Ported from ``poc/common.py`` (unchanged in behaviour): the frozen PoC
tree stays a historical record, while this module is the single source
of truth for tools that need the fixed Aozora Bunko corpus, the ruri-v3
instruction prefixes, or the hand-written reranker query set without
pulling in ``torch``/``transformers`` (none of the functions below need
them).
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Hand-written reranker query set, consumed by verify_server.py.
RERANK_QUERIES_PATH: Path = _REPO_ROOT / "testdata" / "rerank_queries.json"

# Fixed Aozora Bunko test corpus directory.
CORPUS_DIR: Path = _REPO_ROOT / "testdata" / "corpus"

# ruri-v3 instruction prefixes used to prepend to raw sentences.
PREFIXES: dict[str, str] = {
    "none": "",
    "topic": "トピック: ",
    "query": "検索クエリ: ",
    "document": "検索文書: ",
}

# Hand-written Japanese sentences covering varied topics (science, history,
# technology, daily life), used to build the prefix-confirmation test set.
# Kept as a fixed constant so results are reproducible across runs.
PREFIX_TEST_SENTENCES: list[str] = [
    # Questions (4)
    "光合成はどのようにして太陽の光を化学エネルギーに変換しているのですか。",
    "明治維新によって日本の政治体制はどのように変化したのでしょうか。",
    "ニューラルネットワークの学習がうまく収束しないのはなぜですか。",
    "毎朝コーヒーを飲むと眠気が覚めるのはどうしてなのでしょうか。",
    # Statements (4)
    "光合成は植物が太陽光のエネルギーを使って栄養分を作り出す仕組みである。",
    "明治維新は江戸幕府が倒れて近代的な中央集権国家が誕生した出来事だ。",
    "ニューラルネットワークは層を重ねることで複雑な関数を近似できる。",
    "毎朝のコーヒーにはカフェインが含まれており集中力を高めてくれる。",
]


def load_corpus_paragraphs(max_kokoro: int = 30, min_chars: int = 40) -> list[str]:
    """Load a deterministic paragraph list from the fixed Aozora corpus.

    Splits all three works on blank lines, strips each paragraph, and
    keeps only paragraphs with at least ``min_chars`` characters. All
    filtered paragraphs are kept for kumonoito and sangetsuki, while only
    the first ``max_kokoro`` filtered paragraphs are kept for kokoro. File
    order is fixed: kumonoito -> sangetsuki -> kokoro.

    Args:
        max_kokoro: Maximum number of leading paragraphs to take from
            kokoro.txt (the other two works contribute all paragraphs).
        min_chars: Minimum paragraph length (in characters) to keep.

    Returns:
        Paragraphs in file order (kumonoito, sangetsuki, kokoro).
    """
    paragraphs: list[str] = []
    file_limits = [
        (CORPUS_DIR / "kumonoito.txt", None),
        (CORPUS_DIR / "sangetsuki.txt", None),
        (CORPUS_DIR / "kokoro.txt", max_kokoro),
    ]
    for path, limit in file_limits:
        text = path.read_text(encoding="utf-8")
        blocks = _split_paragraphs(text)
        blocks = [block for block in blocks if len(block) >= min_chars]
        if limit is not None:
            blocks = blocks[:limit]
        paragraphs.extend(blocks)
    return paragraphs


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines.

    Consecutive blank lines (including whitespace-only lines) are treated
    as a paragraph separator. Uses ``str.splitlines`` to be agnostic to
    the line-ending style.

    Args:
        text: Full text to split.

    Returns:
        Stripped paragraph strings (empty paragraphs are excluded).
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def load_rerank_queries() -> list[dict]:
    """Load the hand-written reranker query set.

    Returns:
        List of dicts with keys ``id``, ``query``, and ``source_work``, in
        the order stored on disk at :data:`RERANK_QUERIES_PATH`.
    """
    with RERANK_QUERIES_PATH.open(encoding="utf-8") as f:
        return json.load(f)
