"""Unit tests for tools/clean_aozora.py.

These tests use synthetic Aozora Bunko-style text so they do not depend
on the network or on files under testdata/corpus/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from clean_aozora import (  # noqa: E402
    clean,
    strip_annotations,
    strip_footer,
    strip_header,
    strip_ruby,
)

# A symbol-legend block as it appears near the top of Aozora Bunko files,
# delimited by two horizontal-rule lines of 8+ hyphens.
_LEGEND_BLOCK = (
    "-------------------------------------------------------\n"
    "【テキスト中に現れる記号について】\n"
    "\n"
    "《》：ルビ\n"
    "-------------------------------------------------------\n"
)


def test_strip_ruby_removes_reading() -> None:
    """Ruby readings such as ``《あおぞら》`` are removed."""
    assert strip_ruby("青空《あおぞら》文庫") == "青空文庫"


def test_strip_ruby_removes_start_marker() -> None:
    """The ruby-start marker ``｜`` is removed."""
    assert strip_ruby("丁度｜地獄《じごく》の底") == "丁度地獄の底"


def test_strip_annotations_removes_indent_instruction() -> None:
    """Bracketed instructions such as ``［＃ここから2字下げ］`` are removed."""
    assert strip_annotations("［＃ここから2字下げ］本文") == "本文"


def test_strip_annotations_removes_gaiji_note_with_marker() -> None:
    """External-character notes prefixed with ``※`` are removed entirely."""
    text = "※［＃「てへん＋劣」、第3水準1-84-77］陀多"
    assert strip_annotations(text) == "陀多"


def test_strip_header_removes_title_author_and_legend_block() -> None:
    """Title/author lines and the symbol-legend block are removed."""
    text = "山月記\n中島敦\n\n" + _LEGEND_BLOCK + "\n本文一行目\n"
    assert strip_header(text) == "本文一行目\n"


def test_strip_header_removes_title_author_without_legend_block() -> None:
    """Without a legend block, the first two non-blank lines are removed."""
    text = "山月記\n中島敦\n\n本文一行目\n"
    assert strip_header(text) == "本文一行目\n"


def test_strip_footer_removes_bibliographic_block() -> None:
    """Everything from the ``底本：`` line onward is removed."""
    text = "本文\n\n底本：「こころ」集英社文庫、集英社\n入力：山田太郎\n"
    assert strip_footer(text) == "本文\n\n"


def test_clean_end_to_end() -> None:
    """A full synthetic sample reduces to trimmed body text only."""
    raw = (
        "山月記\n"
        "中島敦\n"
        "\n" + _LEGEND_BLOCK + "\n"
        "［＃８字下げ］一［＃「一」は中見出し］\n"
        "\n"
        "　隴西《ろうさい》の李徴《りちょう》は博学｜才穎《さいえい》、"
        "※［＃「てへん＋劣」、第3水準1-84-77］の男である。\n"
        "\n"
        "\n"
        "底本：「山月記」新潮文庫、新潮社\n"
        "入力：山田太郎\n"
        "校正：鈴木花子\n"
    )
    expected = "一\n\n　隴西の李徴は博学才穎、の男である。"
    assert clean(raw) == expected
