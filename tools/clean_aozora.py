"""Cleaning utilities for Aozora Bunko plain-text files.

Aozora Bunko ".txt" (ruby) files bundle a title/author header, an
optional symbol-legend block, ruby readings, transcriber annotations,
and a bibliographic footer around the plain body text. The functions
in this module strip all of that boilerplate down to body text only,
while preserving the paragraph structure (blank lines and full-width
leading spaces) of the original text.
"""

from __future__ import annotations

import re

# A horizontal rule made of 8 or more ASCII hyphens, used to delimit the
# "symbol legend" block near the top of an Aozora Bunko file.
_DELIMITER_RE = re.compile(r"^-{8,}[ \t]*$", re.MULTILINE)

# Ruby readings, e.g. "《あおぞら》".
_RUBY_RE = re.compile(r"《[^》]*》")

# Marker (U+FF5C FULLWIDTH VERTICAL LINE) indicating where a ruby-annotated
# word starts.
_RUBY_START_MARK = "｜"

# Transcriber annotations, e.g. "［＃ここから2字下げ］" or gaiji notes such
# as "※［＃「てへん＋劣」、第3水準1-84-77］" (the leading "※" is optional).
_ANNOTATION_RE = re.compile(r"※?［＃[^］]*］")

# The line that starts the bibliographic footer block.
_FOOTER_START_RE = re.compile(r"^底本[：:]", re.MULTILINE)

# Run of blank (or whitespace-only) lines at the very start of the text.
_LEADING_BLANK_RE = re.compile(r"\A(?:[ \t]*\n)+")


def strip_ruby(text: str) -> str:
    """Remove ruby readings and the ruby-start marker.

    Removes ``《...》`` reading annotations and the ``｜`` marker that
    indicates where a ruby-annotated word begins.

    Args:
        text: Input text.

    Returns:
        Text with ruby readings and the ruby-start marker removed.
    """
    without_readings = _RUBY_RE.sub("", text)
    return without_readings.replace(_RUBY_START_MARK, "")


def strip_annotations(text: str) -> str:
    """Remove Aozora Bunko transcriber annotations.

    Removes bracketed annotations such as ``［＃ここから2字下げ］`` and
    external-character (gaiji) notes such as
    ``※［＃「てへん＋劣」、第3水準1-84-77］``, including the leading
    ``※`` when present.

    Args:
        text: Input text.

    Returns:
        Text with transcriber annotations removed.
    """
    return _ANNOTATION_RE.sub("", text)


def strip_header(text: str) -> str:
    """Remove the leading title/author block and symbol-legend block.

    If a symbol-legend block (delimited by two horizontal-rule lines of
    8 or more hyphens) is present near the top of the text, everything
    from the start of the text through the end of the second delimiter
    line is removed (this also covers the title/author lines, which
    precede the block). Otherwise, the first two non-blank lines are
    treated as the title and author lines and removed. Any blank lines
    remaining before the body starts are trimmed.

    Args:
        text: Input text.

    Returns:
        Text with the leading header block removed.
    """
    delimiters = list(_DELIMITER_RE.finditer(text))
    if len(delimiters) >= 2:
        body = text[delimiters[1].end() :]
    else:
        lines = text.split("\n")
        non_blank_seen = 0
        cut = len(lines)
        for index, line in enumerate(lines):
            if line.strip():
                non_blank_seen += 1
                if non_blank_seen == 2:
                    cut = index + 1
                    break
        body = "\n".join(lines[cut:])
    return _LEADING_BLANK_RE.sub("", body)


def strip_footer(text: str) -> str:
    """Remove the bibliographic footer block.

    Removes the first line starting with ``底本：`` (fullwidth colon) or
    ``底本:`` (halfwidth colon) and everything after it.

    Args:
        text: Input text.

    Returns:
        Text with the footer block removed.
    """
    match = _FOOTER_START_RE.search(text)
    if match is None:
        return text
    return text[: match.start()]


def extract_source_info(text: str) -> str:
    """Extract the bibliographic footer block for documentation purposes.

    Used to surface the source-edition metadata (e.g. for
    ``testdata/corpus/README.md``) without embedding it in the cleaned
    corpus text.

    Args:
        text: Raw text (before cleaning) with newlines already
            normalized to ``\\n``.

    Returns:
        The ``底本：`` line onward, with surrounding whitespace
        stripped. Empty string if no footer block is found.
    """
    match = _FOOTER_START_RE.search(text)
    if match is None:
        return ""
    return text[match.start() :].strip()


def clean(text: str) -> str:
    """Clean a raw Aozora Bunko text down to plain body text.

    Applies, in order: newline normalization, header removal, footer
    removal, ruby removal, annotation removal, and trimming of leading
    and trailing blank lines. Paragraph structure (blank lines and
    full-width leading spaces within the body) is preserved.

    Args:
        text: Raw Aozora Bunko text (ruby edition).

    Returns:
        Cleaned body text.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    without_header = strip_header(normalized)
    without_footer = strip_footer(without_header)
    without_ruby = strip_ruby(without_footer)
    without_annotations = strip_annotations(without_ruby)
    return without_annotations.strip("\n")
