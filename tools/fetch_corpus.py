"""Download and clean the Aozora Bunko test corpus.

Downloads the "ruby" edition plain-text file for each work listed in
``WORKS`` from its Aozora Bunko card page, cleans it with
:func:`clean_aozora.clean`, and writes the result as UTF-8 (LF) text
under ``testdata/corpus/``. The bibliographic source-edition info
(the "底本：" footer block) is printed to stdout for use when writing
``testdata/corpus/README.md``.

Network access only happens while this script runs; the rest of the
repository (tests, PoC code) works entirely from the corpus files
already committed under ``testdata/corpus/``.

Usage:
    uv run python tools/fetch_corpus.py
"""

from __future__ import annotations

import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from clean_aozora import clean, extract_source_info

# Identifies oneself politely to the Aozora Bunko server.
_USER_AGENT = "eeane-corpus-fetch/0.1 (test corpus download script)"

# Matches the link to the ruby-edition plain-text zip on a card page,
# e.g. href="./files/92_ruby_164.zip".
_ZIP_LINK_RE = re.compile(r'href="([^"]*_ruby_[^"]*\.zip)"')

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _REPO_ROOT / "testdata" / "corpus" / "raw"
_CORPUS_DIR = _REPO_ROOT / "testdata" / "corpus"


@dataclass(frozen=True)
class Work:
    """A single corpus source: an Aozora Bunko card page and output name."""

    name: str
    card_url: str


WORKS: tuple[Work, ...] = (
    Work(name="kumonoito", card_url="https://www.aozora.gr.jp/cards/000879/card92.html"),
    Work(name="sangetsuki", card_url="https://www.aozora.gr.jp/cards/000119/card624.html"),
    Work(name="kokoro", card_url="https://www.aozora.gr.jp/cards/000148/card773.html"),
)


def _fetch_bytes(url: str) -> bytes:
    """Fetch a URL over HTTPS with default (enabled) certificate verification.

    Args:
        url: URL to fetch.

    Returns:
        Raw response body.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


def _find_zip_url(card_html: str, card_url: str) -> str:
    """Find the ruby-edition zip link on a card page and resolve it.

    Args:
        card_html: HTML of the card page.
        card_url: URL the HTML was fetched from (for resolving the
            relative link).

    Returns:
        Absolute URL of the ``*_ruby_*.zip`` file.

    Raises:
        ValueError: If no matching link is found.
    """
    match = _ZIP_LINK_RE.search(card_html)
    if match is None:
        raise ValueError(f"no *_ruby_*.zip link found on card page: {card_url}")
    return urljoin(card_url, match.group(1))


def _download_zip(work: Work) -> Path:
    """Download (or reuse a cached copy of) a work's ruby-edition zip.

    If ``testdata/corpus/raw/<name>.zip`` already exists, it is reused
    and no network access is performed for this work.

    Args:
        work: Work to download.

    Returns:
        Path to the downloaded (or cached) zip file.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _RAW_DIR / f"{work.name}.zip"
    if zip_path.exists():
        print(f"[{work.name}] using cached {zip_path}")
        return zip_path

    print(f"[{work.name}] fetching card page: {work.card_url}")
    card_html = _fetch_bytes(work.card_url).decode("utf-8")
    zip_url = _find_zip_url(card_html, work.card_url)

    print(f"[{work.name}] downloading: {zip_url}")
    zip_bytes = _fetch_bytes(zip_url)
    zip_path.write_bytes(zip_bytes)
    return zip_path


def _extract_text(zip_path: Path) -> str:
    """Extract and decode the single text file bundled in an Aozora zip.

    Args:
        zip_path: Path to the downloaded ``*_ruby_*.zip`` file.

    Returns:
        Decoded text, with newlines as originally encoded.

    Raises:
        ValueError: If the zip contains no ``.txt`` entry.
    """
    with zipfile.ZipFile(zip_path) as archive:
        txt_names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not txt_names:
            raise ValueError(f"no .txt entry found in {zip_path}")
        raw_bytes = archive.read(txt_names[0])

    try:
        return raw_bytes.decode("cp932")
    except UnicodeDecodeError:
        return raw_bytes.decode("shift_jis")


def _process_work(work: Work) -> None:
    """Download, clean, and save a single work; print its source info.

    Args:
        work: Work to process.
    """
    zip_path = _download_zip(work)
    raw_text = _extract_text(zip_path).replace("\r\n", "\n").replace("\r", "\n")

    cleaned = clean(raw_text)
    output_path = _CORPUS_DIR / f"{work.name}.txt"
    output_path.write_text(cleaned + "\n", encoding="utf-8", newline="\n")
    print(f"[{work.name}] wrote {output_path} ({len(cleaned)} chars)")

    source_info = extract_source_info(raw_text)
    print(f"[{work.name}] source info:\n{source_info}\n")


def main() -> None:
    """Fetch, clean, and save the full corpus."""
    _CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for work in WORKS:
        _process_work(work)


if __name__ == "__main__":
    sys.exit(main())
