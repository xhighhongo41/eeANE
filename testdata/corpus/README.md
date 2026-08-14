# Fixed Test Corpus (Aozora Bunko)

This directory contains the fixed Japanese text corpus used for eeANE's
accuracy verification and performance benchmarks. Using a committed,
immutable corpus keeps measurements reproducible across runs and
machines.

## Sources

All three works are in the **public domain** (their authors died in
1916–1942, so copyright has expired). The texts were obtained from
[Aozora Bunko](https://www.aozora.gr.jp/), a volunteer-run digital
library of public-domain Japanese literature.

| File | Work | Author | Card page |
|---|---|---|---|
| `kumonoito.txt` | 蜘蛛の糸 (Kumo no Ito) | 芥川龍之介 (Ryunosuke Akutagawa) | https://www.aozora.gr.jp/cards/000879/card92.html |
| `sangetsuki.txt` | 山月記 (Sangetsuki) | 中島敦 (Atsushi Nakajima) | https://www.aozora.gr.jp/cards/000119/card624.html |
| `kokoro.txt` | こころ (Kokoro) | 夏目漱石 (Soseki Natsume) | https://www.aozora.gr.jp/cards/000148/card773.html |

Source editions (底本), as recorded in the original Aozora Bunko files:

- **蜘蛛の糸**: 「芥川龍之介全集2」ちくま文庫、筑摩書房 (1986年10月28日第1刷 /
  1996年7月15日第11刷)。親本: 筑摩全集類聚版芥川龍之介全集 (1971年3月〜11月)。
  Aozora file: published 1997-11-10, last revised 2011-01-28.
- **山月記**: 「李陵・山月記」新潮文庫、新潮社 (1969年9月20日発行)。
  Aozora file: published 1998-11-12, last revised 2010-11-02.
- **こころ**: 「こころ」集英社文庫、集英社 (1991年2月25日第1刷 /
  1995年6月14日第10刷)。初出: 「朝日新聞」1914年4月20日〜8月11日。
  Aozora file: published 1999-07-31, last revised 2010-10-31.

## Modifications

**These files are modified versions of the Aozora Bunko source files;
they are not the original distributions.** In accordance with Aozora
Bunko's file-handling guidelines, the following mechanical
modifications were applied by `tools/fetch_corpus.py` /
`tools/clean_aozora.py`:

- Removed ruby readings (`《...》`) and ruby-start markers (`｜`)
- Removed transcriber annotations and gaiji notes (`［＃...］`, `※［＃...］`)
- Removed the title/author header and the symbol-legend block
- Removed the bibliographic footer (the `底本：` block, reproduced above)
- Normalized newlines to LF and re-encoded from Shift_JIS to UTF-8

The body text itself is otherwise unmodified.

## Licensing note

These text files are public-domain data and are **not** covered by the
GPL-3.0-or-later license that applies to the eeANE source code.

## Regenerating

```
uv run python tools/fetch_corpus.py
```

Downloads go to `testdata/corpus/raw/` (not committed); cached zips are
reused on re-runs. Note that Aozora Bunko may revise its files over
time, so regenerated output can differ from the committed corpus — the
committed files are the reference for benchmarks.
