# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

[![PyPI](https://img.shields.io/pypi/v/eeane)](https://pypi.org/project/eeane/)
[![CI](https://github.com/xhighhongo41/eeANE/actions/workflows/ci.yml/badge.svg)](https://github.com/xhighhongo41/eeANE/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

eeANEは、Apple Silicon MacのApple Neural Engine (ANE) 上でテキスト
埋め込みモデルとrerankingモデルを実行します。モデルはHugging Face
配布形式のまま取得し、ローカルでCore ML形式の成果物へコンパイル
します。この成果物は数秒でロードでき、ANE上で実行されるため、
GPUとユニファイドメモリの大部分を他の作業のために空けておけます。

> **macOS 27以降をお使いの場合: 大きなモデルはeeANE 1.7.0以降で
> 作り直してください。** macOS 27では、Neural Engineが受け付ける
> モデルの大きさの上限が2 GiBから1 GiBへ下がりました。エラーも警告も
> 出ず、上限を超えるモデルは黙ってCPUで実行されます。Qwen3の0.6B
> モデルとXLM-RoBERTa large級のモデルは1 GiBを超えます(XLM-RoBERTaの
> モデルはその場合、出力がNaNになります)。eeANEを1.7.0以降へ上げ、
> macOS 27上で`eeane compile <id>`を実行してコンパイルし直してください:
> これらのモデルは自動的に、上限に収まる部分へ分割されるようになりました。
> [大きいモデルとNeural Engineの上限](#大きいモデルとneural-engineの上限)を
> 参照してください。

## ハイライト

- **ANE推論**: M2 Mac miniで埋め込み処理は最大約13,600実効
  トークン/秒に達し、PyTorchが同じモデルをMPS GPUでサービングした
  場合の2〜3倍です — その間GPUはアイドルのままです(詳細は後述の
  パフォーマンスを参照)。
- **モデルの改変も再配布も不要**: `eeane compile`はHugging Faceの
  モデルID(またはHF配布形式のローカルディレクトリ)を受け取って
  あなたのマシン上で変換し、組み込みのセルフチェックで変換結果を
  FP32のオリジナルと突き合わせて検証します。
- **標準API**: OpenAI互換の`/v1/embeddings`とInfinity互換の
  `/rerank`を提供するため、既存のクライアント(Open WebUIを含む)は
  base URLを変更するだけで接続できます。
- **常駐コストが低い**: モデルはオンデマンドで1秒未満のうちに
  ロードされ、アイドルタイムアウト後にアンロードされます。常駐
  させるサーバー本体はランタイム依存が5つだけの小さなPythonプロ
  セスです(torchもtransformersも不要)。
- **マルチモデルサービング**: リクエスト単位のルーティング、受付
  制御(429/503 + `Retry-After`)、同一リクエストの併合、グレース
  フルシャットダウンに対応。
- **現時点で対応するアーキテクチャ系統は5つ**: ModernBERTと
  XLM-RoBERTa(いずれもembeddingモデルとクロスエンコーダ型reranker
  モデルの両方に対応)、BERT(embeddingモデルのみに対応)、
  Qwen3(デコーダ型: 系列の最終トークンをプーリングするembedding
  モデルと、分類ヘッドの代わりにyes/noのロジット対からスコアを出す
  生成型reranker)、そしてGemma 3(双方向のembeddingモデルのみに
  対応)。さらなる系統の追加を計画中です。

## 動作要件

- Apple Silicon Mac (M1以降)
- macOS 13以降
- Python 3.11または3.12(3.13以降は未対応)。`uv`は対応する
  インタプリタを自動的に解決します。pipxやpip + venvでインストール
  する場合は、対応バージョンを自分で用意する必要があります。
- Xcodeコマンドラインツール (`xcode-select --install`) —
  `eeane compile`が`xcrun coremlcompiler`を使用します
- [uv](https://docs.astral.sh/uv/) — eeANEのインストールに推奨
  される方法(下記のインストールを参照)であり、開発ワークフロー
  にも必要です

eeANEはApple Neural Engineを必須とします: CPUのみでの実行はサポート
していません(既知の制限を参照)。Dockerもサポート対象外です —
macOS上のコンテナはLinux VM内で動作し、ANEはそのVMへパススルー
されません。

## 大きいモデルとNeural Engineの上限

**上限とは。** Neural Engineが受け付けるコンパイル済みCore MLモデルの
大きさには上限があり、コンパイル済みモデル1つごとに判定されます:
macOS 26までは約2 GiB、macOS 27以降は1 GiB(1,073,741,824バイト)です。
上限を超えるモデルは拒否されるわけではありません。変換・ロード・推論は
すべて成功し、システムのログにも何も出ませんが、全演算がCPUで実行され
ます。1 GiBという値はM2 Mac miniのmacOS 27.0.1で計測したものです。
Appleはこの上限を文書化していないため、ほかの機種では異なる可能性や、
今後また変わる可能性があります。

**eeANEの対処。** 1.7.0以降、コンパイル済みモデルが`eeane compile`を
実行しているmacOSのバージョンの上限を超える場合、`eeane compile`は
変換したモデルを、それぞれが上限に収まる部分へ分割します(coremltools
の`bisect_model`を使います。まだ大きすぎる部分はさらに分割し、最大16個
まで分けます)。分割したモデルは、1つの`<stem>.mlmodelc`の代わりに、
`segments.json`と`part01.mlmodelc`、`part02.mlmodelc`、…を収めた
ディレクトリ`<stem>.parts/`として保存されます。`eeane serve`は各部分を
まとめてロードし、順に実行します。API・設定・結果は、分割していない
モデルと変わりません。重みはすべてFP16の精度のままで、分割しても数値は
変わりません: 同じ重みをNeural Engine上で分割あり・なしで実行した出力は、
ビット単位で一致しました。

| モデル | コンパイル後の大きさ | macOS 27で分割しない場合 | macOS 27でeeANE 1.7.0を使う場合 |
|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B`、`Qwen/Qwen3-Reranker-0.6B` | 1.11 GiB | CPU実行になる: 結果は正しいが、128トークンで約3.5倍遅い | 2つに分割 |
| XLM-RoBERTa large級の7モデル(下記) | 1.04〜1.06 GiB | CPU実行になる: 出力がNaNになり、セルフチェックが不合格になる | 2つに分割 |
| そのほかの検証済みモデル(既定のruri-v3の2モデルを含む) | 1 GiB未満 | 影響なし | これまでと全く同じようにコンパイル |

**対処: eeANE 1.7.0以降へ上げて、コンパイルし直す。** 1 GiBを超える
モデルごとに、そのモデルを配信するMacで`eeane compile`をもう一度実行
してください:

```
eeane compile Qwen/Qwen3-Embedding-0.6B
eeane compile intfloat/multilingual-e5-large
```

オプションは要りません。`--force`も不要です: 古いeeANEで作った成果物は
自動的に作り直され、現在のシステムの上限を超える成果物(分割済みの
成果物では、上限を超える部分を含むもの)も同様に作り直されます。
したがって、**MacをmacOS 26から27へ上げた後は、以前そのMacで
コンパイルしたモデルごとに`eeane compile <id>`をもう一度実行して
ください**。(macOS 27で分割した成果物はmacOS 26でも動作します。各部分が
両方の上限を下回るためです。)`eeane compile`の出力には、使う上限と
分割の結果が表示されます:

```
NE size limit   : 1 GiB (from the macOS version)
...
s128: the compiled model (1.111 GiB) is above the Neural Engine size limit (1 GiB); splitting it into parts that each fit
s128: split into 2 parts (0.557 GiB, 0.553 GiB) in 7.5 s
```

**実測結果。** macOS 27.0.1、eeANE 1.7.0(M2 Mac mini、16 GB)での計測
です。配置率は全部分を通してNeural Engineに配置された演算の割合、
「CPUのみとの比」は同じ成果物をCPUのみで実行した場合との比です。

| モデル | バケツ | Neural Engine配置率 | ウォーム時のレイテンシ | CPUのみとの比 |
|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | 128 | 99.4% | 20.1 ms | 3.5倍 |
| Qwen/Qwen3-Embedding-0.6B | 512 | 99.4% | 104.4 ms | 2.6倍 |
| Qwen/Qwen3-Embedding-0.6B | 1024 | 99.4% | 292.9 ms | 2.4倍 |
| Qwen/Qwen3-Reranker-0.6B | 512 | 99.3% | 104.2 ms | 2.5倍 |
| Qwen/Qwen3-Reranker-0.6B | 1024 | 99.3% | 292.0 ms | 2.3倍 |

配置率と速度は、これらのモデルが分割なしでmacOS 26で示していた値
(99.3〜99.4%、128トークンで18.76 ms)に近く、部分の境目1つあたりの
コストは1 msを大きく下回ります(同じモデルを4つに分割すると128トークン
で20.9 ms、2つでは20.1 ms)。内蔵のセルフチェックでは、embeddingモデル
のcosineは0.99994です。参照実装との比較では、embeddingモデルの行ごとの
cosineは0.9998以上、rerankerのスコアの差はロジット空間で最大0.21です
(合否の閾値は0.3)。9件のクエリすべてで最上位の文書は同じで、参照
スコアの差が0.2を超える567組の文書の対では、順序の入れ替わりは
ありませんでした。

XLM-RoBERTa large級のモデルは、いずれも2つに分割されます。表の値は
バケツごとのウォーム時のレイテンシで、「—」はそのバケツがないことを
表します:

| モデル | Neural Engine配置率 | 128 | 512 | 1024 |
|---|---|---|---|---|
| intfloat/multilingual-e5-large | 96.7% | 12.2 ms | 70.3 ms | — |
| intfloat/multilingual-e5-large-instruct | 96.7% | 11.6 ms | 70.0 ms | — |
| BAAI/bge-m3 | 97.8% | 11.6 ms | 73.9 ms | 220.8 ms |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 97.8% | 11.6 ms | 74.1 ms | 222.0 ms |
| BAAI/bge-reranker-v2-m3 | 97.8% | — | 73.8 ms | 220.9 ms |
| BAAI/bge-reranker-large | 97.8% | — | 70.4 ms | — |
| hotchpotch/japanese-bge-reranker-v2-m3-v1 | 97.8% | — | 74.1 ms | 220.8 ms |

全バケツがセルフチェックに合格し、出力は有限値です。配置率と速度は、
これらのモデルがmacOS 26で示していた値(96.7〜97.8%、128 / 512トークン
で約11.6 / 71.5 ms)と同じ水準です。8言語の短文と短いペアを使った
参照実装との比較では、embeddingのcosineは0.99998以上、rerankerの確率の
差は0.0014以下で、最上位の文書はすべて一致しました。長い日本語の文章
では、multilingual-e5-large-instructのcosineは0.9987まで下がり、
bge-reranker-largeの確率は最大0.024ずれます。これはFP16・Neural Engine
上でのこれらのモデルの精度によるもので、分割の影響ではありません。
同じ重みなら、分割してもしなくても出力はビット単位で一致します。

**上限の上書き。** `eeane compile --ne-size-limit SIZE`は、macOSの
バージョンから決まる上限を置き換えます。SIZEは`1GiB`、`2GiB`、
`800MiB`またはバイト数で、`1GB`のような10進の単位はあえて受け付け
ません。別のバージョンのmacOSで配信するMac向けにコンパイルするときに
使います — たとえば、macOS 27で動くMac向けにmacOS 26上でコンパイル
する場合です:

```
eeane compile Qwen/Qwen3-Embedding-0.6B --ne-size-limit 1GiB
```

`--ne-size-limit none`は分割を無効にします: 配信するMacの上限を超える
モデルは1つのコンパイル済みモデルのままとなり、CPUで実行されます。

**サーバーが知らせます。** `eeane serve`は起動時に、設定された全モデルを
確かめ、Neural Engineで実行されないモデルについて警告をログに出します。
コンパイルした時点でNeural Engineに配置された演算の割合が低かった場合と、
実行中のmacOSの上限を超えている場合のどちらも対象です。分割済みの
モデルは部分ごとに判定されます:

```
WARNING eeane.server: model 'Qwen/Qwen3-Embedding-0.6B': the compiled model for bucket(s) [128] will not run on the Neural Engine (only 0.0% of its operations were placed on the Neural Engine when it was compiled; it is 1.11 GiB, above the 1 GiB the Neural Engine accepts on macOS 27 and later); it will run on the CPU instead, which is slower and, for some models, gives invalid results. Run `eeane compile Qwen/Qwen3-Embedding-0.6B` with this version of eeANE to rebuild it: a model above the size limit is split into parts that each fit, but a rebuild does not raise a low placement for every model. See the README section "Large models and the Neural Engine size limit"
```

この例は、macOS 27上で`--ne-size-limit none`を付けてコンパイルした
モデルのものです。分割済みのモデルで大きすぎる部分がある場合、理由は
`N of its M compiled parts are above the 1 GiB the Neural Engine
accepts on macOS 27 and later (the largest is X GiB)`となります。
サーバーはそのまま起動します。兆候はこの警告だけです。上限を超える
モデルはコンパイルし直せば解消しますが、配置率の低さは、作り直しても
すべてのモデルで改善するわけではありません。

**互換性。** eeANE 1.7.0が書き出すキャッシュは、`model_info.json`の
形式4を使います。eeANE 1.6.1以前はこの記録を受け付けず、eeANEの更新を
求めるエラーを出します(分割したモデルを配信できないためです)。
eeANE 1.7.0は、それ以前のバージョンが書き出したキャッシュも引き続き
読み込めます。1.7.0のmacOS 26(2 GiB)での挙動はテストでのみ確認した
もので、macOS 26の実機では検証していません。

## インストール

eeANEは[PyPI](https://pypi.org/project/eeane/)で公開されています。
`[compile]`エクストラはtorch/transformersを追加しますが、これらが
必要になるのは`eeane compile`だけです。下記の統合インストールを
使えば、モデルのコンパイルとサービングの両方を1つの環境で行えます。

### uv (推奨)

```sh
uv tool install "eeane[compile]"
```

後でアップグレードする場合は`uv tool upgrade eeane`を実行してください。

### pipx

```sh
pipx install --python python3.12 "eeane[compile]"
```

pipxの既定Pythonインタプリタは3.13以降であることがあり、eeANEは
まだこれに対応していません。そのため`--python`にはマシン上で利用
可能なPython 3.11または3.12の実行ファイル名(例: `python3.11`)、
あるいはそのフルパスを指定してください。

### pip + venv

```sh
python3.12 -m venv eeane-env
eeane-env/bin/pip install "eeane[compile]"
```

利用可能な対応インタプリタが`python3.11`である場合はそちらに読み
替えてください。

### 軽量インストール(サーバーのみ)

`[compile]`エクストラはtorchとtransformersを導入しますが、サーバー
本体はこれらを一切importしません — 必要になるのはモデルをCore ML
成果物に変換する`eeane compile`を実行するときだけです。サーバーと
一緒にインストールしておいてもディスク容量(数GB)を消費するだけで、
サーバーのメモリ使用量や動作には影響しないため、上記の統合インス
トールを既定とするのが妥当です。それでも常設環境をeeANEのランタイム
依存5つだけに保ちたい場合は、エクストラなしでeeANEをインストールし、
`eeane compile`は使い捨ての環境から実行してください:

```sh
uv tool install eeane
uvx --from "eeane[compile]" eeane compile <model>
```

### GitHubからのインストール

開発中のスナップショットをインストールしたい場合や、リポジトリの
特定リビジョンを固定したい場合は、PyPIの代わりにgit URLからインス
トールします — 上記のいずれのツールでも、たとえば:

```sh
uv tool install "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@main"
```

`@main`は最新の開発版を追跡し、`@v1.0.0`(または他の任意のリリース
タグ)はリリース済みのリビジョンを固定します。既存のインストールを
切り替えるには、同じコマンドに`--force`を付けて実行してください。

### eeANE 1.xからのアップグレード

eeANE 2.0は、モデルのコンパイルにtransformers 4.57ではなく5.20を使い
ます。eeANE 1.xでコンパイルしたモデルは、2.0のサーバーでそのまま使え
ます(サーバーはtransformersを使いません)。そのモデルに対して
`eeane compile`をもう一度実行すると、コンパイル済みモデルに記録された
依存の版が変わっているため、自動的に作り直されます(`--force`は不要)。
transformers 4.57でコンパイルし続けたい場合は、eeANE 1.7.xに留めてく
ださい(例: `uv tool install "eeane[compile]>=1.7,<2"`)。

## クイックスタート

```sh
# Hugging FaceのモデルID(自動ダウンロード)、またはHF配布形式の
# ローカルディレクトリから直接モデルをコンパイルします。初回のみ
# 必要で、成果物は ~/.cache/eeane/ 配下に生成され、1バケツあたり
# 約30〜100秒です:
eeane compile cl-nagoya/ruri-v3-310m
eeane compile cl-nagoya/ruri-v3-reranker-310m
eeane compile intfloat/multilingual-e5-base

# 各実行の最後に、そのまま使える[[models]]のTOMLスニペットが標準
# 出力に表示されます。このスニペットは最小限(通常はモデルidのみ)
# です。残りはサーバーがコンパイル済みモデルのキャッシュから自動
# 解決するためです。スニペットを ./eeane.toml に貼り付けたら
# (eeane.example.toml参照)、サーバーを起動します:
eeane serve
```

続けて、別のシェルから:

```sh
curl -s http://127.0.0.1:7997/health

curl -s http://127.0.0.1:7997/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "intfloat/multilingual-e5-base", "input": "hello eeANE"}'
```

### `eeane compile`について

`eeane compile`はモデルの`config.json`からバックエンドを自動選択
します。対応するアーキテクチャ系統は5つです: **ModernBERT**と
**XLM-RoBERTa**(embeddingとクロスエンコーダrerankerの両方に対応)、
**BERT**(embeddingモデルのみに対応 — BERT系クロスエンコーダ
rerankerは代わりに明確なエラーで拒否されます。理由は、コンパイル
済みグラフではsegment idをゼロに固定せざるを得ず、それがこの
アーキテクチャにおけるquery/documentペアの意味を変えてしまうため
です)、そして**Qwen3**(デコーダ型: 系列の最終トークンをプーリング
するembeddingモデルと、分類ヘッドの代わりに2つの語彙ロジットから
query/documentペアのスコアを出す生成型reranker — 詳細は後述)、
そして**Gemma 3**(双方向のembeddingモデルのみ — 詳細は後述)です。
`RobertaModel`アーキテクチャのモデルもXLM-RoBERTaバックエン
ドで扱われます — transformers上ではRoBERTaとXLM-RoBERTaは語彙のみ
が異なる同一のエンコーダ実装だからです。さらなる系統の追加を計画中
です。embeddingモデルでは、いずれの
バックエンドも、モデルディレクトリのsentence-transformers宣
言(`1_Pooling/config.json`)からプーリング方式を検出し、対応す
るグラフをコンパイルします — 3つのエンコーダ系バックエンドはmean/
CLSプーリング、Gemma 3はmeanプーリング、Qwen3のみlast-tokenプーリングです。last-tokenプーリ
ングは因果(左から右への)アーキテクチャを前提とするため、エンコー
ダ系バックエンドはこれを受け付けません。対応するプーリングモードを
宣言していないembeddingモデルは、推測でコンパイルするのではなく
エラーで拒否されます — 誤ったプーリングでコンパイルされた成果物は、
一見もっともらしいのに意味の違うベクトルを返してしまうためです。
rerankerはこの宣言と無関係です — スコアリングはモデル自身の分類
ヘッド(Qwen3の生成型rerankerの場合は自身の指示文テンプレート —
詳細は後述)の一部であり、独立したsentence-transformersモジュール
ではありません。コンパイルログには、embeddingモデルごとに検出された
プーリングが表示されます。実際に一通り動作を検証したモデルは
[検証済みモデル](#検証済みモデル)を参照してください。

sentence-transformers形式のモデルディレクトリは、`modules.json`で
**Dense**モジュールの列を宣言していることがあります。`eeane
compile`は`Transformer → Pooling → Dense(0個以上) →
Normalize(任意・末尾)`という構成に対応しており、宣言されたDense
線形射影(活性化はIdentityまたはTanh)を1つずつコンパイル済みグラフ
へ焼き込み、出力次元もそれに追随させます — Denseモジュールを持つ
モデルの埋め込み次元は、生のhidden sizeとは異なるものになります。
この構成で表現できないモジュール列は、重みを読み込む前にエラーで
拒否されます。意味の違う成果物を黙って作らないためです。Normalize
モジュール自体はグラフには含めません — 代わりにサーバーが、各モデ
ルの`normalize`設定(既定で有効。[API](#api)を参照)に応じてL2正規化
を行います。Denseの重みも、本体の重みと同じsafetensors優先ポリシー
に従います(`.bin`のみのDenseチェックポイントには`--allow-pickle`
が必要。[チェックポイント形式](#チェックポイント形式)を参照)。

コンパイラは、ModernBERT・XLM-RoBERTa・BERTの3つのエンコーダ系
バックエンドについては、`config.json`のアーキテクチャ名の末尾から
モデルがembeddingモデルかrerankerかを判別します(`ForSequence
Classification`で終わればreranker、`...Model`で終わればembedding)。
Qwen3のembeddingと生成型rerankerのチェックポイントは、どちらも
同じアーキテクチャ名`Qwen3ForCausalLM`で配布されるため、`ForCausal
LM`で終わるアーキテクチャ名は別の方法で解決します: `eeane compile`
はモデルディレクトリのsentence-transformersモジュール宣言
(`modules.json`)を読み、プーリングモジュールがあればembedding、
スコアリングモジュールがあればrerankerと判断します。どちらも宣言
していないディレクトリは、`--kind`の明示を案内するエラーで拒否
されます。他のアーキテクチャの判別方法は変わりません。いずれの
場合も、既定のバケツは
embeddingが128/512/1024、rerankerが512/1024で、モデルの最大系列長
にクリップされます — 最大512トークンのモデルはembeddingなら
128/512、rerankerなら512のみとしてコンパイルされ、外したバケツは
コンパイルのログに表示されます。`--buckets 512,2048`のようにカスタムの集合を
コンパイルすることもできます(S2048はM2実機で約518ms/推論を検証
済み)。再実行時は最新の成果物をスキップします(`--force`で再変換)。
変換のたびに**セルフチェック**が走り、FP32オリジナルに対する精度を
検証し、どれだけの演算がNeural Engineに配置されたかを計測し、
ウォームレイテンシを記録します — 表示されるサマリは互換性レポート
を兼ねるため、未検証のハードウェア(M1/M3/M4など)でeeANEを実行した
場合は、ぜひIssueに貼り付けてください。精度のセルフチェックは、
英語・日本語・中国語の3つの固定入力セットで評価され、いずれか1つ
のセットが閾値を満たせば合格となります: モデルのトークナイザが
ほとんど語彙を持たない言語の入力は、fp16とfp32の差が増幅されて
しまい、モデル自身の実力ではなく入力そのものを反映してしまうため、
モデルが実際に読める言語で採点することが公平な判定に必要だから
です。コンパイルログにはセットごとの実測値と、どのセットが採用され
たかが表示され、セットごとの数値は採用セットとともに成果物メタ
データにも記録されます。バケツごとの実測値はキャッ
シュ内のキャリブレーション記録(`model_info.json`)に集約され、
セルフチェックに失敗したバケツは、キャッシュ自動解決の設定がロード
する推奨集合から除外されます。トークナイザは成果物ディレクトリへ
凍結され、元のトークナイズを完全に再現することが検証されるため、
サーバーは実行時に元のモデルファイルもtransformersライブラリも
必要としません([docs/dependency-policy.md](docs/dependency-policy.md)
を参照)。

Qwen3の生成型reranker系統は分類ヘッドを持ちません: `eeane compile`
はクエリと文書を固定のチャット形式の指示文テンプレートに埋め込み、
コンパイル済みグラフは分類スコアの代わりに1つの値 — 系列の最終
位置における語彙"yes"のロジットから"no"のロジットを引いた値 — を
出力します。このテンプレートの組み立て方(前置き・ペアごとの本文・
末尾の後置き)は、コンパイル時にモデル自身の宣言から一度だけ解決
され、コンパイル済み成果物に記録されます。サーバーはそれを読み
戻し、リクエストごとに適用します。クエリ/文書のペアがコンパイル済み
バケツに対して長すぎる場合、切り詰められるのは本文だけです —
前置きと後置きは常にそのまま残ります。後置きにはテンプレートが
モデルへ回答を書かせる位置が含まれるためです。この単一の出力値は、
2択の恒等式`softmax([no, yes])[yes] = sigmoid(yes − no)`の引数その
ものであるため、意味を変えることなく`/rerank`の既存のsigmoidスコア
リングにそのまま組み込めます。`raw_scores=true`は、他のreranker
アーキテクチャで生のロジットを返すのと同じように、この生のロジット
差を返します。APIそのものは一切変わりません — 呼び出し側は、サービ
ング中のrerankerが生成型かどうかを意識する必要がありません。

Gemma 3バックエンドは、`google/embeddinggemma-300m`のような双方向の
embeddingモデルをコンパイルします(meanプーリングの後に2段のDense
射影を行い、出力は768次元。L2正規化はサーバーが行います)。生成用
(因果的)のGemma 3、マルチモーダルのGemma 3、Gemma 3nは対象外で、
`eeane compile`は理由を示すエラーで拒否します。Gemma 3の活性値は
そのままではfloat16に収まらないため、コンパイラは変換対象のモデルの
コピーにだけ、関数としては同値な書き換えを行います。詳しくは
[実測の精度と速度(EmbeddingGemma)](#実測の精度と速度embeddinggemma)
を参照してください。

### 検証済みモデル

以下のモデルはいずれも、Hugging Faceの配布形式そのままからコンパイル
し、セルフチェックに通過し、参照実装(`sentence-transformers` /
`CrossEncoder`)の出力とM2実機で突き合わせたものです。**バケツ**は
`--buckets`を指定しない場合に`eeane compile`が生成する構成(モデルの
最大系列長でクリップした後)です。分類は`config.json`が選択する
バックエンドごとで、モデル名からは必ずしも判別できない点に注意して
ください(`paraphrase-multilingual-mpnet-base-v2`はXLM-RoBERTa系、
`multilingual-e5-small`はBERT系です)。

いずれもeeANE 2.0.0(transformers 5.20)でコンパイルし直してセルフ
チェックに通過しています。各系統の参照モデルは、参照実装との突き合わせ
に加えて、eeANE 1.7が生成するコンパイル済みモデルとも出力を比べました。

これら5系統のアーキテクチャに基づく他のモデルも動作する可能性が高く、
以下は実際に一通り動作を確認したものの一覧にすぎません。

**ModernBERT**

| モデル | 種別 | バケツ |
|---|---|---|
| cl-nagoya/ruri-v3-30m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-70m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-130m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-310m | embedding | 128/512/1024 |
| hotchpotch/bekko-embedding-v1-a25m | embedding | 128/512/1024 |
| Alibaba-NLP/gte-modernbert-base | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-small-english-r2 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-english-r2 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-97m-multilingual-r2 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-311m-multilingual-r2 | embedding | 128/512/1024 |
| nomic-ai/modernbert-embed-base | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-reranker-310m | reranker | 512/1024 |
| hotchpotch/japanese-reranker-tiny-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-xsmall-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-small-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-base-v2 | reranker | 512/1024 |
| ibm-granite/granite-embedding-reranker-english-r2 | reranker | 512/1024 |

**XLM-RoBERTa**

素のRoBERTaアーキテクチャに基づくモデルもこのバックエンドで扱われ
ます(transformers上ではRoBERTaも同一のエンコーダ実装で、語彙のみ
が異なるため)。下表のgranite-embedding r1英語系モデル
(`config.json`で`RobertaModel`)がBERTの表ではなくこの表に載って
いるのはそのためです。

| モデル | 種別 | バケツ |
|---|---|---|
| BAAI/bge-m3 <sup>1</sup> | embedding | 128/512/1024 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-107m-multilingual | embedding | 128/512 |
| ibm-granite/granite-embedding-278m-multilingual | embedding | 128/512 |
| ibm-granite/granite-embedding-30m-english | embedding | 128/512 |
| ibm-granite/granite-embedding-125m-english | embedding | 128/512 |
| intfloat/multilingual-e5-base | embedding | 128/512 |
| intfloat/multilingual-e5-large | embedding | 128/512 |
| intfloat/multilingual-e5-large-instruct | embedding | 128/512 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | embedding | 128/512 |
| BAAI/bge-reranker-v2-m3 | reranker | 512/1024 |
| BAAI/bge-reranker-base | reranker | 512 |
| BAAI/bge-reranker-large | reranker | 512 |
| hotchpotch/japanese-reranker-cross-encoder-xsmall-v1 | reranker | 512 |
| hotchpotch/japanese-reranker-cross-encoder-small-v1 | reranker | 512 |
| hotchpotch/japanese-bge-reranker-v2-m3-v1 | reranker | 512/1024 |

この表のlarge級の7モデル(bge-m3、snowflake-arctic-embed-l-v2.0、
multilingual-e5-large、multilingual-e5-large-instruct、
bge-reranker-v2-m3、bge-reranker-large、japanese-bge-reranker-v2-m3-v1)
はコンパイル後に1 GiBを超えるため、macOS 27以降では`eeane compile`が
それぞれを2つに分割します。この形でmacOS 27.0.1上のeeANE 1.7.0で検証
しており、それ以前にはmacOS 26でも検証しています。
[大きいモデルとNeural Engineの上限](#大きいモデルとneural-engineの上限)を
参照してください。

**BERT**(embeddingモデルのみ)

| モデル | 種別 | バケツ |
|---|---|---|
| BAAI/bge-small-en | embedding | 128/512 |
| BAAI/bge-small-en-v1.5 | embedding | 128/512 |
| BAAI/bge-base-en-v1.5 | embedding | 128/512 |
| BAAI/bge-large-en-v1.5 | embedding | 128/512 |
| BAAI/bge-small-zh-v1.5 | embedding | 128/512 |
| BAAI/bge-base-zh-v1.5 <sup>1</sup> | embedding | 128/512 |
| BAAI/bge-large-zh-v1.5 <sup>1</sup> | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-xs | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-s | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-m | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-m-v1.5 | embedding | 128/512 |
| Snowflake/snowflake-arctic-embed-l | embedding | 128/512 |
| intfloat/e5-small-v2 | embedding | 128/512 |
| intfloat/e5-base-v2 | embedding | 128/512 |
| intfloat/e5-large-v2 | embedding | 128/512 |
| intfloat/multilingual-e5-small | embedding | 128/512 |
| thenlper/gte-base | embedding | 128/512 |
| thenlper/gte-large | embedding | 128/512 |
| mixedbread-ai/mxbai-embed-large-v1 | embedding | 128/512 |
| sentence-transformers/all-MiniLM-L6-v2 | embedding | 128/512 |
| sentence-transformers/all-MiniLM-L12-v2 | embedding | 128/512 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | embedding | 128/512 |
| sentence-transformers/paraphrase-MiniLM-L6-v2 | embedding | 128/512 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | embedding | 128/512 |
| MongoDB/mdbr-leaf-mt | embedding | 128/512 |
| MongoDB/mdbr-leaf-ir | embedding | 128/512 |
| sentence-transformers/LaBSE | embedding | 128/512 |

<sup>1</sup> `pytorch_model.bin`のみを配布しているため、コンパイルには
`--allow-pickle`が必要です([チェックポイント形式](#チェックポイント形式)
を参照)。

**Qwen3(デコーダ型)**

生成型rerankerのスコアは、他のrerankerと同じ意味を持ちます(上記の
「`eeane compile`について」を参照) — コンパイル後は`/rerank`から
区別なく扱われます。

| モデル | 種別 | バケツ |
|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | embedding | 128/512/1024 |
| Qwen/Qwen3-Reranker-0.6B | reranker | 512/1024 |

この2モデルは、macOS 26.6.2と、eeANE 1.7.0でコンパイル(2つに分割)
したうえでmacOS 27.0.1で検証したものです。macOS 27以降では、eeANE
1.7.0以降でコンパイルしてください。
[大きいモデルとNeural Engineの上限](#大きいモデルとneural-engineの上限)を
参照してください。

**Gemma 3(双方向のembeddingモデルのみ)**

| モデル | 種別 | バケツ |
|---|---|---|
| google/embeddinggemma-300m <sup>2</sup> | embedding | 128/512/1024 |

<sup>2</sup> ダウンロードするには、Hugging Face上でモデルの利用条件に
同意する必要があります(トラブルシューティングを参照)。eeANEは利用者
の手元でモデルを変換するだけで、モデルの重みは配布しません。変換・
利用にあたっては、Gemmaの規約を含め、各モデルのライセンス・利用規約
に利用者自身が従う必要があります。

### 実測の精度と速度(EmbeddingGemma)

`google/embeddinggemma-300m`をM2 Mac mini(16GB、macOS 27.0.1)の
Neural Engine上でFP16実行して計測したものです。組み込みのセルフ
チェックは既定のすべてのバケツで合格しており、その実測値は次の表の
とおりです。

| バケツ | FP32基準とのcosine(en / ja / zh) | NE配置率 | ウォーム時レイテンシ(中央値) |
|---|---|---|---|
| 128 | 0.99936 / 0.99934 / 0.99914 | 99.3% | 5.05 ms |
| 512 | 0.99936 / 0.99934 / 0.99926 | 99.2% | 24.98 ms |
| 1024 | 0.99936 / 0.99934 / 0.99926 | 99.2% | 68.94 ms |

**元モデルとの一致度。** 基準は、sentence-transformers 5.7.0の
`encode`をCPU・FP32で1件ずつ実行したものです。各行は、コンパイル済み
モデルをNeural Engineで実行した結果との、入力ごとのcosineです。

| 条件 | 件数 | 系列長バケツ(トークン数) | cosine min | cosine mean |
|---|---|---|---|---|
| 前置文字列なし(英・日・中の短文) | 9 | 128(11〜37) | 0.99914 | 0.99938 |
| クエリ(`prompt_name="query"`対クライアント側で同じ文字列を連結) | 9 | 128(18〜45) | 0.99930 | 0.99945 |
| 文書(`prompt_name="document"`対クライアント側連結) | 9 | 128(17〜44) | 0.99941 | 0.99957 |
| 日本語の段落 | 60 | 512(94〜512) | 0.99841 | 0.99867 |
| 長文 | 12 | 512(302〜483) | 0.99845 | 0.99867 |
| 長文 | 12 | 1024(604〜961) | 0.99821 | 0.99843 |
| `dimensions` 512 / 256 / 128 | 各9 | 128 | 0.99920 / 0.99932 / 0.99944 | — |

同じ変換手順をFP32のまま(PyTorch上で)実行した結果は、基準とcosine
0.9999998以上で一致します。つまり上の差は変換手順ではなく、FP16と
Neural Engineの演算精度に由来します。クエリと文書の行は、前置文字列を
クライアント側で付けた結果が、モデル自身の`prompt_name`と同じに
なることを示しています。

**速度。** 同じ入力を1件ずつ(バッチ1)、ウォームアップ3回の後20回の
中央値を、同じMac mini M2上で直列に測定しました。eeANEはトークナイズ
+推論+正規化の全体の時間で、PyTorchはsentence-transformersの
`encode`(FP32、torch 2.7.1、CPUスレッド4)です。

| 系列長 | eeANE(Neural Engine) | PyTorch CPU | PyTorch MPS | 対CPU | 対MPS |
|---|---|---|---|---|---|
| 128 | 7.56 ms(16,925 tokens/s) | 67.63 ms(1,893) | 82.47 ms(1,552) | 8.9倍 | 10.9倍 |
| 512 | 27.27 ms(18,774) | 189.58 ms(2,701) | 136.32 ms(3,756) | 7.0倍 | 5.0倍 |
| 1024 | 70.40 ms(14,545) | 510.24 ms(2,007) | 246.21 ms(4,159) | 7.2倍 | 3.5倍 |

**FP16の扱い。** 上流のモデルカードは、このモデルの活性値がfloat16に
収まらないと述べています。`eeane compile`は、変換対象のモデルの
コピーにだけ、関数としては同値な書き換えを、フラグなしで常に行い
ます:

1. 残差ストリームを2のべき乗で割ります。後半の層で値がfloat16の
   上限を超えるためです。
2. 層ごとにMLPの出力を2のべき乗で持ち上げます。後半の層で出力が
   小さくなりすぎ、Neural Engineで精度が落ちるためです。
3. 系列長1024以上では、attentionをqueryの範囲に分けて計算します。
   1ヘッドのスコア行列が2^20要素に達すると、Neural Engine上でだけ
   結果が壊れるためです。

係数は変換時にモデルを短文で実測して決め、成果物のメタデータに記録
されます。

### チェックポイント形式

`eeane compile`は既定でsafetensors形式のチェックポイントのみを受け
付けます。`pytorch_model.bin`のみを配布し`.safetensors`ファイルを
持たないHugging Faceリポジトリやローカルモデルディレクトリは、
Hubからのダウンロードでもローカルディレクトリでも、明確なエラーで
拒否されます。`--allow-pickle`を指定すると、代わりにpickleベースの
`.bin`重みへオプトインできます: この場合`eeane compile`は
transformersに`torch.load(weights_only=True)`での読み込みを強制し、
WARNINGログを出力します。`weights_only=True`はpickleファイル読み込み
のリスクを緩和しますが、完全には排除しません — 過去にバイパスが
発見されたことがあり(例: CVE-2026-24747、torch 2.10.0で修正)、
`eeane compile`が依存するtorchのバージョンは、Core ML変換ツール
チェーンとの互換性のため、その修正より前のリリースに固定されて
います([docs/dependency-policy.md](docs/dependency-policy.md)を
参照)。`--allow-pickle`は、信頼できる配布元のチェックポイントに
限って使用してください。リポジトリにsafetensorsが存在する場合、
`--allow-pickle`を付けても挙動は変わりません: safetensorsが常に
優先され、`.bin`ファイルはダウンロードされません。たとえば
BAAI/bge-m3は`pytorch_model.bin`のみを配布しているため、コンパイル
にはこのフラグが必要です:

```sh
eeane compile BAAI/bge-m3 --allow-pickle
```

## 設定

サーバーは組み込みのデフォルト設定でそのまま動作します。変更する
には、[`eeane.example.toml`](eeane.example.toml)を`./eeane.toml`
(または`~/.config/eeane/eeane.toml`)にコピーして編集してください。
設定ファイルは次の順序で探索されます: `--config PATH` >
`./eeane.toml` > `~/.config/eeane/eeane.toml` > 組み込みデフォルト。
`--host`/`--port`/`--log-level`のCLIフラグと環境変数
`EEANE_API_KEY`は設定ファイルより優先されます。

```sh
eeane serve --config /path/to/eeane.toml
eeane serve --host 192.168.1.20 --port 7997

# 設定ファイルを検証し、解決後の有効な設定を表示します
# (APIキーの値は表示されません)。サーバーは起動しません:
eeane check-config --config /path/to/eeane.toml
```

設定ファイルにはサービングするモデルを列挙します — embedding・
rerankerとも任意の数のエントリを書けます。`[[models]]`エントリは
通常`id = "..."`だけで足ります: kind・凍結済みトークナイザ・バケツ
ごとの成果物・埋め込み次元は、コンパイル済みモデルのキャッシュ
(`server.cache_root`、既定`~/.cache/eeane/`)から、キャリブレーション
が推奨するバケツに従って自動解決されます。`kind`・`tokenizer`・
`[models.artifacts]`を明示的に書く方式も引き続き有効で、その場合は
キャッシュとは独立にエントリが固定されます。各kindの中では最初に
列挙したエントリが既定モデルとなり、`model`を指定しないリクエスト
で使われます。rerankerエントリはembedding専用サーバーの場合は丸ごと
省略できます(その場合`/rerank`は503を返します)。`python -m
eeane.server`と`python -m eeane <サブコマンド>`は、それぞれ
`eeane serve`と`eeane`コマンドの後方互換エイリアスとして引き続き
利用できます(開発環境ではどちらも先頭に`uv run`を付けます)。

## サービングと運用

### モデルのロード

`[[models]]`エントリの既定`load_policy`は`"on_demand"`です(既定値
は`[server] default_load_policy`で変更できます。設定方法は
`eeane.example.toml`を参照): サーバーは起動時にどのモデルもロード
せず、最初にそのモデルを必要とするリクエストが来た時点でロードし
ます。モデルの成果物が一度ロードされていれば、以降のロードは1秒
未満で完了します(M2 Macでの実測は0.3〜0.8秒)。例外は`eeane compile`
が成果物を生成した直後にその成果物を初めてロードするときで、macOS
がそのモデル用のNeural Engineキャッシュを構築するため数十秒かかる
ことがあります — これは一度きりのコストで、サーバーを再起動した
後であっても、以降のロードでは再び発生しません。この待ち時間は、
そのロードを誘発したリクエストの応答時間に含まれます。

on_demandモデルは、`keep_alive`秒(`[server] keep_alive`、既定300、
モデルごとに上書き可能。`0`はアイドルになり次第アンロード)の間
どのリクエストにも応答しなかった場合、自動的にアンロードされ、次に
そのモデルを必要とするリクエストで再ロードされます。エントリに
`load_policy = "resident"`を設定すると、起動時にロードされ、
サーバーの実行中ずっとメモリに保持されます。`load_policy =
"disabled"`を設定すると、エントリを設定ファイルに残したまま
サービング対象から外せます: `GET /models`と`GET /health`には現れ
ず、そのidを指定したリクエストは404になります。

`[server] max_loaded_models`は、同時にメモリへ保持できるモデル数の
上限です(未設定なら無制限)。この上限を超えてモデルをロードする
必要が生じた場合は、最も長くアイドルな`on_demand`モデルから追い
出してメモリを確保します。`resident`モデルと現在リクエストを処理中
のモデルはこの方法で追い出されることはないため、`resident`エントリ
だけで上限を超える設定は起動時に拒否されます。

### embeddingリクエスト向けバッチ2成果物

embeddingモデル(reranker以外)は任意で、バケツごとに2件目の成果物
を追加コンパイルできます。これは1回のNeural Engine呼び出しに2件の
入力をまとめて詰め込む成果物です: 通常のバッチ1コンパイルに加えて
`eeane compile <model> --buckets <S> --batch 2`を実行します。
サービングするかどうかは任意選択で、`[[models]]`エントリの
`[models.batch_artifacts]`(`[models.artifacts]`と同じ形の、バケツ
→成果物パスのテーブル)で有効化します。1リクエスト内で2件以上の
入力が同じバケツにルーティングされた場合、それらはペアにまとめら
れ、1件ずつではなくバッチ2成果物で推論されます。これはM2 Mac上の
ベンチマークで、短い入力を多数含むリクエストのスループットを約
25%向上させました。id-onlyエントリは、バッチ2成果物がコンパイル
済みであれば`batch_artifacts`をコンパイル済みモデルのキャッシュ
から自動解決します。明示形式(`[models.artifacts]`を明示するもの)
では`[models.batch_artifacts]`も明示する必要があり、単独では設定
できません。バッチ2成果物を持たない構成の動作は従来と完全に同一
です。

### リクエストの受付・キューイング・シャットダウン

`server.max_pending_requests`は、サーバーが同時に受け付ける推論
リクエスト数の上限です。現在実行中のリクエストと順番待ち中の
リクエストの両方を数えます(既定500、`0`は無制限)。上限に達した後
に到着したリクエストは、`Retry-After`ヘッダ付きの
`429 Too Many Requests`で即座に拒否されます。

`server.queue_timeout`は、受け付けられたリクエストが実際に推論を
開始するまで待ってよい時間の上限です(既定600秒、`0`でタイムアウト
無効)。この上限を超えて待ったリクエストは、`Retry-After`ヘッダ付
きの`503 Service Unavailable`で打ち切られます。推論を開始した後の
リクエストは、どれだけ時間がかかってもこのタイムアウトによって
中断されることはありません。いずれの場合も`Retry-After`は、クライ
アントに何秒待って再試行すべきかを伝えます。

`server.coalesce_requests`(既定`true`)は、処理中のリクエストと
内容が同一(同一モデル・同一入力)の新規リクエストが届いた場合、
それを既存のリクエストへ併合します: 推論を2回実行する代わりに、
2件目のリクエストは1件目に相乗りし、完了時に同じ結果を受け取り
ます。

`server.graceful_shutdown_timeout`は、SIGTERMやCtrl-Cを受信した
ときに、実行中(in-flight)のリクエストの完了をサーバーが待つ時間の
上限です(既定は未設定で、その場合はどれだけ時間がかかっても全件
の完了まで待ちます。待っている間は新規接続を受け付けません)。秒数
を指定すると、その待ち時間に上限をかけられます。

### localhost外への公開

非loopbackアドレスへのbind(`--host`または`server.host`)は、
サーバーをあなたのネットワークへ公開します。APIキーを設定して
ください — 設定ファイルの`api_key`(`chmod 600`にしておくこと)、
または環境変数`EEANE_API_KEY`です — そうすると`GET /health`を除く
全エンドポイントが`Authorization: Bearer <key>`ヘッダを要求する
ようになります。APIキーなしで非loopbackアドレスをサービングする
と、サーバーは警告ログを出します。`/health`は監視用途のため常に
開放されており、代わりにレート制限がかかります
(`server.health_rate_limit`、既定60リクエスト/分/クライアントIP、
`0`で無効)。これらはあくまでアプリケーション層の保護策です: 信頼
できるLAN/VPNの外へ公開する場合は、サーバーをリバースプロキシや
ファイアウォールの背後に置いてください。

### サービスとして起動する

ログイン時にサーバーを自動起動し、稼働させ続けるには、macOSの
launchd agentとして登録します — 手順とそのまま使えるplistテンプ
レートは[docs/launchd.md](docs/launchd.md)を参照してください。
オンデマンドロードのおかげで、常駐させたeeANE agentはアイドル中
ほぼコストがかかりません。

## API

- `GET /health` — ステータスと、サービス中モデルごとのエントリ
  (`id`・`kind`・サービス中のバケツ・`loaded`)。無認証・レート
  制限あり
- `GET /models`(エイリアス: `GET /v1/models`) — サービング中の
  全モデルのOpenAI互換リスト
- `POST /v1/embeddings`(エイリアス: `POST /embeddings`) —
  OpenAI互換(`input`は文字列またはリスト、`encoding_format`は
  `float`/`base64`、`dimensions`は省略可)。埋め込みは既定でL2正規化
  されます(モデルごとの`normalize`設定)
- `POST /rerank`、`POST /v1/rerank` — Infinity互換
  (`query`/`documents`/`top_n`/`return_documents`/`raw_scores`)

embeddingsとrerankリクエストの`model`フィールド(省略可能)は、
設定されたidでサービング対象のモデルを選択します。省略した場合は、
そのエンドポイントのkindにおいて設定に最初に列挙されたモデルが
選ばれます。未知のidを指定すると、サービング可能なid一覧付きの
404が返り、別のkindのモデルを指定すると400が返ります。embeddings
とrerankのエンドポイントは`/v1`配下とルート直下の両方でサービング
されるため、base URLは`/v1`サフィックスの有無どちらでも動作します。
各入力はそのモデルの、収まる最小の系列長バケツにルーティングされ、
それより長い場合は最大バケツに切り詰められ、サーバー側で警告が
出ます。

embeddingsリクエストの`dimensions`フィールド(省略可能、OpenAI互換)
を指定すると、返す埋め込みを先頭`dimensions`個の成分に切り詰めてか
ら(正規化が有効な場合は)再正規化し、推論をやり直すことなくより
小さいベクトルを返します。Matryoshka表現学習(MRL)で学習された
モデルのように、任意の接頭辞長でも意味を持つ埋め込みを生成する
モデルで有効です。モデルの埋め込み次元を超える値を指定すると400が
返ります。

eeANEはリクエストのテキストを一切加工しません — `input`と`query`は
そのままトークナイザに渡ります。一部のembeddingモデルは、検索クエリ
だけに(文書には付けない)指示文を前置して使うことを前提に学習されて
います。サービング対象のモデルがそのように学習されている場合、その
文字列を付けるのは呼び出し側の責任です。たとえばQwen3-Embedding-0.6B
は次の書式を公開しています:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{検索クエリ本文}
```

EmbeddingGemmaも同様の設計で、文書側にも前置文字列を付けます。検索
用途では、検索クエリの本文の前に`task: search result | query: `を、
文書の本文の前に`title: none | text: `を付けます(ほかの用途の書式は
モデルカードを参照)。

[Open WebUI](https://github.com/open-webui/open-webui)(v0.6.0以降)
からは、環境変数`RAG_EMBEDDING_QUERY_PREFIX`にこの指示文を設定すれば
クエリ側にだけ付与され、`RAG_EMBEDDING_CONTENT_PREFIX`(文書側)は
空のままにしておけます。この方式は、`sentence-transformers`が同じ
モデルのプロンプト機能を使って同じ埋め込みを作る場合と比較検証済み
です: 両者はコサイン0.9999で一致します。
EmbeddingGemmaでは、`RAG_EMBEDDING_QUERY_PREFIX`に
`task: search result | query: `を、`RAG_EMBEDDING_CONTENT_PREFIX`に
`title: none | text: `を設定します(クライアント側で連結した結果が
モデル自身のプロンプトと一致することは、「実測の精度と速度」の
クエリ・文書の行で確認しています)。

[Open WebUI](https://github.com/open-webui/open-webui)からeeANEを
使うには: 埋め込みエンジンをOpenAIに設定し、base URLを
`http://127.0.0.1:7997/v1`にします。rerankingエンジンはExternalに
設定し、URLを`http://127.0.0.1:7997/rerank`にします。APIキーを
設定した場合は、それをOpenAI APIキー欄 / External reranker APIキー
欄に入力してください — Open WebUIはeeANEが期待する`Authorization`
ヘッダとしてそれを送信します。

## パフォーマンス

以下の数値はM2 Mac mini(macOS 13以降、16GB)で計測したものです。
ベースラインには、同じモデルをPyTorch(sentence-transformers)が
MPS GPUでサービングした場合を用いています:

- **埋め込みスループット**: ANE上で最大約13,600実効(パディング
  除外後)トークン/秒 — MPSベースラインの2〜3倍。消費電力は同程度
  ながら、トークンあたりのエネルギー効率は2.6〜3.8倍で、GPUは完全
  に空いたままです。
- **Reranking**: HTTP経由の36文書rerankはチャンク長に応じて約
  2.0〜5.6秒で完了し、同じモデルをMPSベースでサービングした場合の
  同一リクエストと比べて約2〜8倍高速です。
- **メモリ**: 310Mクラスのembeddingモデル1つとreranker 1つを常駐
  させたサーバーは約750MBに収まります。コンパイル済みの重みの大
  部分はPythonプロセスの外に存在し、on_demandエントリはアイドル
  時にメモリを解放します。
- **ロード時間**: macOSがコンパイル済み成果物をキャッシュした後
  は、モデル1つあたり約0.2〜0.8秒です(コンパイル直後の最初の1回
  だけは数十秒かかります)。

HTTP経由のレスポンスは、Core ML直接推論と完全に一致することを
検証しています(リポジトリのチェックアウトにある
`tools/verify_server.py`)。

## トラブルシューティング

- **`404 model not found`**: クライアントが送る`model`フィールド
  は、サービング中モデルの設定`id`と完全に一致している必要があり
  ます。eeANEが実際にサービングしているidは`GET /models`で確認し
  てください。旧バージョンのeeANEから移行したクライアントは、
  以前は`model`フィールドが完全に無視されていたため、任意の値
  (または未指定)を指定したリクエストが成功していた点に注意して
  ください — その寛容な挙動はなくなりました。
- **`500 ... produced a non-finite output ...`**: 下記の既知の
  制限を参照してください。これはモデルがNeural Engine以外で実行
  されたことを意味します。リクエストを処理しているマシンでNeural
  Engineが利用可能か確認してください。
  macOS 27以降では、分割されていない1 GiB超のモデル(1.7.0より前の
  eeANEや古いmacOSでコンパイルしたもの、または`--ne-size-limit none`
  を付けてコンパイルしたもの)が主な原因です。`eeane compile <id>`で
  コンパイルし直してください。
  [大きいモデルとNeural Engineの上限](#大きいモデルとneural-engineの上限)を
  参照してください。
- **起動時に、モデルが"will not run on the Neural Engine"という警告が
  出る**、またはmacOSを上げた後にモデルが数倍遅くなった: 現在の
  システム上で`eeane compile <id>`を実行してコンパイルし直して
  ください。1.7.0以降、上限を超えるモデルはそれぞれが上限に収まる
  部分へ分割されます。`--ne-size-limit none`を付けてコンパイルした
  モデルは分割されず、上限を超えていればCPUで実行されます。
  [大きいモデルとNeural Engineの上限](#大きいモデルとneural-engineの上限)を
  参照してください。
- **リクエストがまれに通常よりずっと時間がかかる**: モデルが
  `keep_alive`を超えてアイドルだった後の最初のリクエストは、オン
  デマンド再ロードのコストを負担します(通常は1秒未満)。また、
  `eeane compile`直後の最初のリクエストは、一度きりのNeural
  Engineキャッシュ構築のコストを負担します(数十秒)。どちらも
  想定内の挙動です。1秒未満の再ロードすら避けたい場合は
  `load_policy = "resident"`を使ってください。
- **`eeane compile`が`no .safetensors weights are available ...`で
  失敗する**: そのモデルはpickleベースの`pytorch_model.bin`
  チェックポイントのみを配布しており、`eeane compile`は既定で
  safetensorsを要求します。エラーメッセージが対処法(`--allow-pickle`
  を付ける。上記のチェックポイント形式を参照)をすでに示しています —
  信頼できる配布元のチェックポイントに限って使用してください。
- **`eeane compile`が`... must declare its pooling in the
  sentence-transformers '1_Pooling/config.json' ...`で失敗する**:
  embeddingモデルのコンパイルには、sentence-transformersのプーリ
  ング宣言`1_Pooling/config.json`が必要です。これが無い・読み取
  れない・対応外のモード(mean/CLS以外)を宣言している場合、
  `eeane compile`は推測せずエラーで停止します。そのモデルが実際
  にsentence-transformers形式で配布されているか確認してください
  — ローカルディレクトリを指定している場合は、配布元にある
  `1_Pooling/config.json`が揃っているか確認してください。eeANEは
  推測でmeanプーリングを補う挙動には戻しません。誤ったプーリング
  でコンパイルされた成果物は、黙って意味の違うベクトルを返してし
  まうためです。
- **`eeane compile`が`cannot tell whether ... is an embedding model or
  a reranker ...`で失敗する**: これは`ForCausalLM`アーキテクチャの
  モデルディレクトリ(例: Qwen3)が、sentence-transformersのプーリ
  ングモジュールもスコアリングモジュールも`modules.json`に宣言して
  いない場合にのみ発生します — このアーキテクチャでembeddingと生成
  型rerankerのチェックポイントを見分けるために`eeane compile`が
  頼れる唯一の手がかりだからです。`--kind embedding`または
  `--kind reranker`を明示して指定すれば、そのままコンパイルできます。
- **利用承諾の要るモデル(例: `google/embeddinggemma-300m`)を
  `eeane compile`が取得できない**: そのモデルのHugging Faceのページで、
  ファイルをダウンロードする前に利用条件への同意が求められます。
  モデルのページで同意し、`hf auth login`でログインしてから、もう
  一度`eeane compile`を実行してください。eeANEはトークンを自分では
  扱わず、Hugging Faceのライブラリに任せています。未同意・未ログイン
  で取得に失敗したときは、エラーに対処を示す1行が付きます。

## 既知の制限

- **ANE専用**: eeANEはApple Neural Engineでの実行を対象としてお
  り、コンパイル済みモデルをCPUのみの計算経路で実行することは
  サポートしていません。Neural Engineがコンパイル済みモデルから
  実際には利用できないマシンや構成では、推論結果が非有限値
  (NaN/Inf)になることがあります — これはeeANEが対応する全アーキ
  テクチャで観測されています。サーバーはそのような結果を黙って
  返す代わりに、推論時に非有限値の出力を検出し、`500 Internal
  Server Error`で応答します。このエラーが出た場合は、あなたの
  環境で実際にはNeural Engineが使われていないことを示す強い
  シグナルです。
- **検証済みハードウェア**: 公開しているすべての計測値と検証は
  M2 Macで実施したものです。他のApple Siliconの世代(M1/M3/M4
  など)でも動作すると見込まれますが、メンテナによる検証は行われ
  ていません。`eeane compile`が表示するセルフチェックのサマリが
  互換性レポートを兼ねているのはまさにこのためです。他のマシン
  からの報告は、成功・失敗を問わずGitHub issueで大歓迎です。
- **長い文書**: 各入力は、そのモデルのコンパイル済み最大バケツに
  切り詰められます(必要であれば`eeane compile --buckets`でより
  大きなバケツを追加してください)。rerankerには、それを超える
  文書に対するスライディングウィンドウ処理はありません。
- **BAAI/bge-m3はdense出力のみ**: eeANEはbge-m3のdense embedding出力
  のみをコンパイル・サービングします。同モデルが持つ別のsparse表現
  やmulti-vector(ColBERTスタイル)表現は別の重みファイルであり、
  `eeane compile`はこれらを取得も公開もしません。
- **BERT系クロスエンコーダrerankerは非対応**: 上記の「`eeane
  compile`について」を参照してください — コンパイル済みグラフの
  segment idをゼロに固定せざるを得ず、それがこのアーキテクチャに
  おけるquery/documentペアの意味を変えてしまうためです。BERT系
  embeddingモデルはこの影響を受けず、対応しています。
- **語彙外の入力に対する精度**: コンパイル済みモデルはfp16で動作
  します。モデルのトークナイザがうまく表現できない入力 — たとえば
  中国語専用モデルに英語の文を与えるような場合 — では、入力自体が
  モデルの学習範囲から大きく外れているため、fp32の参照実装との
  丸め差が目に見えて大きくなります。`eeane compile`自身のセルフ
  チェックは、英語・日本語・中国語の3つの固定セットで採点し、
  モデルが実際に語彙を持つセットで合否判定することでこの問題を
  回避しています(上記の「`eeane compile`について」を参照)が、
  自分でモデルの想定言語外の入力に対してfp16とfp32の出力を比較する
  際には、同じ注意が依然として当てはまります。モデルが想定する言語
  の範囲内では一致度はずっと高くなります(上記一覧の全モデルで
  コサイン0.9999以上。ただしEmbeddingGemmaを除きます。その実測値は
  「実測の精度と速度」にあります)。
- **メモリはロードしたバケツの数に応じて増えるが、バケツごとに
  複製はされない**: コンパイル済みの重みはメモリマップされるため、
  同じモデルの複数バケツをサービングしても、常駐メモリがバケツ数
  倍に増えることはありません。0.6Bクラスのembeddingモデルで実測
  したところ、3バケツすべてをロードしてもサーバープロセスの物理
  フットプリントの増分は約150MBでした(未ロード時313MB →
  128/512/1024すべてロード後456MB)。8モデルを並べた構成でも
  ワーカーは592MB(ピーク730MB)に収まりました。
- **Neural Engineが受け付けるモデルサイズの上限**: コンパイル済み
  モデルが一定の大きさを超えると、Neural Engineには一切配置されず、
  全演算がCPUへ落ちます — 変換・ロード・推論自体は成功します。
  この上限はmacOS 26までは約2 GiB、**macOS 27以降は1 GiB**です。
  1.7.0以降の`eeane compile`は、上限を超えるモデルをそれぞれが上限に
  収まる部分へ分割します。古いeeANEや古いmacOSでコンパイルしたモデルは
  コンパイルし直す必要があり、`--ne-size-limit none`を付けると、上限を
  超えるモデルも分割されずCPUで実行されます。
  [大きいモデルとNeural Engineの上限](#大きいモデルとneural-engineの上限)を
  参照してください。
  Qwen3系統の4B・8Bモデルは2 GiBも大きく超えており、引き続き対応して
  いません。分割で試したことはなく、16 GBのマシンで変換・配信に要する
  メモリも確かめていません。
- **EmbeddingGemmaのバケツ2048**: このモデルの上限長は2048で、
  `--buckets 2048`は変換できますが、手元の計測ではそのバケツは
  Neural Engineに配置されずCPU実行になります(配置率0%、約3.3秒/件)。
  既定の128/512/1024を推奨します。

## 開発

eeANE自体の開発を行う場合、または以下のリポジトリ限定のツールを
使う場合は、リポジトリをcloneし、パッケージをインストールする
代わりにチェックアウトから`uv run`でコマンドを実行します:

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync --extra compile   # torch/transformersはコンパイル時のみ必要
uv run eeane compile cl-nagoya/ruri-v3-310m
uv run eeane serve
```

`uv run eeane <サブコマンド>`は、上記で説明した`compile`/`serve`/
`check-config`と同じサブコマンドを、インストール済みパッケージで
はなくチェックアウトから実行します。

稼働中のサーバーをエンドツーエンドで検証する(Core ML直接推論との
精度比較・API互換性・レイテンシ)、および1ステップでコードベースを
lint・テストする、いずれもリポジトリのチェックアウトを前提とし
ます:

```sh
uv run python tools/verify_server.py all
# 特定のサービング中モデルをCore ML直接推論と突き合わせて確認する場合:
uv run python tools/verify_server.py verify-embedding --model intfloat/multilingual-e5-base
uv run python tools/verify_server.py verify-rerank --model BAAI/bge-reranker-v2-m3
./tools/check.sh   # ruff lint + フォーマットチェック + pytest を1ステップで実行
```

### PoCを試す(歴史的な開発スナップショット)

`poc/`配下のスクリプトは、v0.1〜v0.3の研究記録として凍結された
ものです。サポートされている変換手段は上記の`eeane compile`です。
これらのスクリプトはベンチマーク研究のために引き続き実行できます:

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync

# HF配布形式のモデルを models/ruri-v3-310m と
# models/ruri-v3-reranker-310m に配置してから(例:
# `huggingface-cli download cl-nagoya/ruri-v3-310m`で取得)、
# 以下を実行します:

# 埋め込みモデル (v0.1)
uv run python poc/convert_embedding.py --seq-len 512   # HF -> .mlmodelc
uv run python poc/verify_accuracy.py --seq-len 512     # FP32基準との精度比較
uv run python poc/benchmark_latency.py --seq-len 512 --compute-units CPU_AND_NE --compute-plan

# rerankerモデル (v0.2)
uv run python poc/convert_reranker.py --seq-len 512
uv run python poc/verify_reranker_accuracy.py --seq-len 512
uv run python poc/benchmark_latency.py --model reranker --seq-len 512 --compute-units CPU_AND_NE --compute-plan

# 性能計測 (v0.3)
uv run python poc/run_sweep.py --seq-lens 128,512 --batches 1,2      # S×Bレイテンシマトリクス
uv run python poc/benchmark_throughput.py --model embedding --chunk-tokens 128 --batch 2
uv run python poc/benchmark_mps.py --model embedding --chunk-tokens 512 --batch 32  # GPUベースライン
```

### デコーダ型モデルの調査を試す

デコーダ型(causal language model)アーキテクチャは、現在では通常の
`eeane compile`で扱えます — 上記の「`eeane compile`について」および
「検証済みモデル」のQwen3の項を参照してください。`poc_qwen/`配下の
スクリプトは、その対応の判断のもとになった研究記録としてリポジトリ
に残してあり、引き続き実行できます。この記録のうち一部分だけは本体
には対応物がありません: サイズを変えた合成モデル(大きなモデルの
ダウンロードは発生しません)を使って、Neural Engineが受け付けなく
なるのはどのくらいの大きさからかを探る部分です(その結果は上記の
「既知の制限」を参照してください):

```sh
# 変換(モデルは初回実行時にHubから取得されます)
uv run python poc_qwen/convert_embedding.py --seq-len 128

# FP32基準およびsentence-transformersとの精度比較
uv run python poc_qwen/verify_accuracy.py \
    --mlmodelc models/compiled/qwen3-embedding-0.6b/s128_b1_fp16_macos13.mlmodelc --seq-len 128

# レイテンシと、各演算がどの計算ユニットに配置されたか
uv run python poc_qwen/benchmark_latency.py \
    --mlmodelc models/compiled/qwen3-embedding-0.6b/s128_b1_fp16_macos13.mlmodelc \
    --seq-len 128 --compute-plan

# Neural Engineへの配置が崩れる大きさを、サイズを変えた合成モデルで
# 調べます(大きなモデルのダウンロードは発生しません)
uv run python poc_qwen/size_sweep.py

# 分類ヘッドの代わりに最終位置のyes/noロジットでスコアを出す、
# 生成型のrerankerモデル
uv run python poc_qwen/convert_reranker.py --seq-len 256
```

## 謝辞と関連プロジェクト

eeANEは[Infinity](https://github.com/michaelfeil/infinity)に触発
されて生まれました。Infinityは、自己ホスト型でAPI互換の埋め込み・
rerankingサーバーがいかに便利であり得るかを示したオープンソースの
サービングエンジンです — eeANEの`/rerank` APIは、クライアントが
URLを変更するだけで両者を切り替えられるよう、意図的にInfinityの
スキーマに従っています。

eeANEは作者がInfinityを使ってGPU上でModernBERTベースのembedding
モデルを使用できた体験があったからこそ開発されました。両プロジェ
クトは競合するのではなく、互いに補完し合う関係にあります: eeANE
はApple Silicon MacのApple Neural Engine上でのみモデルを実行し、
対応するモデルアーキテクチャの集合も意図的に絞り込んでいます。
LinuxやWindows上で、あるいはNVIDIA/AMDのGPUやCPU上でembeddingや
rerankingモデルをサービングしたい場合、またはもっと幅広いモデル
カタログが必要な場合は、是非Infinityを使ってください。

## 変更履歴

| バージョン | ハイライト |
|---|---|
| 2.0.0 | `eeane compile`がtransformers 4.57から5.20へ移行(huggingface_hubは2系)。eeANE 1.xでコンパイルしたモデルは2.0のサーバーでそのまま使え、`eeane compile`を再実行すると自動的に作り直される(transformers 4.57でコンパイルし続けたい場合はeeANE 1.7.xに留める)。ModernBERT・BERT・XLM-RoBERTaのバックエンドはアテンションマスクをコンパイル済みグラフの中で作るようになり、これらのモデルはNeural Engineで実行される割合もわずかに上がった。検証済みの63モデルをすべてコンパイルし直して再検証。一部のXLM-RoBERTa系モデルの凍結トークナイザは、SentencePieceモデルと同じく末尾の空白をトークンにしなくなった |
| 1.7.0 | Neural Engineの大きさの上限を超えるコンパイル済みモデルを、それぞれが上限に収まる部分へ自動的に分割し、`eeane serve`が順に実行するように(API・設定・結果は変わらず、重みはすべてFP16のまま)。上限は`eeane compile`を実行するmacOSのバージョンで決まり(macOS 26までは2 GiB、macOS 27以降は1 GiB)、`--ne-size-limit`で上書きできる。XLM-RoBERTa large級の7モデルがmacOS 27で使えるように。Qwen3モデルの埋め込み表を8ビット整数で保存する必要がなくなり、この回避策は廃止。起動時の警告は、分割済みのモデルを部分ごとに判定するように。`model_info.json`は形式4になった(eeANE 1.6.1以前では読み込めない) |
| 1.6.1 | macOS 27(Neural Engineが受け付けるモデルの大きさが2 GiBから1 GiBへ下がった)で、Qwen3系のモデルが再びNeural Engineで動くようになった: macOS 27以降でコンパイルしたQwen3モデルが上限を超える場合、`eeane compile`が語彙の埋め込み表を8ビット整数で保存し、コンパイル済みモデルが上限に収まる(配置率・速度・精度はmacOS 26と同水準)。`eeane serve`は、Neural Engineで実行されないモデルを起動時に警告する。XLM-RoBERTa large級のモデルは、当面macOS 27では使えないことを明記 |
| 1.6.0 | Gemma 3(双方向のembeddingモデル)がeeANEの5つ目の対応アーキテクチャ系統になり、通常の`eeane compile`でコンパイルして既存のサーバーでそのまま提供できるように。生成用・マルチモーダル・Gemma 3nは理由つきで拒否。関数としては同値な書き換えでモデルをFP16の範囲に収める。検証済みモデルを1件追加(google/embeddinggemma-300m、62→63)。利用承諾の要るモデルの取得に失敗したとき、対処を案内するように |
| 1.5.0 | Qwen3(デコーダ型)がeeANEの4つ目の対応アーキテクチャ系統になり、通常の`eeane compile`/`eeane serve`の流れでコンパイル・サービングできるように。embeddingモデルにlast-tokenプーリング、分類ヘッドの代わりにyes/noロジット対からスコアを出す生成型rerankerを追加。`ForCausalLM`アーキテクチャのモデル種別判定は、アーキテクチャ名だけでなくモデルのsentence-transformersモジュール宣言も読むように変更。検証済みモデルを2件追加(Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B) |
| 1.4.5 | `poc_qwen/`を追加。デコーダ型(causal LM)の埋め込みモデルがNeural Engineで動作するか、また、どのくらいの大きさまで受け付けられるかを調べる調査用スクリプト群。エンジンの変更なし |
| 1.4.0 | コンパイルのセルフチェックが、英語・日本語・中国語の3つの固定言語セットで評価し、いずれか1セットが閾値を満たせば合格とするようになった(モデルが実際に語彙を持つ言語で判定するため)。sentence-transformersのDenseモジュール(`Transformer → Pooling → Dense → Normalize`)に対応。`RobertaModel`アーキテクチャのモデルがXLM-RoBERTaバックエンドで動作するように。OpenAI互換の`dimensions`パラメータを`/v1/embeddings`に追加。検証済みモデルを9件追加(51→60) |
| 1.3.0 | ModernBERTバックエンドが、meanのみのコンパイルから、モデルのsentence-transformers宣言に基づくmean/CLSプーリングの自動判別に対応し、CLSプーリングを宣言するModernBERT系embeddingモデル(granite-embedding-*-r2系など)も正しくコンパイルできるようになった。判別したプーリングはコンパイルログと成果物メタデータに記録される。検証済みモデルを5件追加(gte-modernbert-base、granite-embedding-*-r2系4件) |
| 1.2.0 | 3つのバックエンド全体で35モデルを追加検証(granite・Snowflake Arctic Embed・GTE・mxbai・MiniLM・e5・ruri-v3の小型・中国語版bge v1.5・日本語reranker)し、検証済みモデルの一覧表を新設。エンジンの変更なし |
| 1.1.0 | BERT embeddingバックエンドを新設。BAAI/bge系モデルを6件追加検証(bge-m3・bge-reranker-base/large・bge-small/base/large-en-v1.5)。pickleベースのチェックポイント向けopt-inの`--allow-pickle`。PyPIメタデータの拡充 |
| 1.0.0 | 最初の安定版リリース: PyPIで公開、launchdサービス化ガイド、ドキュメント全面改訂 |
| 0.10.0 | uv/pipx/pipでGitHubから直接インストール可能に、`eeane`コンソールコマンドを追加 |
| 0.9.0 | 受付制御(429/503 + `Retry-After`)、同一リクエストの併合、グレースフルシャットダウン、非有限出力ガード、opt-inのバッチ2成果物 |
| 0.8.0 | オンデマンドロード、アイドルアンロード(`keep_alive`)、`max_loaded_models`による追い出し |
| 0.7.0 | 複数アーキテクチャ対応バックエンド(XLM-RoBERTaがModernBERTに加わる)、マルチモデルサービングとルーティング、マシン別キャリブレーションによるキャッシュ自動解決 |
| 0.6.0 | `eeane compile`: HFのID/ローカルディレクトリ→セルフチェックと凍結トークナイザ付きのCore ML成果物への変換。torch不要のサーバーランタイム |
| 0.5.0 | TOML設定+CLI、APIキー認証、`GET /models`、`/health`のレート制限、CI |
| 0.4.0 | 最初のHTTPサーバー: OpenAI互換embeddings、Infinity互換rerank |
| 0.1.0〜0.3.0 | 概念実証: embeddingモデルとrerankerのANE変換・推論、精度検証、GPUとの性能比較 |

各リリースの詳細: [GitHub Releases](https://github.com/xhighhongo41/eeANE/releases)。

## ライセンス

GPL-3.0-or-later。[LICENSE](LICENSE)を参照してください。

`testdata/corpus/`配下のテストコーパスは、[青空文庫](https://www.aozora.gr.jp/)
由来のパブリックドメイン文学作品であり、GPLの対象外です。詳細は
`testdata/corpus/README.md`を参照してください。

---

The English README is [README.md](README.md).
