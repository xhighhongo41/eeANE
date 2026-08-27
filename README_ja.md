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
- **現時点で対応するアーキテクチャ系統は3つ**: ModernBERTと
  XLM-RoBERTa(いずれもembeddingモデルとクロスエンコーダ型reranker
  モデルの両方に対応)、そしてBERT(embeddingモデルのみに対応)。
  さらなる系統の追加を計画中です。

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
します。対応するアーキテクチャ系統は3つです: **ModernBERT**と
**XLM-RoBERTa**(embeddingとクロスエンコーダrerankerの両方に対応)、
そして**BERT**(embeddingモデルのみに対応 — BERT系クロスエンコーダ
rerankerは代わりに明確なエラーで拒否されます。理由は、コンパイル
済みグラフではsegment idをゼロに固定せざるを得ず、それがこの
アーキテクチャにおけるquery/documentペアの意味を変えてしまうため
です)。さらなる系統の追加を計画中です。embeddingモデルでは、BERTと
XLM-RoBERTaのバックエンドについてはモデルディレクトリが宣言する
mean/CLSプーリングが自動的に適用されます。ModernBERTバックエンドは
現在meanプーリングのみをコンパイルします。実際に一通り動作を検証
したモデルは[検証済みモデル](#検証済みモデル)を参照してください。
コンパイラは
モデルがembeddingモデルかrerankerかを自動判別し、既定のバケツは
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
場合は、ぜひIssueに貼り付けてください。バケツごとの実測値はキャッ
シュ内のキャリブレーション記録(`model_info.json`)に集約され、
セルフチェックに失敗したバケツは、キャッシュ自動解決の設定がロード
する推奨集合から除外されます。トークナイザは成果物ディレクトリへ
凍結され、元のトークナイズを完全に再現することが検証されるため、
サーバーは実行時に元のモデルファイルもtransformersライブラリも
必要としません([docs/dependency-policy.md](docs/dependency-policy.md)
を参照)。

### 検証済みモデル

以下のモデルはいずれも、Hugging Faceの配布形式そのままからコンパイル
し、セルフチェックに通過し、参照実装(`sentence-transformers` /
`CrossEncoder`)の出力とM2実機で突き合わせたものです。**バケツ**は
`--buckets`を指定しない場合に`eeane compile`が生成する構成(モデルの
最大系列長でクリップした後)です。分類は`config.json`が選択する
バックエンドごとで、モデル名からは必ずしも判別できない点に注意して
ください(`paraphrase-multilingual-mpnet-base-v2`はXLM-RoBERTa系、
`multilingual-e5-small`はBERT系です)。

これら3系統のアーキテクチャに基づく他のモデルも動作する可能性が高く、
以下は実際に一通り動作を確認したものの一覧にすぎません。

**ModernBERT**

| モデル | 種別 | バケツ |
|---|---|---|
| cl-nagoya/ruri-v3-30m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-70m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-130m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-310m | embedding | 128/512/1024 |
| hotchpotch/bekko-embedding-v1-a25m | embedding | 128/512/1024 |
| cl-nagoya/ruri-v3-reranker-310m | reranker | 512/1024 |
| hotchpotch/japanese-reranker-tiny-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-xsmall-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-small-v2 | reranker | 512/1024 |
| hotchpotch/japanese-reranker-base-v2 | reranker | 512/1024 |
| ibm-granite/granite-embedding-reranker-english-r2 | reranker | 512/1024 |

**XLM-RoBERTa**

| モデル | 種別 | バケツ |
|---|---|---|
| BAAI/bge-m3 <sup>1</sup> | embedding | 128/512/1024 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | embedding | 128/512/1024 |
| ibm-granite/granite-embedding-107m-multilingual | embedding | 128/512 |
| ibm-granite/granite-embedding-278m-multilingual | embedding | 128/512 |
| intfloat/multilingual-e5-base | embedding | 128/512 |
| intfloat/multilingual-e5-large | embedding | 128/512 |
| intfloat/multilingual-e5-large-instruct | embedding | 128/512 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | embedding | 128/512 |
| BAAI/bge-reranker-v2-m3 | reranker | 512/1024 |
| BAAI/bge-reranker-base | reranker | 512 |
| BAAI/bge-reranker-large | reranker | 512 |

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

<sup>1</sup> `pytorch_model.bin`のみを配布しているため、コンパイルには
`--allow-pickle`が必要です([チェックポイント形式](#チェックポイント形式)
を参照)。

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
  `float`/`base64`)。埋め込みは既定でL2正規化されます(モデル
  ごとの`normalize`設定)
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
- **CLSプーリングのModernBERT系embeddingモデルは未対応**:
  ModernBERTバックエンドはmeanプーリングのみをコンパイルするため、
  CLSプーリングを宣言する同系統のembeddingモデル(例: ibm-graniteの
  `granite-embedding-*-r2`のembeddingモデル)は誤った
  プーリングでコンパイルされてしまいます。対応を計画中です。
  BERTとXLM-RoBERTaのバックエンドは宣言されたプーリングを読み取り
  両方に対応しており、ModernBERT系の*reranker*も影響を受けません。
- **語彙外の入力に対する精度**: コンパイル済みモデルはfp16で動作
  します。モデルのトークナイザがうまく表現できない入力 — たとえば
  中国語専用モデルに英語の文を与えるような場合 — では、入力自体が
  モデルの学習範囲から大きく外れているため、fp32の参照実装との
  丸め差が目に見えて大きくなります。モデルが想定する言語の範囲内
  では一致度はずっと高くなります(上記一覧の全モデルでコサイン
  0.9999以上)。

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
