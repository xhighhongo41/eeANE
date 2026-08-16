# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANEは、Apple SiliconマックのApple Neural Engine (ANE)上でテキスト埋め込み
モデルとrerankingモデルを動かすためのエンジンです。モデルは
Hugging Faceの配布形式のまま取得し、ローカルでCore ML形式にコンパイルします。
コンパイル済みモデルは数秒でロードでき、ANE上で推論するため、GPUと
ユニファイドメモリの大部分を他の作業のために空けておけます。

> **開発状況: 初期開発中 (v0.10)。**
> v0.1〜v0.3で概念実証を完了:
> [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)
> (日本語ModernBERT埋め込みモデル)と
> [cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)
> (そのクロスエンコーダ型reranker)を共通パイプラインでCore MLに変換し、
> ANE推論・数値精度を検証のうえ、M2 Mac mini上で埋め込み最大約13,600
> 実効トークン/秒(MPS GPUベースラインの2〜3倍)を計測済みです。
> v0.4で最初のHTTPサーバーを実装: OpenAI互換の`/v1/embeddings`と
> Infinity互換の`/rerank`をANEから直接サービングし、レスポンスはCore ML
> 直接推論と完全一致します(HTTP経由の36文書rerankはチャンク長に応じて
> 約2.0〜5.6秒で、同一マシンのInfinity_emb/MPS構成の約3〜8倍速。常駐
> メモリ約750MB対6〜8GB)。**v0.5でサーバーが設定可能になりました**:
> TOML設定ファイル+`eeane serve` / `eeane check-config` CLI(bind
> アドレス・ポート・モデルとバケツ構成・ログレベル)、localhost外へ
> 公開するための任意のBearer APIキー認証、OpenAI互換`GET /models`、
> `/health`のレート制限を追加。**v0.6でモデル変換が製品機能になりました**:
> `eeane compile <model>`はローカルディレクトリまたはHugging FaceのモデルID
> (自動ダウンロード)を受け取り、バケツごとの`.mlmodelc`成果物を
> `~/.cache/eeane/`配下に生成します。トークナイザは成果物ディレクトリへ
> 「凍結」され(元のトークナイズとトークン単位で一致することを機械検証)、
> 変換後にはセルフチェック(FP32基準の精度・ANE配置率・ウォームレイテンシ)
> が走り、そのレポートはハードウェア互換性レポートを兼ねます。最後に
> 設定ファイルへそのまま貼れるスニペットを出力します。サーバー本体は
> torch/transformersに依存しなくなり、重量級の依存はオプションの
> `[compile]`エクストラへ分離されました。**v0.7で多アーキテクチャ・
> 多モデルの基盤ができました**: バックエンドインターフェースの確定に
> より、アーキテクチャの追加は「バックエンドモジュールを1つ書く」作業
> に定型化されました。最初の非ModernBERTバックエンド(XLM-RoBERTa)は
> [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)・
> [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)・
> [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
> でエンドツーエンド検証済みです(compile→serve→HTTP応答とCore ML直接
> 推論の完全一致、ANE配置率93〜98%)。サーバーは任意の数のembedding/
> rerankerモデルを同時サービングし、リクエストの`model`フィールドで
> ルーティングします。`eeane compile`はマシン別のキャリブレーション
> 実測を成果物キャッシュに記録し、設定の`[[models]]`エントリは
> `id = "..."`の1行だけで書けるようになりました(残りはキャッシュから
> 自動解決)。**v0.8でオンデマンドロードとアイドルアンロードを追加
> しました**: `[[models]]`エントリの既定`load_policy`が`on_demand`に
> なりました(`[server] default_load_policy`で変更可)。サーバーは
> 起動時にモデルをロードせず、最初にそのモデルを必要とするリクエスト
> の時点でロードします。成果物が一度ロードされていれば、以降のロード
> は1秒未満です(開発機実測0.3〜0.8秒程度)。例外はコンパイル直後の
> 成果物を初めてロードするときで、OSがANE向けキャッシュを生成する
> ため数十秒かかることがあります(サーバー再起動後も含め、2回目以降
> は高速に戻ります)。この待ち時間は最初にロードを誘発したリクエスト
> の応答時間に含まれます。on_demandモデルはアイドル時間が
> `keep_alive`(`[server] keep_alive`、既定300秒、モデル毎上書き可)
> を超えると自動アンロードされ、次のリクエストで再ロードされます。
> `load_policy = "resident"`を指定するとv0.8以前と同様に常時ロード
> のままになり、`load_policy = "disabled"`を指定するとエントリを
> 設定に残したままサービング対象から外せます(APIに現れず、当該id
> へのリクエストは404)。`[server] max_loaded_models`は同時にメモリへ
> 保持するモデル数の上限で、超過時は最も長くアイドルなon_demand
> モデルから追い出されます。`GET /health`は各モデルの`loaded`状態を
> 返すようになりました。**v0.9で高負荷時の堅牢性を強化しました**:
> 推論リクエストは受付制御を通るようになり、`[server]
> max_pending_requests`(既定500)を超えた分は即座に429と
> `Retry-After`ヘッダで拒否され、順番待ちが`[server] queue_timeout`
> (既定600秒)を超えたリクエストは待ち続ける代わりに503で応答され
> ます(推論が始まったリクエストは必ず完走します。シャットダウン
> 時も処理中のリクエストは完遂まで待たれ、`graceful_shutdown_timeout`
> で待ち時間の上限を設定できます)。同一内容のリクエストが処理中の
> ものと重なった場合は計算を1回にして結果を共有します
> (`coalesce_requests`、既定on。同一8リクエストの実測で約7倍高速)。
> モデル出力にNaN/無限大が含まれる場合(サポート外の計算経路で実行
> された兆候)は、壊れた数値を黙って返す代わりに明確な500エラーに
> なります。短い入力を多数含むembeddingリクエストは、バッチ2成果物
> を追加コンパイル(`eeane compile <model> --buckets <S> --batch 2`)
> すると約25%スループットが向上します(開発機実測。id-onlyエントリ
> では自動解決され、同一リクエスト内の同バケツ入力を2件ずつまとめて
> 推論します)。**v0.10でGitHubから直接インストールできるように
> なりました**: `uv`・`pipx`・`pip`のいずれからもこのリポジトリから
> 直接インストールでき(下記のインストール節を参照)、`eeane`
> コンソールコマンドが使えるようになりました(`python -m eeane`に
> 代わるものです)。

## 動作要件

- Apple Siliconマック (M1以降)
- macOS 13以降
- Python 3.11または3.12(3.13以降は未対応)。`uv`は対応するPythonを
  自動的に解決します。pipxやpip+venvでインストールする場合は、対応
  バージョンのPythonを自分で用意する必要があります。
- Xcodeコマンドラインツール (`xcode-select --install`) — `eeane compile`
  が`xcrun coremlcompiler`を使用します
- [uv](https://docs.astral.sh/uv/) — eeANEのインストールに推奨(下記
  インストール節を参照)。開発環境の構築にも必要です

## インストール

eeANEは以下のいずれかの方法で、このGitHubリポジトリから直接
インストールできます。

### uv (推奨)

以下の1コマンドで`[compile]`エクストラ(torch/transformers)も
含めてインストールされるため、同じ環境でモデルのコンパイルと
サーバー起動の両方ができます:

```sh
uv tool install "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@v0.10.0"
```

タグを指定する`@v0.10.0`(最新リリース)は再現性のあるインストールに、
`@main`は最新の開発版を追いたい場合に使います。アップグレードする
ときは、新しいタグを指定した上で`--force`を付けて同じコマンドを
実行し、既存のインストールを入れ替えてください。

### pipx

```sh
pipx install --python python3.12 "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@v0.10.0"
```

pipxの既定Pythonが3.13以降だとeeANEの動作要件を満たさないため、
`--python`にはマシン上にあるPython 3.11/3.12の実行ファイル名
(例: `python3.11`)またはフルパスを指定してください。

### pip + venv

```sh
python3.12 -m venv eeane-env
eeane-env/bin/pip install "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@v0.10.0"
```

対応するPythonが`python3.11`しかない場合はそちらに読み替えてください。

### 軽量インストール(サーバーのみ)

`[compile]`エクストラが導入するtorchとtransformersが必要になるのは
`eeane compile`(モデルをCore ML形式に変換するコマンド)を実行する
ときだけで、サーバー本体はこれらを一切importしません。常設環境に
含めておいてもディスクを数GB消費するだけでメモリ面への影響はない
ため、上記のuvによる一括インストール(compile込み)のままで実害は
ありません。それでも常設環境をeeANEのランタイム依存5つだけに保ち
たい場合は、`[compile]`なしでeeANEをインストールし、`eeane compile`
だけを使い捨ての一時環境から実行してください:

```sh
uv tool install "eeane @ git+https://github.com/xhighhongo41/eeANE@v0.10.0"
uvx --from "eeane[compile] @ git+https://github.com/xhighhongo41/eeANE@v0.10.0" eeane compile <model>
```

## モデルのコンパイルとサーバー起動

```sh
# Hugging FaceのモデルID(自動ダウンロード)またはHF配布形式のローカル
# ディレクトリから直接コンパイルします。初回のみ。成果物は
# ~/.cache/eeane/ 配下に生成され、1バケツあたり約30〜100秒です:
eeane compile cl-nagoya/ruri-v3-310m
eeane compile cl-nagoya/ruri-v3-reranker-310m
eeane compile intfloat/multilingual-e5-base

# 各実行の最後に[[models]]のTOMLスニペットが標準出力に表示されます。
# v0.7以降のスニペットは最小形(基本はモデルidのみ)です — 残りの情報は
# サーバーがコンパイル済みキャッシュから自動解決します。スニペットを
# ./eeane.toml に貼り付けて(eeane.example.toml参照)、サーバーを起動します:
eeane serve
```

`eeane compile`はモデルの`config.json`からバックエンドを自動選択します。
対応アーキテクチャは2系統: **ModernBERT**(cl-nagoya/ruri-v3-310mと
同rerankerで検証済み)と**XLM-RoBERTa**(intfloat/multilingual-e5-base・
intfloat/multilingual-e5-large・BAAI/bge-reranker-v2-m3で検証済み。
embeddingモデルはモデルディレクトリが宣言するmean/CLSプーリングを自動
適用)です。対応系統は1.0以降に順次拡充予定です。embedding/rerankerの
種別も自動判別します。バケツの既定は埋め込み128/512/1024、reranker
512/1024で、モデルの最大系列長へ自動クリップされます(最大512トークン
のmultilingual-e5系は128/512になります)。`--buckets 512,2048`のように
変更もできます(S2048はM2実機で約518ms/推論を検証済み)。再実行時は
最新の成果物をスキップします(`--force`で再変換)。変換後には
**セルフチェック**が走り、FP32基準の精度検証・Neural Engineへの配置率
計測・ウォームレイテンシ記録を行います。表示されるサマリは互換性
レポートを兼ねるので、未検証ハードウェア(M1/M3/M4など)で動かした際は
ぜひIssueに貼ってください。バケツ別の実測はキャッシュ内の
キャリブレーション記録(`model_info.json`)に集約され、セルフチェックに
失敗したバケツはキャッシュ自動解決の設定がロードする推奨集合から
除外されます。トークナイザは
成果物ディレクトリへ凍結され、元のトークナイズとの完全一致が機械検証
されるため、サーバー実行時には元のモデルファイルもtransformers
ライブラリも不要です([docs/dependency-policy.md](docs/dependency-policy.md)
参照)。

### 設定

サーバーは設定なしでも組み込みデフォルトで動作します。変更するには
[`eeane.example.toml`](eeane.example.toml)を`./eeane.toml`(または
`~/.config/eeane/eeane.toml`)にコピーして編集してください。設定ファイル
の探索順は `--config PATH` > `./eeane.toml` >
`~/.config/eeane/eeane.toml` > 組み込みデフォルト、です。CLIフラグ
(`--host`/`--port`/`--log-level`)と環境変数`EEANE_API_KEY`は設定
ファイルより優先されます。

```sh
eeane serve --config /path/to/eeane.toml
eeane serve --host 192.168.1.20 --port 7997

# 設定ファイルを検証し、解決後の有効設定を表示します (サーバーは起動
# しません。APIキーの値は表示されません):
eeane check-config --config /path/to/eeane.toml
```

設定ファイルにはサービングするモデルを列挙します — embedding/reranker
とも複数エントリを書けます。`[[models]]`エントリは通常`id = "..."`だけ
で足ります: kind・凍結済みトークナイザ・バケツごとの成果物・埋め込み
次元は、`eeane compile`が書いたキャッシュ(`server.cache_root`、既定
`~/.cache/eeane/`)から自動解決され、キャリブレーションの推奨バケツが
ロードされます。`kind`/`tokenizer`/`[models.artifacts]`を明示する
書き方(v0.7以前の形式)も引き続き有効で、キャッシュと独立にエントリを
固定できます。各kindの中では設定に最初に書いたエントリが既定モデルに
なり、`model`未指定のリクエストを処理します。rerankerエントリは省略
可能で、その場合はembedding専用サーバーになります(`/rerank`は503を
返します)。`python -m eeane.server`と`python -m eeane <サブコマンド>`は、
それぞれ`eeane serve`と`eeane`コマンドの後方互換エイリアスとして
引き続き使えます(開発環境では両方とも`uv run`を先頭に付けます)。

### バケツごとのバッチ2成果物 (embeddingリクエスト向け)

embeddingモデル(reranker除く)は任意で、バケツごとに2件目の成果物を
追加コンパイルできます。1回のNeural Engine呼び出しで2件の入力を
まとめて処理する成果物です:
`eeane compile <model> --buckets <S> --batch 2`を、通常のバッチ1
コンパイルと合わせて実行します。サービングは任意選択で、`[[models]]`
エントリの`[models.batch_artifacts]`(`[models.artifacts]`と同じ形の
バケツ→成果物パスのテーブル)で有効化します。1リクエスト内で2件以上
の入力が同じバケツにルーティングされた場合、1件ずつではなくバッチ2
成果物でまとめて2件ずつ推論され、短い入力を多数含むリクエストで手元
計測では約25%スループットが向上しました。id-onlyエントリは、バッチ2
成果物がキャッシュにコンパイル済みであれば`batch_artifacts`を自動
解決します。明示形式(`[models.artifacts]`を明示するもの)では
`[models.batch_artifacts]`も明示する必要があり、単独では設定できません。
バッチ2成果物が無い構成の動作は従来と完全に同一です。

### モデルのロード

v0.8から`[[models]]`エントリの既定`load_policy`は`"on_demand"`です
(既定値は`[server] default_load_policy`で変更可能。設定例は
`eeane.example.toml`参照)。サーバーは起動時に一切モデルをロードせず、
最初にそのモデルを必要とするリクエストの時点でロードします。一度
ロード済みの成果物であれば、次のロードは1秒未満です(開発機実測
0.3〜0.8秒程度)。例外は`eeane compile`が生成した直後の成果物を初めて
ロードするときで、OSがそのモデル用にANE向けキャッシュを生成するため
数十秒かかることがあります(サーバーを再起動しても、2回目以降は
高速です)。この待ち時間は、ロードを誘発したリクエストの応答時間に
含まれます。

on_demandモデルは、`keep_alive`秒(`[server] keep_alive`、既定300、
モデル毎に上書き可能。`0`はアイドルになり次第アンロード)の間リクエ
ストに応答しなかった場合に自動アンロードされ、次に必要とするリクエ
ストで再ロードされます。エントリに`load_policy = "resident"`を指定
すると起動時にロードされ、サーバーの実行中ずっとメモリに保持されます
(v0.8以前の既定動作)。`load_policy = "disabled"`を指定すると、その
エントリを設定ファイルに残したままサービング対象から外せます:
`GET /models`と`GET /health`には現れず、そのidを指定したリクエストは
404になります。

`[server] max_loaded_models`は、同時にメモリへ保持できるモデル数の
上限です(未設定なら無制限)。この上限を超えてモデルをロードする必要
があるときは、最も長くアイドルなon_demandモデルから追い出してメモリ
を確保します。residentモデルと現在リクエストを処理中のモデルは追い
出し対象になりません。したがって、residentエントリの数だけで上限を
超える設定は起動時にエラーになります。

### リクエストの受付・キューイング・シャットダウン

`server.max_pending_requests`は、サーバーが同時に受け付ける推論
リクエスト数の上限です(処理中+待機中の合計。既定500、`0`は無制限)。
上限に達した後に届いたリクエストは、`Retry-After`ヘッダ付きの
`429 Too Many Requests`で即座に拒否されます。

`server.queue_timeout`は、受理されたリクエストが推論開始まで待って
よい時間の上限です(既定600秒、`0`はタイムアウト無効)。この上限を
超えて待ったリクエストは、`Retry-After`ヘッダ付きの
`503 Service Unavailable`で打ち切られます。推論が開始した後の
リクエストは、どれだけ時間がかかってもこのタイムアウトで中断される
ことはありません。いずれの場合も`Retry-After`は、クライアントに
何秒後の再試行を推奨するかを伝えます。

`server.coalesce_requests`(既定`true`)は、処理中のリクエストと
内容が同一(同一モデル・同一入力)の新規リクエストが届いた場合に、
推論を2回実行する代わりに新規リクエストを処理中のリクエストへ
併合し、完了時に同じ結果を共有します。

`server.graceful_shutdown_timeout`は、SIGTERMやCtrl-Cを受けたときに
in-flightリクエストの完了をサーバーが待つ時間の上限です(既定は
未設定=全件完了まで無期限に待つ。待っている間は新規接続を受け付け
ません)。秒数を指定すると、その待ち時間に上限をかけられます。

### localhost外への公開

非loopbackアドレスへのbind(`--host`または`server.host`)はサーバーを
ネットワークへ公開します。APIキー — 設定ファイルの`api_key`(ファイル
は`chmod 600`推奨)または環境変数`EEANE_API_KEY` — を設定すると、
`GET /health`以外の全エンドポイントが`Authorization: Bearer <key>`
ヘッダを要求するようになります。キーなしで非loopbackアドレスを
サービングすると起動時に警告が出ます。`/health`は監視用途のため常に
無認証で開放され、代わりにレート制限がかかります
(`server.health_rate_limit`、既定60リクエスト/分/クライアントIP、`0`で
無効)。これらはアプリケーション層の保護にすぎません。信頼できる
LAN/VPNの外へ公開する場合はリバースプロキシやファイアウォールの背後に
置いてください。

### エンドポイント

- `GET /health` — ステータスとサービス中モデルの一覧(モデルごとに
  `id`/`kind`/バケツ/`loaded`)。無認証・レート制限あり
- `GET /models` (エイリアス: `GET /v1/models`) — OpenAI互換の全モデル一覧
- `POST /v1/embeddings` (エイリアス: `POST /embeddings`) — OpenAI互換
  (`input`は文字列またはリスト、`encoding_format`は`float`/`base64`)。
  埋め込みは既定でL2正規化して返します (モデル毎の`normalize`設定)
- `POST /rerank`, `POST /v1/rerank` — Infinity互換
  (`query`/`documents`/`top_n`/`return_documents`/`raw_scores`)

embeddings/rerankリクエストの`model`フィールド(省略可)は、設定した
モデルidでサービング対象を選択します。省略時はそのエンドポイントの
kindの既定モデル(設定順の先頭)が使われます。未知のidは利用可能なid
一覧付きの404、別kindのモデルidを指定した場合は400になります。
embeddings/rerankエンドポイントは`/v1`配下とルート直下の両方で提供される
ため、base URLは`/v1`付き・なしのどちらでも動作します。各入力はその
モデルの最小の収まる固定長バケツに自動ルーティングされ、最大バケツを
超える入力は警告ログ付きで切り詰められます。

[Open WebUI](https://github.com/open-webui/open-webui)から使う場合:
埋め込みエンジンをOpenAIにしてbase URLを`http://127.0.0.1:7997/v1`、
rerankingエンジンをExternalにしてURLを`http://127.0.0.1:7997/rerank`に
設定してください。APIキーを設定した場合は、OpenAI APIキー欄 / External
reranker APIキー欄にそのキーを入力してください — Open WebUIはeeANEが
期待する`Authorization`ヘッダとして送信します。

### 既知の制限

eeANEはApple Neural Engineでの実行を前提としており、コンパイル済み
モデルをCPUのみの計算経路で実行することはサポートしていません。
Neural Engineが実際には利用できないマシンや構成でコンパイル済み
モデルを実行すると、推論結果が非有限値(NaN/Inf)になることが実測で
確認されています — これは対応する全アーキテクチャで発生し、発生
頻度はアーキテクチャと入力の系列長に依存します。サーバーはこの
ような結果を黙って返す代わりに、実行時に非有限値の出力を検出した
場合`500 Internal Server Error`を返します(エラーメッセージ例:
`model '<id>' produced a non-finite output for bucket <N>; the
compiled model may have run on an unsupported compute path`)。この
エラーが出た場合、実行環境でNeural Engineが実際には使えていない
可能性が高いです。詳しくは下記のトラブルシューティングを参照して
ください。

### トラブルシューティング

- **`404 model not found`**: クライアントが送る`model`フィールドは、
  サーバー側で設定した`id`と完全に一致している必要があります。
  `GET /models`でeeANEが実際にサービングしているid一覧を確認して
  ください。旧バージョンからの移行クライアントは、以前は`model`
  フィールドが完全に無視されていたため任意の名前(または未指定)で
  動作していた点に注意してください — その挙動はなくなりました。
- **`500 ... produced a non-finite output ...`**: 上記「既知の制限」
  を参照してください。モデルがNeural Engine上で動作していないことを
  意味します。リクエストを処理したマシンでNeural Engineが利用可能か
  確認してください。

## 開発

eeANE自体の開発や、以下のリポジトリ前提のツールを使う場合は、
リポジトリをcloneし、インストール済みパッケージの代わりに`uv run`
でチェックアウトから直接コマンドを実行します:

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync --extra compile   # torch/transformersはコンパイル時のみ必要
uv run eeane compile cl-nagoya/ruri-v3-310m
uv run eeane serve
```

`uv run eeane <サブコマンド>`は、上記で説明した`compile`/`serve`/
`check-config`と同じサブコマンドを、インストール済みパッケージでは
なくチェックアウトから実行します。

起動中のサーバーの検証(Core ML直接推論との一致・API互換・レイテンシ)、
およびlint・テストの一括実行は、いずれもリポジトリのチェックアウトが
前提です:

```sh
uv run python tools/verify_server.py all
# 特定のサービング中モデルをCore ML直接推論と突き合わせる場合:
uv run python tools/verify_server.py verify-embedding --model intfloat/multilingual-e5-base
uv run python tools/verify_server.py verify-rerank --model BAAI/bge-reranker-v2-m3
./tools/check.sh   # ruff lint + フォーマットチェック + pytest を一括実行
```

### PoCを試す (歴史的な開発スナップショット)

`poc/`のスクリプトはv0.1〜v0.3の研究記録として凍結されています。
公式の変換手段は上記の`eeane compile`です。ベンチマーク用途では
引き続き実行できます:

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync

# HF配布形式のモデルを models/ruri-v3-310m と models/ruri-v3-reranker-310m に
# 配置してから (例: `huggingface-cli download cl-nagoya/ruri-v3-310m` で取得):

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

## ライセンス

GPL-3.0-or-later。[LICENSE](LICENSE)を参照してください。

`testdata/corpus/` 以下のテストコーパスは[青空文庫](https://www.aozora.gr.jp/)
由来のパブリックドメイン作品であり、GPLの対象外です。詳細は
`testdata/corpus/README.md` を参照してください。
