# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANEは、Apple SiliconマックのApple Neural Engine (ANE)上でテキスト埋め込み
モデルとrerankingモデルを動かすためのエンジンです。モデルは
Hugging Faceの配布形式のまま取得し、ローカルでCore ML形式にコンパイルします。
コンパイル済みモデルは数秒でロードでき、ANE上で推論するため、GPUと
ユニファイドメモリの大部分を他の作業のために空けておけます。

> **開発状況: 初期開発中 (v0.6)。**
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
> `[compile]`エクストラへ分離されました。パッケージ配布は後の
> マイルストーンで実装予定です。

## 動作要件

- Apple Siliconマック (M1以降)
- macOS 13以降
- Xcodeコマンドラインツール (`xcode-select --install`) — `eeane compile`
  が`xcrun coremlcompiler`を使用します
- [uv](https://docs.astral.sh/uv/) (開発環境用)

## モデルのコンパイルとサーバー起動 (v0.6)

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync --extra compile   # torch/transformersはコンパイル時のみ必要

# Hugging FaceのモデルID(自動ダウンロード)またはHF配布形式のローカル
# ディレクトリから直接コンパイルします。初回のみ。成果物は
# ~/.cache/eeane/ 配下に生成され、1バケツあたり約30〜100秒です:
uv run python -m eeane compile cl-nagoya/ruri-v3-310m
uv run python -m eeane compile cl-nagoya/ruri-v3-reranker-310m

# 各実行の最後に[[models]]のTOMLスニペットが標準出力に表示されます。
# それを ./eeane.toml に貼り付けて([server]節はeeane.example.toml参照)、
# サーバーを起動します:
uv run python -m eeane serve
```

`eeane compile`はモデルの`config.json`からバックエンドを自動選択し
(v0.6はModernBERTアーキテクチャに対応)、embedding/rerankerの種別も
自動判別します。バケツの既定は埋め込み128/512/1024、reranker 512/1024
で、`--buckets 512,2048`のように変更できます(S2048はM2実機で約518ms/
推論を検証済み)。再実行時は最新の成果物をスキップします(`--force`で
再変換)。変換後には**セルフチェック**が走り、FP32基準の精度検証・
Neural Engineへの配置率計測・ウォームレイテンシ記録を行います。表示
されるサマリは互換性レポートを兼ねるので、未検証ハードウェア
(M1/M3/M4など)で動かした際はぜひIssueに貼ってください。トークナイザは
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
uv run python -m eeane serve --config /path/to/eeane.toml
uv run python -m eeane serve --host 192.168.1.20 --port 7997

# 設定ファイルを検証し、解決後の有効設定を表示します (サーバーは起動
# しません。APIキーの値は表示されません):
uv run python -m eeane check-config --config /path/to/eeane.toml
```

設定ファイルはサービングするモデル(凍結済み`tokenizer.json`、シーケンス長
バケツごとのコンパイル済み成果物、L2正規化)を定義するため — これは
まさに`eeane compile`のスニペットが埋める内容です — コードに
触れずにバケツを増減できます。rerankerエントリは省略可能で、その場合は
embedding専用サーバーになります(`/rerank`は503を返します)。
`uv run python -m eeane.server`(v0.4の起動法)は`eeane serve`の
エイリアスとして引き続き使えます。

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

- `GET /health` — ステータスとサービス中のシーケンス長バケツ
  (無認証・レート制限あり)
- `GET /models` (エイリアス: `GET /v1/models`) — OpenAI互換のモデル一覧
- `POST /v1/embeddings` (エイリアス: `POST /embeddings`) — OpenAI互換
  (`input`は文字列またはリスト、`encoding_format`は`float`/`base64`)。
  埋め込みはL2正規化して返します (Infinity_embと同じ挙動)
- `POST /rerank`, `POST /v1/rerank` — Infinity互換
  (`query`/`documents`/`top_n`/`return_documents`/`raw_scores`)

embeddings/rerankエンドポイントは`/v1`配下とルート直下の両方で提供される
ため、base URLは`/v1`付き・なしのどちらでも動作します。各入力はトークン数に応じて最小の
固定長バケツ(デフォルト — 埋め込み: 128/512/1024、reranker: 512/1024)に
自動ルーティングされ、最大バケツを超える入力は警告ログ付きで切り詰められます。

[Open WebUI](https://github.com/open-webui/open-webui)から使う場合:
埋め込みエンジンをOpenAIにしてbase URLを`http://127.0.0.1:7997/v1`、
rerankingエンジンをExternalにしてURLを`http://127.0.0.1:7997/rerank`に
設定してください。APIキーを設定した場合は、OpenAI APIキー欄 / External
reranker APIキー欄にそのキーを入力してください — Open WebUIはeeANEが
期待する`Authorization`ヘッダとして送信します。

起動中のサーバーの検証(Core ML直接推論との一致・API互換・レイテンシ)は:

```sh
uv run python tools/verify_server.py all
```

## PoCを試す (歴史的な開発スナップショット)

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
