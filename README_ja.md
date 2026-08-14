# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANEは、Apple SiliconマックのApple Neural Engine (ANE)上でテキスト埋め込み
モデルとrerankingモデルを動かすためのエンジンです。モデルは
Hugging Faceの配布形式のまま取得し、ローカルでCore ML形式にコンパイルします。
コンパイル済みモデルは数秒でロードでき、ANE上で推論するため、GPUと
ユニファイドメモリの大部分を他の作業のために空けておけます。

> **開発状況: 初期開発中 (v0.4)。**
> v0.1〜v0.3で概念実証を完了:
> [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)
> (日本語ModernBERT埋め込みモデル)と
> [cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)
> (そのクロスエンコーダ型reranker)を共通パイプラインでCore MLに変換し、
> ANE推論・数値精度を検証のうえ、M2 Mac mini上で埋め込み最大約13,600
> 実効トークン/秒(MPS GPUベースラインの2〜3倍)を計測済みです。
> **v0.4で最初のHTTPサーバーを実装**: OpenAI互換の`/v1/embeddings`と
> Infinity互換の`/rerank`をANEから直接サービングし、レスポンスはCore ML
> 直接推論と完全一致します。HTTP経由でも36文書のrerankは約2.0秒
> (同一マシンのInfinity_emb/MPS構成に対する同一リクエストの約8倍速)、
> サーバー常駐メモリは約750MB(置き換え対象の現行構成は6〜8GB)です。
> 設定ファイル・オンデマンドロード・パッケージ配布は後のマイルストーンで
> 実装予定で、現時点のサーバー設定は固定(組み込み)です。

## 動作要件

- Apple Siliconマック (M1以降)
- macOS 13以降
- [uv](https://docs.astral.sh/uv/) (開発環境用)

## サーバーを起動する (v0.4)

```sh
git clone https://github.com/xhighhongo41/eeANE.git
cd eeANE
uv sync

# HF配布形式のモデルを models/ruri-v3-310m と models/ruri-v3-reranker-310m に
# 配置してから (例: `huggingface-cli download cl-nagoya/ruri-v3-310m` で取得)、
# サーバーがロードするCore ML成果物をコンパイルします (初回のみ、各約30秒):
uv run python poc/convert_embedding.py --seq-len 128
uv run python poc/convert_embedding.py --seq-len 512
uv run python poc/convert_embedding.py --seq-len 1024
uv run python poc/convert_reranker.py --seq-len 512
uv run python poc/convert_reranker.py --seq-len 1024

# サーバー起動 (固定設定: 127.0.0.1:7997)
uv run python -m eeane.server
```

エンドポイント:

- `GET /health` — ステータスとサービス中のシーケンス長バケツ
- `POST /v1/embeddings` (エイリアス: `POST /embeddings`) — OpenAI互換
  (`input`は文字列またはリスト、`encoding_format`は`float`/`base64`)。
  埋め込みはL2正規化して返します (Infinity_embと同じ挙動)
- `POST /rerank`, `POST /v1/rerank` — Infinity互換
  (`query`/`documents`/`top_n`/`return_documents`/`raw_scores`)

embeddings/rerankエンドポイントは`/v1`配下とルート直下の両方で提供される
ため、base URLは`/v1`付き・なしのどちらでも動作します。各入力はトークン数に応じて最小の
固定長バケツ(埋め込み: 128/512/1024、reranker: 512/1024)に自動ルーティング
され、最大バケツを超える入力は警告ログ付きで切り詰められます。

[Open WebUI](https://github.com/open-webui/open-webui)から使う場合:
埋め込みエンジンをOpenAIにしてbase URLを`http://127.0.0.1:7997/v1`、
rerankingエンジンをExternalにしてURLを`http://127.0.0.1:7997/rerank`に
設定してください。

起動中のサーバーの検証(Core ML直接推論との一致・API互換・レイテンシ)は:

```sh
uv run python tools/verify_server.py all
```

## PoCを試す (開発スナップショット)

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
