# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANEは、Apple SiliconマックのApple Neural Engine (ANE)上でテキスト埋め込み
モデルとrerankingモデルを動かすためのエンジンです。モデルは
Hugging Faceの配布形式のまま取得し、ローカルでCore ML形式にコンパイルします。
コンパイル済みモデルは数秒でロードでき、ANE上で推論するため、GPUと
ユニファイドメモリの大部分を他の作業のために空けておけます。

> **開発状況: 初期開発中 (v0.2、概念実証)。**
> PoCは現在
> [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)
> (日本語ModernBERT埋め込みモデル、v0.1)と
> [cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)
> (そのクロスエンコーダ型reranker、v0.2)の両方をカバーしています。
> 両モデルを共通パイプラインでCore MLに変換し、ANE推論・
> PyTorch FP32基準に対する数値精度・レイテンシを検証済みです。
> サーバー機能やインストール可能なパッケージはまだありません
> (OpenAI互換のembeddings / rerankサーバーは後のマイルストーンで実装予定)。

## 動作要件

- Apple Siliconマック (M1以降)
- macOS 13以降
- [uv](https://docs.astral.sh/uv/) (開発環境用)

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
```

## ライセンス

GPL-3.0-or-later。[LICENSE](LICENSE)を参照してください。

`testdata/corpus/` 以下のテストコーパスは[青空文庫](https://www.aozora.gr.jp/)
由来のパブリックドメイン作品であり、GPLの対象外です。詳細は
`testdata/corpus/README.md` を参照してください。
