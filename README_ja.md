# eeANE

**e**mbedding **e**ngine for **A**pple **N**eural **E**ngine

eeANEは、Apple SiliconマックのApple Neural Engine (ANE)上でテキスト埋め込み
モデル(将来的にはrerankingモデルも)を動かすためのエンジンです。モデルは
Hugging Faceの配布形式のまま取得し、ローカルでCore ML形式にコンパイルします。
コンパイル済みモデルは数秒でロードでき、ANE上で推論するため、GPUと
ユニファイドメモリの大部分を他の作業のために空けておけます。

> **開発状況: 初期開発中 (v0.1、概念実証)。**
> 現在のマイルストーンでは
> [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)
> (日本語ModernBERT埋め込みモデル)をCore MLに変換し、ANE推論・
> PyTorch FP32基準に対する数値精度・レイテンシを検証しています。
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

# HF配布形式のモデルを models/ruri-v3-310m に配置してから
# (例: `huggingface-cli download cl-nagoya/ruri-v3-310m` で取得):
uv run python poc/convert_embedding.py --seq-len 512   # HF -> .mlmodelc
uv run python poc/verify_accuracy.py --seq-len 512     # FP32基準との精度比較
uv run python poc/benchmark_latency.py --seq-len 512 --compute-units CPU_AND_NE --compute-plan
```

## ライセンス

GPL-3.0-or-later。[LICENSE](LICENSE)を参照してください。

`testdata/corpus/` 以下のテストコーパスは[青空文庫](https://www.aozora.gr.jp/)
由来のパブリックドメイン作品であり、GPLの対象外です。詳細は
`testdata/corpus/README.md` を参照してください。
