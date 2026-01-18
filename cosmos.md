

---

# Nvidia-cosmos Predict 2.5 ファインチューニング実践ガイド

このドキュメントは、Nvidia Cosmos Predict 2.5モデルを独自の動画データセット等で追加学習（Fine-tuning）するための手順書です。
**「ローカル環境（venv/uv）」** と **「Singularityコンテナ（HPC等）」** の2つのパターンに対応しています。

## 0. 前提条件とハードウェア

作業を始める前に、以下の環境が整っているか確認してください。

* **GPU:** NVIDIA Ampereアーキテクチャ以上 (RTX 30シリーズ, A100, H100など)
* ※VRAMはモデルサイズ(2B/14B)によりますが、学習には最低でも24GB以上（できればA100/H100クラス）が推奨されます。


* **OS:** Linux x86-64 (Ubuntu 22.04推奨)
* **ドライバ:** NVIDIA Driver >= 570.124.06 (CUDA 12.8互換)
* **Hugging Face:** アカウント作成とAccess Tokenの取得、および[Cosmosモデルのライセンス同意](https://www.google.com/search?q=https://huggingface.co/nvidia/Cosmos-Predict-2.5-2B)が完了していること。

---

## 1. 環境構築

まずはソースコードを取得します。

```bash
# Git LFSのインストール（重いファイルのダウンロードに必要）
sudo apt install git-lfs
git lfs install

# リポジトリのクローン
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
cd cosmos-predict2.5
git lfs pull

```

これ以降は、**[パターンA: ローカル直接実行]** か **[パターンB: Singularity]** かを選んで進めてください。

### パターンA: ローカル環境 (venv / uv使用)

公式推奨のパッケージマネージャ `uv` を使用して環境を作ります。

```bash
# 1. システムライブラリのインストール
sudo apt install curl ffmpeg tree wget

# 2. 'uv' (高速なPythonパッケージ管理ツール) のインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. 仮想環境の作成と依存関係のインストール
# これにより .venv フォルダが作成され、そこにライブラリが入ります
uv python install
uv sync --extra=cu128
source .venv/bin/activate

# 4. Hugging Face CLIのインストールとログイン
uv tool install -U "huggingface_hub[cli]"
hf auth login
# トークンを入力してログインしてください

```

### パターンB: Singularity (Apptainer) 環境

HPCや大学のスパコン等でDockerが使えない場合、Singularityコンテナを使用します。
Singularityイメージを作成するには、**一度Dockerでビルドしてから変換する**のが最も確実です（Dockerが使える手元のPC等でビルドしてください）。

#### 手順B-1: Dockerイメージのビルドと変換

```bash
# 1. Dockerイメージのビルド (リポジトリルートで実行)
# Ampere/Hopper GPU向け
docker build -f Dockerfile -t cosmos-predict:latest .

# 2. DockerイメージをSingularityイメージ(.sif)に変換
# docker-daemonから直接変換する場合（spythonやsingularityが必要です）
singularity build cosmos-predict.sif docker-daemon://cosmos-predict:latest

# ※もし変換環境がない場合、Dockerでtarに書き出してからSingularityで読み込む手もあります
# docker save cosmos-predict:latest -o cosmos-predict.tar
# singularity build cosmos-predict.sif docker-archive://cosmos-predict.tar

```

#### 手順B-2: 実行環境への配置

作成した `cosmos-predict.sif` を、実行したいサーバー（GPUマシン）の `cosmos-predict2.5` フォルダ直下に配置してください。

---

## 2. データセットの準備と配置

ここが最も分かりにくい部分です。公式ガイドの例（GR1データセット）をベースに、**独自のデータセットをどう配置すべきか**を解説します。

学習スクリプトは、特定のフォルダ構造を期待しています。

### 2.1 フォルダ構成の作成

プロジェクトルート（`cosmos-predict2.5/`）の中に、以下の構造を作成します。

```text
cosmos-predict2.5/
└── datasets/
    └── benchmark_train/
        └── my_custom_data/      <-- 任意のデータセット名
            ├── videos/          <-- 動画ファイル(.mp4)はすべてここに入れる
            │   ├── video_001.mp4
            │   ├── video_002.mp4
            │   └── ...
            ├── metas/           <-- プロンプト(テキスト)はここ（後述のスクリプトで生成推奨）
            │   ├── video_001.txt
            │   └── ...
            └── metadata.csv     <-- 動画とパスの対応リスト

```

### 2.2 動画データの配置

学習させたい動画ファイル（mp4等）を `datasets/benchmark_train/my_custom_data/videos/` に入れてください。

### 2.3 メタデータとプロンプトの作成

動画に対応するテキストプロンプト（「ロボットがリンゴを掴んでいる」など）を用意します。

1. **metadata.csv の作成:**
最低限、動画ファイル名やパスが記述されたCSVが必要です。公式スクリプトを利用する場合、以下のような形式が一般的です。
`video_path, caption` のような列を持つCSVを用意するか、公式の `scripts.create_prompts_for_gr1_dataset` を参考に自作スクリプトで生成します。
2. **プロンプトtxtの生成:**
各動画に対応する `.txt` ファイルを `metas/` フォルダに作成します。
ファイル名は動画ファイル名と対応している必要があります（例: `video_001.mp4` に対し `video_001.txt`）。
**簡易スクリプト例 (全動画に同じプロンプトをつける場合):**
```python
import os

video_dir = "datasets/benchmark_train/my_custom_data/videos"
meta_dir = "datasets/benchmark_train/my_custom_data/metas"
os.makedirs(meta_dir, exist_ok=True)

prompt = "A robot arm picking up an object based on physical AI simulation."

for video in os.listdir(video_dir):
    if video.endswith(".mp4"):
        txt_name = os.path.splitext(video)[0] + ".txt"
        with open(os.path.join(meta_dir, txt_name), "w") as f:
            f.write(prompt)

```



---

## 3. 追加学習 (Fine-tuning) の実行

### 3.1 出力先とキャッシュの設定

学習結果（チェックポイント）やモデルのダウンロード先を指定します。容量の大きいディスクを指定してください。

```bash
# モデルの一時保存場所
export HF_HOME=$HOME/.cache/huggingface

# 学習結果（チェックポイント）の保存先
export IMAGINAIRE_OUTPUT_ROOT=$HOME/cosmos_output
mkdir -p $IMAGINAIRE_OUTPUT_ROOT

```

### 3.2 設定ファイル (Config) の確認

学習コマンドで指定する `experiment` 名は、Pythonの設定ファイル内で定義されています。
`cosmos_predict2/_src/predict2/configs/video2world/config.py` を確認してください。

独自データで学習する場合、このファイルを編集して、新しい `dataset` 定義を追加するのが正攻法ですが、まずは公式の例 (`predict2_video2world_training_2b_groot_gr1_480`) のデータパス部分を読み替えるか、コードを少し修正して自分のデータパスを指すようにする必要があります。

**★重要:** 初回はとりあえず動かすために、データセットフォルダ名を公式と同じ `datasets/benchmark_train/gr1` にしてしまうのが一番手っ取り早いです（中身を自分のデータに差し替える）。

### 3.3 学習コマンドの実行

#### パターンA: ローカル (venv) の場合

```bash
# 2Bモデルの学習例 (GPU 1枚の場合)
torchrun --nproc_per_node=1 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=predict2_video2world_training_2b_groot_gr1_480

```

※ `experiment=` の後は config.py に書かれている実験設定名を指定します。

#### パターンB: Singularity の場合

Singularityでは、ホスト側のフォルダをコンテナ内にマウント（バインド）する必要があります。
また、NVIDIA GPUを使うための `--nv` オプションが必須です。

```bash
# カレントディレクトリと出力先、キャッシュをバインドして実行
singularity exec --nv \
  -B .:/workspace \
  -B $IMAGINAIRE_OUTPUT_ROOT:/output \
  -B $HOME/.cache/huggingface:/root/.cache/huggingface \
  --pwd /workspace \
  cosmos-predict.sif \
  torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- \
    experiment=predict2_video2world_training_2b_groot_gr1_480 \
    job.workspace.root_dir=/output

```

**Singularityコマンドの解説:**

* `--nv`: GPUを利用可能にします。
* `-B .:/workspace`: 現在のフォルダ（コードがある場所）をコンテナ内の `/workspace` にマウントします。
* `-B ...:/output`: 出力先をマウントします。
* `job.workspace.root_dir=/output`: 設定ファイルの出力先設定を、コンテナ内のマウントパスで上書きしています。

---

## 4. 学習後のチェックポイント変換と推論

学習が終わると、指定した出力ディレクトリに **DCP形式 (Distributed Checkpoint)** で保存されます。これを推論用に変換する必要があります。

### 4.1 チェックポイントの変換 (.pt形式へ)

```bash
# 最新のチェックポイントパスを取得
CHECKPOINTS_ROOT=$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_ROOT/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_ROOT/$CHECKPOINT_ITER

# 変換スクリプトの実行
# (Singularityの場合は singularity exec ... python ... としてください)
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

```

これにより、`model_ema_bf16.pt` などが生成されます。

### 4.2 推論 (動画生成)

推論には設定用JSONファイルが必要です（`assets/sample_gr00t_dreams_gr1/gr00t_image2world.json` などを参考に作成）。

```bash
# 推論実行
torchrun --nproc_per_node=1 examples/inference.py \
  -i assets/sample_gr00t_dreams_gr1/gr00t_image2world.json \
  -o outputs/my_result \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_video2world_training_2b_groot_gr1_480

```

`outputs/my_result` フォルダに生成された動画が保存されます。

---

## トラブルシューティング・注意点

1. **メモリ不足 (OOM):**
GPUメモリが足りない場合は、バッチサイズを下げてください。configファイル内の `batch_size` 関連のパラメータを調整するか、CLI引数でオーバーライドを試みてください。
2. **Hugging Face認証エラー:**
コンテナ内で `HF_TOKEN` 環境変数が渡っていない可能性があります。Singularity実行時に `--env HF_TOKEN=$HF_TOKEN` をつけるか、キャッシュディレクトリを正しくマウントしてください。
3. **パスが見つからない:**
特にSingularityの場合、`-B` でマウントしたパスと、プログラムが参照するパス（config内のパス）が一致しているかよく確認してください。コンテナ内では `/home/user/...` は見えないことが多いため、明示的なマウントが必要です。
