

---

# CVAT + Nuclio (GPU) 環境構築ガイド (WSL2 / Ubuntu 24.04)

## 1. 前提条件

* OS: Windows 11 + WSL2 (Ubuntu 24.04 Noble)
* GPU: NVIDIA GPU
* Docker Desktop を使用せず、WSL2内の Docker Engine を使用

---

## 2. NVIDIA Container Toolkit のインストール (重要)

WSL2上のDockerでGPUを使用するために必須です。Ubuntu 24.04ではリポジトリの登録方法に注意が必要です。

### 詰まったポイントと対策

**エラー:** `E: Unable to locate package nvidia-container-toolkit`
**原因:** 24.04用のリポジトリパスが正しく認識されない、またはシェル変数 `$(ARCH)` が展開されない。

**解決策:**
標準的な `sources.list.d` に、アーキテクチャを明示したリポジトリを追加します。

```bash
# 1. GPGキーの登録
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2. リポジトリの追加 (amd64を明示)
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. インストール
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4. Dockerへの紐付けと再起動
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

```

**確認:**
`docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` でGPU情報が出ればOK。

---

## 3. CVAT & Nuclio の起動

```bash
# CVATディレクトリにて
export CVAT_HOST=localhost # ローカルの場合
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

```

---

## 4. サーバーレスモデル (Nuclio) のデプロイ

### 詰まったポイントと対策

**エラー:** `nuctl get function` でステータスが `error` になる、または名前に `#` が混じる。
**原因:** `function-gpu.yaml` 内のコメント（`#`）の書き方が不適切で、Nuclioが名前の一部として誤認した。

**解決策:**

1. **YAMLの清掃:** `name:` の行の末尾にあるコメントを削除する。
2. **プロジェクトのリセット:**
```bash
# プロジェクトごと消去してリセット
nuctl delete project cvat --force
# もし消えない場合は、コンテナを直接消去
docker rm -f $(docker ps -a | grep nuclio | awk '{print $1}')

```


3. **再デプロイ:**
```bash
./serverless/deploy_gpu.sh ./serverless/ultralytics/yolov26/nuclio/

```



---

## 5. 外部PC（同じネットワーク）からの接続設定

WSL2はWindowsの内部ネットワークにあるため、他のPCから接続するには「ポートフォワーディング」が必要です。

### 手順 (Windows側のPowerShellを管理者で実行)

1. **IPの確認:**
* Windows側: `ipconfig` → `192.168.1.6` (例)
* WSL2側: `ip addr` → `172.27.231.211` (例)


2. **ポート転送の設定:**
```powershell
# WindowsへのアクセスをWSL2へ転送
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=172.27.231.211
netsh interface portproxy add v4tov4 listenport=8070 listenaddress=0.0.0.0 connectport=8070 connectaddress=172.27.231.211

```


3. **ファイアウォールの許可:**
```powershell
New-NetFirewallRule -DisplayName "CVAT" -Direction Inbound -LocalPort 8080,8070 -Protocol TCP -Action Allow

```


4. **CVATの再起動 (WSL2側):**
```bash
# 外部から見えるIPを指定して起動
export CVAT_HOST=192.168.1.6
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

```



---

## 6. トラブルシューティングまとめ

* **モデルが「Models」タブに出ない:** - `nuctl get function -n cvat` でステータスが `ready` か確認。
* `docker compose ... up -d` を再実行して CVAT サーバーをリフレッシュ。


* **WSLを再起動したら繋がらなくなった:** - WSLの内部IPが変わっているため、`netsh interface portproxy` を新しいIPでやり直す。
* **GPUが使えない:** - `nvidia-smi` がWSL上で動くか、`nvidia-container-toolkit` が入っているかを再確認。

---
