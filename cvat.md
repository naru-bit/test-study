

---

# CVAT + Nuclio (GPU) 完全環境構築ガイド

## 1. インフラ構成

* **WSL2**: Ubuntu 24.04 (Noble)
* **Docker Engine**: Docker Desktopを使用せず、Ubuntu内で直接動作
* **NVIDIA GPU**: NVIDIA Container Toolkitによるパススルー設定

---

## 2. NVIDIA Container Toolkit のインストール

Ubuntu 24.04ではパッケージが見つからないエラーが多発するため、以下の手順で固定します。

```bash
# GPGキーとリポジトリの登録
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# インストール
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Docker設定の更新と再起動
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

```

---

## 3. CVAT & Nuclio の起動

CVAT本体と、モデルを管理するNuclioダッシュボードを起動します。

```bash
cd ~/cvat
# CVAT_HOSTは外部接続時に重要（手順5参照）
export CVAT_HOST=localhost 
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

```

---

## 4. モデルの作成とデプロイ (Serverless Functions)

モデルの置き場所は `~/cvat/serverless/ultralytics/yolov26/nuclio/` と仮定します。

### ① `function-gpu.yaml` (設定ファイル)

**注意:** 名前の後ろにコメント (`#`) を書くと、Nuclioが名前の一部と誤認して `error` 状態になるため、コメントは削除すること。

```yaml
metadata:
  name: gpu-ultralytics-yolov26
  namespace: cvat
  annotations:
    name: YOLOv26 GPU
    type: detector
    framework: pytorch
    spec: |
      [
        { "id": 0, "name": "person" },
        { "id": 1, "name": "bicycle" }
      ]

spec:
  handler: main:handler
  runtime: 'python:3.8'
  eventTimeout: 30s
  build:
    baseImage: ultralytics/ultralytics:latest
  resources:
    limits:
      nvidia.com/gpu: 1

```

### ② `main.py` (推論ロジック)

モデル読み込みと画像解析の橋渡しを記述します。

```python
import json
import base64
import io
from PIL import Image
from ultralytics import YOLO

def init_context(context):
    context.logger.info("Init context...  0%")
    model = YOLO('yolov8n.pt') # または独自の重みファイル
    setattr(context.user_data, 'model', model)
    context.logger.info("Init context... 100%")

def handler(context, event):
    context.logger.info("Run YOLOv26 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data['image']))
    threshold = data.get('threshold', 0.5)
    image = Image.open(buf)

    results = context.user_data.model(image, conf=threshold)
    
    encoded_results = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            encoded_results.append({
                'confidence': str(box.conf[0]),
                'label': result.names[int(box.cls[0])],
                'points': box.xyxy[0].tolist(),
                'type': 'rectangle'
            })

    return context.Response(body=json.dumps(encoded_results), headers={},
                            content_type='application/json', status_code=200)

```

### ③ デプロイ用スクリプト (`deploy_gpu.sh`)

```bash
#!/bin/bash
func_root="./serverless/ultralytics/yolov26/nuclio"
nuctl deploy --project cvat \
    --path "$func_root" \
    --file "$func_root/function-gpu.yaml" \
    --platform local

```

---

## 5. 外部接続（LAN内の他PCから接続）の設定

WSL2のIPアドレスは変動するため、Windows側でポートフォワーディングを設定します。

### Windows側 (管理者PowerShell)

```powershell
# WSL2のIPを確認 (WSLで ip addr show eth0 を実行した値)
$wsl_ip = "172.27.231.211" 

# ポート転送 (8080: CVAT, 8070: Nuclio)
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wsl_ip
netsh interface portproxy add v4tov4 listenport=8070 listenaddress=0.0.0.0 connectport=8070 connectaddress=$wsl_ip

# ファイアウォール許可
New-NetFirewallRule -DisplayName "CVAT" -Direction Inbound -LocalPort 8080,8070 -Protocol TCP -Action Allow

```

### WSL2側

WindowsのLAN側IP（例: `192.168.1.6`）をセットして起動します。

```bash
export CVAT_HOST=192.168.1.6
docker compose up -d

```

---

## 6. トラブルシューティング

1. **Nuclioコンテナが止まらない**: `docker rm -f nuclio-local-storage-reader` 等で個別に削除。
2. **モデルがModelsタブに現れない**: `nuctl get function -n cvat` で `ready` であることを確認。
3. **WSL再起動後に繋がらない**: `ip addr` でIPが変化していないか確認し、Windowsの `netsh` を再設定。

---

