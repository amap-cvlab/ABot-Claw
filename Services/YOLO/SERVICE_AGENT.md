# YOLO Service（Agent 详细接入手册）

## 1. 服务作用与能力边界
YOLO 服务用于“单张图像目标检测”。

可实现能力：
- 检测常见类别目标（如 person、bus 等）
- 返回标准检测框与置信度
- 支持按请求覆盖阈值（`conf_thres`、`iou_thres`）

非目标：
- 不做时序跟踪（tracking）
- 不做语义记忆存储
- 不输出实例分割掩码

默认端口：`8013`

## 2. 启动与运行参数
```bash
cd /home/crh/ABotClaw-Services/YOLO
PORT=8013 DEVICE=auto /home/crh/miniconda3/envs/service_yolo/bin/python main.py
```

关键环境变量：
- `PORT`：监听端口，默认 `8013`
- `DEVICE`：统一设备参数，支持 `auto | cpu | cuda | cuda:0 | 0`
- `YOLO_DEVICE`：兼容旧参数（`DEVICE` 优先）
- `YOLO_MODEL_PATH`：权重路径，默认 `./yolov5l6.pt`

设备说明：
- 服务内部会转换为 torch hub 需要的 `hub_device`（如 `0`）。
- 健康接口会同时返回 `device` 与 `hub_device` 便于排障。

## 3. 图像输入规范（Agent 必读）
`image` 支持：
1) raw base64（推荐）
2) data URI（`data:image/jpeg;base64,...`）
3) 本地路径
4) URL（http/https）

建议 Agent 一律使用 raw base64，避免服务端路径依赖。

## 4. 接口清单与语义

### `GET /health`
用途：探活 + 模型加载状态确认。

典型返回字段：
- `status`
- `version`
- `device`
- `hub_device`
- `model_path`
- `model_loaded`

### `POST /detect`
用途：执行目标检测。

请求体：
```json
{
  "image": "<base64>",
  "conf_thres": 0.25,
  "iou_thres": 0.45
}
```

参数建议：
- `conf_thres`：低召回场景可降低到 `0.15~0.25`
- `iou_thres`：去重更强可降低，保留更多重叠框可提高

返回体：
```json
{
  "detections": [
    {
      "x1": 50.5,
      "y1": 398.3,
      "x2": 248.8,
      "y2": 903.9,
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "count": 7
}
```

## 5. Agent 推荐调用策略
1) 先 `GET /health`（确认 `model_loaded=true`）
2) 执行 `POST /detect`
3) 如果需要后处理，按 `class_id/class_name` 过滤

建议：
- 对 5xx 可短重试一次
- 对 400 直接修请求体，不重试

## 6. 常见错误与排查
- `400 Invalid image payload`：图像字段非法
- `503 Model not initialized`：服务仍在加载模型
- 启动失败且报 CUDA device 错误：检查 `DEVICE` 是否设置为 `auto` 或合法值

## 7. 最小可用请求示例
```bash
curl -X POST http://127.0.0.1:8013/detect \
  -H 'Content-Type: application/json' \
  -d '{"image":"<base64>","conf_thres":0.25,"iou_thres":0.45}'
```
