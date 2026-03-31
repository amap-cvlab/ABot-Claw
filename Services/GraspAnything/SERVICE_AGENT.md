# GraspAnything Service（Agent 详细接入手册）

## 1. 服务作用与能力边界
GraspAnything 服务用于在 **单帧 RGB-D** 图像上执行抓取位姿检测。

可实现能力：
- YOLO 检测指定目标物体（如 `cup`、`bottle`）
- 在目标区域点云上运行 AnyGrasp
- 返回抓取候选的 6DoF 位姿（相机坐标系）
- 输出每个候选的分数、夹爪宽度、旋转矩阵、四元数

非目标：
- 不直接输出机械臂规划轨迹
- 不做跨帧 tracking
- 不做 memory 存储

默认端口：`8015`

## 2. 启动与运行参数
```bash
cd /home/crh/ABotClaw-Services/GraspAnything
PORT=8015 GRASP_CHECKPOINT_PATH=/abs/path/to/checkpoint_detection.tar DEVICE=auto <python_path> main.py
```

关键环境变量：
- `PORT`：监听端口，默认 `8015`
- `GRASP_CHECKPOINT_PATH`：AnyGrasp checkpoint 路径（必填）
- `GRASP_YOLO_MODEL`：YOLO 模型名，默认 `yolov5l6`
- `DEVICE`：统一设备参数，支持 `auto | cpu | cuda | cuda:0 | 0`
- `GRASPANYTHING_DEVICE`：兼容旧参数（`DEVICE` 优先）

## 3. 输入数据规范（非常重要）
`POST /grasp/detect` 必须提供：
1) `color_image`：RGB 图（建议 raw base64）
2) `depth_image`：深度图（建议 16-bit PNG 的 base64）
3) `camera_intrinsics`：3x3 相机内参矩阵 K
4) `object_name`：目标类别名

图像字段支持：
- raw base64（推荐）
- data URI（`data:image/png;base64,...`）
- 本地路径
- URL（http/https）

建议 Agent 始终发送 base64，避免路径依赖。

## 4. 接口说明

### `GET /health`
用途：探活与模型加载状态确认。

返回字段：
- `status`
- `version`
- `device`
- `checkpoint_path`
- `model_name`
- `model_loaded`

### `POST /grasp/detect`
用途：执行单帧抓取检测。

请求体：
```json
{
  "color_image": "<base64_rgb>",
  "depth_image": "<base64_depth_png>",
  "camera_intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],
  "object_name": "cup",
  "top_k": 5
}
```

返回体（示意）：
```json
{
  "frame_id": "wrist_camera_color_optical_frame",
  "target": "cup",
  "top_k": 5,
  "count": 1,
  "results": [
    {
      "label": "cup",
      "confidence": 0.91,
      "xyxy": [120, 80, 260, 290],
      "grasps": [
        {
          "score": 0.66,
          "width": 0.05,
          "translation_camera": [0.11, 0.03, 0.42],
          "translation_camera_retreat": [0.11, 0.03, 0.52],
          "quaternion_camera_xyzw": [0.0, 0.7, 0.0, 0.7],
          "rotation_camera": [[...],[...],[...]]
        }
      ]
    }
  ],
  "latency_ms": 185.3
}
```

## 5. Agent 推荐调用流程
1. 调 `GET /health`，确认 `model_loaded=true`
2. 准备 RGB + 深度 + 相机内参
3. 调 `POST /grasp/detect`
4. 若 `results[].grasps` 为空，调整目标类别或图像质量后重试

## 6. 常见错误与处理
- `400 Invalid input payload`：图像或内参格式错误
- `400 object_name cannot be empty`
- `503 Grasp model not initialized`
- `500 Grasp inference failed`：模型推理异常（可重试）

排障建议：
- 深度图请用 `uint16` PNG（毫米）
- `camera_intrinsics` 必须是 3x3
- 先用 `top_k=1` 验证链路，再提升

## 7. 最小调用示例
```bash
curl -X POST http://127.0.0.1:8015/grasp/detect \
  -H 'Content-Type: application/json' \
  -d '{"color_image":"<base64_rgb>","depth_image":"<base64_depth>","camera_intrinsics":[[600,0,320],[0,600,240],[0,0,1]],"object_name":"cup","top_k":3}'
```
