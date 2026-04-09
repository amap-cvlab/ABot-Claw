# GraspAnything Service

GraspAnything 是一个纯 HTTP 的抓取检测服务，输入单帧 RGB-D（支持 base64），输出目标物体的抓取候选位姿。

当前服务不依赖 ROS 运行环境。

## 功能概览

- 输入：RGB 图、深度图、相机内参、目标类别名
- 推理：YOLO 检测目标 + AnyGrasp 生成 6DoF 抓取候选
- 输出：按分数排序的抓取列表（相机坐标系）

## 启动

```bash
cd /home/crh/ABotClaw-Services/GraspAnything
pip install -r requirements.txt
PORT=8015 GRASP_CHECKPOINT_PATH=/abs/path/to/checkpoint_detection.tar DEVICE=auto python main.py
```

说明：首次启动可能会通过 torch.hub 下载 YOLO 权重。

## API

### GET /health

用于探活与模型加载状态确认。

### POST /grasp/detect

请求体示例：

```json
{
	"color_image": "<base64_rgb>",
	"depth_image": "<base64_depth_png>",
	"camera_intrinsics": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
	"object_name": "cup",
	"top_k": 5
}
```

图像字段支持：

- raw base64
- data URI（例如 data:image/png;base64,...）
- 本地路径
- URL（http/https）

深度图建议使用 uint16 PNG（单位：毫米）。

## 本地调用示例

```bash
python test_api.py \
	--url http://127.0.0.1:8015/grasp/detect \
	--color ./example_data/color.png \
	--depth ./example_data/depth.png \
	--camera-k '[[600,0,320],[0,600,240],[0,0,1]]' \
	--object-name cup \
	--top-k 3
```

## 常见错误

- 400 Invalid input payload: 图像解码失败、内参不是 3x3、RGB/Depth 分辨率不一致
- 400 object_name cannot be empty
- 503 Grasp model not initialized
- 500 Grasp inference failed

