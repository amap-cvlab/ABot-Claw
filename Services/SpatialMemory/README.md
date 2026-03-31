# SpatialMemory

SpatialMemory 是一个独立的统一机器人记忆模块，面向 Agent、导航和任务系统提供可写入、可检索、可追踪的空间记忆能力。

设计目标:

- 一个入口承载多种记忆类型
- 所有查询结果都可直接用于导航
- 同时支持在线记忆写入和离线记忆构建
- 与旧实现代码隔离，便于独立演进

## 四大类记忆

本模块围绕以下四类能力构建。

### 1. 对象记忆 Object Memory

用于保存机器人视觉检测到的具体物体信息，适合执行找物体、回忆物体出现位置、追踪物体状态等任务。

典型内容:

- 物体名称: 例如 cup、bottle、remote
- 物体位置与姿态: object_pose
- 机器人信息: robot_id、robot_type、robot_pose
- 检测证据: bbox、detect_confidence、图像证据
- 时间信息: timestamp

核心价值:

- 支持按对象名字段检索
- 支持多机器人记忆归属
- 返回可导航目标位置

### 2. 地点记忆 Place Memory

用于保存语义地点锚点，适合建图和场景标注，例如厨房、客厅、充电区、工具台。

典型内容:

- 地点名称: place_name
- 地点位姿: place_pose
- 地点别名: alias
- 标注备注: note
- 标注来源机器人: robot_id、robot_type

核心价值:

- 支持按地点名直接检索
- 支持按空间半径检索附近地点
- 为任务规划提供稳定锚点

### 3. 关键帧记忆 Keyframe Memory

用于承接离线视频或巡检数据提取的关键帧结果，适合大规模新场景记忆构建。

典型内容:

- 相机来源: camera_source
- 关键帧质量: score、rank
- 关键帧时间: timestamp、timestamp_ns
- 关键帧对应位姿: pose
- 图像证据: image

核心价值:

- 支持批量入库
- 支持离线任务接口接入
- 支持回放式场景记忆回填

### 4. 语义帧记忆 Semantic Frame Memory

用于保存图像记忆并支持文本语义检索，适合通过自然语言进行回忆与定位。

典型内容:

- 图像输入: image
- 机器人位姿: robot_pose
- 语义注释: note、tags
- 向量表示: embedding

核心价值:

- 支持文本到图像记忆检索
- 支持补充场景语义描述
- 与对象/地点记忆共同组成统一检索层

## 能力总览

- 写入能力:
	- 对象写入/更新
	- 地点写入/更新
	- 语义帧写入
	- 关键帧批量写入
- 检索能力:
	- 按对象名检索
	- 按地点名检索
	- 按位置半径检索
	- 按文本语义检索
	- 统一组合检索
- 流程能力:
	- 离线关键帧任务创建与状态查询
- 统一返回能力:
	- 所有查询结果统一包含 target_pose、confidence、evidence

## 启动

```bash
cd SpatialMemory
pip install -r requirements.txt
python main.py
```

默认地址:

- http://127.0.0.1:8022
- Swagger: http://127.0.0.1:8022/docs

## 接口说明

### 写入接口

- POST /memory/object/upsert
	- 用途: 新增或更新对象记忆
	- 典型场景: YOLO 检测后写入 cup 的空间位置
- POST /memory/place/upsert
	- 用途: 写入地点锚点
	- 典型场景: 人机交互标注“这里是厨房”
- POST /memory/semantic/ingest
	- 用途: 写入语义帧记忆
	- 典型场景: 机器人拍到新场景并写入文本注释
- POST /memory/keyframe/ingest-batch
	- 用途: 批量写入关键帧
	- 典型场景: 离线巡检视频生成关键帧后一次性入库

### 检索接口

- POST /query/object
	- 用途: 按对象名检索对象记忆
- POST /query/place
	- 用途: 按地点名检索地点记忆
- POST /query/position
	- 用途: 按 x,y,radius 检索附近记忆
- POST /query/semantic/text
	- 用途: 按文本语义检索相关记忆
- POST /query/unified
	- 用途: 组合式检索入口

### 任务接口

- POST /pipeline/tasks
	- 用途: 创建离线关键帧处理任务
- GET /pipeline/tasks/{task_id}
	- 用途: 查询任务状态与进度

### 系统接口

- GET /health
	- 用途: 服务健康检查与记录数监控

## 统一检索返回

所有检索接口都遵循统一结果结构，便于下游导航与控制模块直接消费。

关键字段:

- memory_type: 记忆类别 object/place/keyframe/semantic_frame
- target_pose: 导航目标位姿
- confidence: 结果置信度
- evidence: 图像路径、备注、扩展证据

示例返回结构:

```json
{
	"results": [
		{
			"id": "obj_xxx",
			"memory_type": "object",
			"name": "cup",
			"robot_id": "humanoid_001",
			"robot_type": "humanoid",
			"target_pose": {
				"x": 1.2,
				"y": 1.1,
				"z": 0.8,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 0.0,
				"frame_id": "map"
			},
			"confidence": 0.92,
			"evidence": {
				"image_path": "...",
				"note": "..."
			}
		}
	]
}
```

## 推荐调用流程

在线场景:

1. 调用 GET /health 确认服务就绪
2. 机器人感知后写入对象/地点/语义帧记忆
3. 调用 query 接口检索目标
4. 将 target_pose 传递给导航模块执行

离线场景:

1. 调用 POST /pipeline/tasks 创建离线关键帧任务
2. 调用 GET /pipeline/tasks/{task_id} 轮询进度
3. 任务完成后通过 query 接口验证记忆可检索性

## 存储

- 结构化存储: data/memory_hub.db
- 图像证据: data/images/*.jpg
