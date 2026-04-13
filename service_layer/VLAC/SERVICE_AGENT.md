# VLAC Critic Service（Agent 详细接入手册）

## 1. 服务作用与能力边界
VLAC 服务用于“任务进度比较评估”：输入“当前图像 + 参考图像 + 任务描述”，输出 critic 分数与累计 value。

可实现能力：
- 比较两张图在任务进展上的相对优劣
- 输出 `critic_list`（相对变化）
- 输出 `value_list`（进度累计轨迹）

典型场景：
- 机械臂操作进度评估
- 成功/失败趋势判断
- 策略回放中的自动打分

非目标：
- 不返回检测框
- 不存储长期记忆
- 不直接给动作控制指令

默认端口：`8014`

## 2. 启动与运行参数
```bash
cd /home/crh/ABotClaw-Services/VLAC
PORT=8014 VLAC_MODEL_PATH=/home/crh/code/hdj/VLAC/models DEVICE=auto /home/crh/miniconda3/envs/vlac/bin/python main.py
```

关键环境变量：
- `PORT`：监听端口，默认 `8014`
- `DEVICE`：统一设备参数，支持 `auto | cpu | cuda | cuda:0 | 0`
- `VLAC_DEVICE`：兼容旧参数（`DEVICE` 优先）
- `VLAC_MODEL_PATH`：模型路径（必填）
- `VLAC_MODEL_TYPE`：默认 `internvl2`

## 3. 图像输入规范（Agent 必读）
`image` 与 `reference_image` 均支持：
1) raw base64（推荐）
2) data URI（`data:image/jpeg;base64,...`）
3) 本地路径
4) URL（http/https）

建议 Agent 一律传 raw base64，保证跨机器可用。

重要：`reference_image` 不是独立上传接口，必须和 `image` 一起在同一个 `POST /critic` 请求中传入。

## 4. 接口说明

### `GET /health`
用途：判断服务是否完成初始化。

返回字段：
- `status`
- `version`
- `device`
- `model_type`
- `model_path`
- `model_loaded`

### `POST /critic`
用途：对当前图与参考图进行任务进度评估。

注意：该接口要求同时提供 `image` 和 `reference_image`；缺少任一字段都会导致请求失败。

请求体：
```json
{
  "image": "<base64>",
  "reference_image": "<base64>",
  "task_description": "Scoop the rice into the rice cooker.",
  "batch_num": 1,
  "rich": false
}
```

字段语义：
- `image`：当前时刻画面
- `reference_image`：对比基准画面
- `task_description`：任务文本（必填）
- `batch_num`：推理批大小，默认 1；压力场景可逐步升高
- `rich`：是否启用富输出模式（通常保持 false）

返回体：
```json
{
  "critic_list": [-40.0],
  "value_list": [0.0, -40.0],
  "latency_ms": 1200.0
}
```

解释：
- `critic_list`：每一段比较的相对得分
- `value_list`：由 critic 累积得到的进度序列
- `latency_ms`：单次请求耗时

## 5. Agent 推荐调用流程
1) `GET /health` 等待 `model_loaded=true`
2) 构造 `image + reference_image + task_description`
3) 发 `POST /critic`
4) 将 `critic_list/value_list` 作为上层策略输入

## 6. 常见错误与处理
- `400 Invalid image payload`：图像编码/格式不对
- `400 task_description cannot be empty`
- `503 Model not initialized`：模型还在加载
- `500 Critic inference failed`：模型推理异常，可重试或降载

建议：
- 首次启动后先做一次 `/health` 轮询
- 对 VLAC 请求设置较高超时（60~180s）

## 7. 最小可用请求示例
```bash
curl -X POST http://127.0.0.1:8014/critic \
  -H 'Content-Type: application/json' \
  -d '{"image":"<base64>","reference_image":"<base64>","task_description":"Scoop the rice into the rice cooker.","batch_num":1,"rich":false}'
```
