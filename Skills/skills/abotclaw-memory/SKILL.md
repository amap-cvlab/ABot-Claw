---
name: abotclaw-memory
description: SpatialMemoryHub-based robot memory querying, writing, and retrieval for AbotClaw. Use when the agent needs to store or retrieve object memory, place memory, keyframe memory, or semantic frame memory; when checking whether a memory service is running; when using robot SDK camera frames to perform memory ingestion or memory-guided retrieval; or when grounding robot tasks in prior visual/spatial experience.
---

# AbotClaw Memory

Use this skill whenever robot behavior should rely on remembered places, objects, visual evidence, or past observations.

## What This Memory Module Is

SpatialMemoryHub is a unified robot memory service. It supports four memory types:

1. **object memory** — concrete detected objects with pose and evidence
2. **place memory** — named semantic places and anchors
3. **keyframe memory** — offline extracted keyframes for large-scene memory building
4. **semantic frame memory** — image memories retrievable by text semantics

## Service Location

Assume the SpatialMemoryHub server is already running.

Get the correct host and port from `service.md`.

This skill is about **using** the memory service, not deploying it.

## Always Check Health First

Before relying on memory, test the service:

```bash
curl -s <MEMORY_BASE_URL>/health
```

If the service is not healthy, do not pretend memory is available.

## Core Endpoints

### Write endpoints

```bash
POST /memory/object/upsert
POST /memory/place/upsert
POST /memory/semantic/ingest
POST /memory/keyframe/ingest-batch
```

### Query endpoints

```bash
POST /query/object
POST /query/place
POST /query/position
POST /query/semantic/text
POST /query/unified
```

### Task endpoints

```bash
POST /pipeline/tasks
GET /pipeline/tasks/{task_id}
```

## Unified Result Shape

All retrieval results should be read as navigation-usable memory hits. Expect fields such as:

- `memory_type`
- `target_pose`
- `confidence`
- `evidence`

## How to Query Memory

### Object memory

Use when the user asks things like:
- where is the cup?
- have you seen the remote?

Example:

```bash
curl -s -X POST <MEMORY_BASE_URL>/query/object \
  -H 'Content-Type: application/json' \
  -d '{"name":"cup"}'
```

### Place memory

Use when the user asks:
- where is the kitchen?
- go to the tool table

Example:

```bash
curl -s -X POST <MEMORY_BASE_URL>/query/place \
  -H 'Content-Type: application/json' \
  -d '{"place_name":"kitchen"}'
```

### Position radius query

Use when retrieving memory near a known point:

```bash
curl -s -X POST <MEMORY_BASE_URL>/query/position \
  -H 'Content-Type: application/json' \
  -d '{"x":1.0,"y":2.0,"radius":2.5}'
```

### Semantic text query

Use when the user describes a scene or object naturally:

```bash
curl -s -X POST <MEMORY_BASE_URL>/query/semantic/text \
  -H 'Content-Type: application/json' \
  -d '{"text":"a table with bottles and tools"}'
```

### Unified query

Use when multiple memory types may help:

```bash
curl -s -X POST <MEMORY_BASE_URL>/query/unified \
  -H 'Content-Type: application/json' \
  -d '{"text":"cup near kitchen table"}'
```

## How to Write Memory

### Object memory upsert

Use after visual detection plus pose estimation:

```bash
curl -s -X POST <MEMORY_BASE_URL>/memory/object/upsert \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"cup",
    "robot_id":"g1_001",
    "robot_type":"humanoid",
    "detect_confidence":0.93,
    "object_pose":{"x":1.2,"y":1.1,"z":0.8,"yaw":0.0,"frame_id":"map"}
  }'
```

### Place memory upsert

Use after human labeling or semantic anchoring:

```bash
curl -s -X POST <MEMORY_BASE_URL>/memory/place/upsert \
  -H 'Content-Type: application/json' \
  -d '{
    "place_name":"kitchen",
    "robot_id":"go2_001",
    "robot_type":"quadruped",
    "place_pose":{"x":4.0,"y":2.0,"z":0.0,"yaw":1.57,"frame_id":"map"},
    "note":"main food prep area"
  }'
```

### Semantic frame ingest

Use after obtaining an image plus note/tags:

```bash
curl -s -X POST <MEMORY_BASE_URL>/memory/semantic/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "robot_id":"go2_001",
    "robot_type":"quadruped",
    "note":"workbench with red screwdriver and black toolbox",
    "tags":["workbench","tools"]
  }'
```

## Robot SDK + Camera + Memory Workflow

This is critical: memory is not only text lookup. The agent may need to use the robot SDK to obtain a live camera frame, then use that frame for memory ingestion or retrieval.

### Standard workflow

1. Use the robot SDK or robot docs to learn how to fetch camera frames.
2. Capture a live frame from Piper, G1, or Go2.
3. If needed, save the frame locally as evidence.
4. Add semantic annotation, object detection result, or place label.
5. Write it into SpatialMemoryHub.
6. Later, query memory to recover `target_pose`, confidence, and evidence.

### Required rule

Before writing camera-related code, first use `abotclaw-sdk-discovery` to learn the real SDK usage pattern for that robot.

What matters here is the outcome of that discovery:

- how to read camera frames
- how to read robot pose
- how to access timestamp or frame metadata

Do not invent camera methods.

## Typical Memory-Driven Behaviors

### 1. Find an object from memory

- query `/query/object`
- read `target_pose`
- hand off `target_pose` to navigation or manipulation

### 2. Recall a semantic scene from description

- query `/query/semantic/text`
- inspect evidence and target pose
- optionally move robot to re-observe the scene

### 3. Build memory from current camera view

- get camera frame from robot SDK
- add note/tags or detections
- call `/memory/semantic/ingest` or `/memory/object/upsert`

### 4. Build memory offline from videos or patrol logs

- create a task with `/pipeline/tasks`
- poll `/pipeline/tasks/{task_id}`
- validate retrievability with query endpoints

## Responsibility Boundary

This skill owns:

- when to query memory
- which memory endpoint to use
- how memory results map to navigation or manipulation targets
- how live robot camera frames can become memory inputs

This skill does not own full robot SDK discovery. Use `abotclaw-sdk-discovery` for that.

## Navigation Handoff Rule

When memory returns a `target_pose`, treat that as a candidate navigation/manipulation goal, not guaranteed truth. Check confidence and evidence before executing risky actions.

## Storage Notes

- structured DB: `data/memory_hub.db`
- image evidence: `data/images/*.jpg`

## Behavioral Rule

A strong agent should combine:

- live robot perception,
- remembered spatial context,
- and explicit evidence from memory results.

Do not treat memory as magic. Treat it as a retrievable, inspectable world model.
