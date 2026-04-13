---
name: abotclaw-progress-critic
description: Use a deployed VLAC-style vision-language-action critic service to evaluate task progress, compare current observations against a reference image, and judge task completion from robot camera frames. Use when the agent needs external progress supervision, completion verification, failure detection, or image-based task-state comparison for Piper, Unitree G1, or Unitree Go2.
---

# AbotClaw Progress Critic

Use this skill when a robot task needs an external judge instead of relying only on hand-written heuristics.

This skill is about **using** an already deployed critic service, not deploying the service.

## What This Service Does

The VLAC critic can compare:

- a **current frame**
- a **reference frame**
- a **task description**

and return progress or completion-related judgment.

This is useful for:

- task progress estimation
- task completion verification
- detecting failed or unchanged task state
- deciding whether the robot should continue, retry, or stop

## Service Contract

For the FastAPI service in this stack:

- endpoint: `POST /critic`
- required inputs:
  - `image`
  - `reference_image`
  - `task_description`

Important:

- `image` and `reference_image` must be sent in the **same request**
- there is **no separate reference-image cache/upload endpoint**

## Minimal Request Shape

```json
{
  "image": "<base64_or_url_or_path>",
  "reference_image": "<base64_or_url_or_path>",
  "task_description": "..."
}
```

## Responsibility Boundary

This skill owns:

- when to use the critic
- how to call `/critic`
- how to prepare current image + reference image + task description
- how to use critic results to decide continue / retry / stop

This skill does not own robot SDK discovery. Use `abotclaw-sdk-discovery` to learn how each robot provides camera frames.

## When to Use It

Use the critic when:

- the robot needs a visual completion check
- hand-authored success conditions are unreliable
- the user asks "is it done?" or "did that succeed?"
- a task needs step-wise supervision from images
- you want to compare current state against a known target state

## Standard Workflow

1. Use `abotclaw-sdk-discovery` to learn how the target robot exposes camera frames.
2. Capture a current frame from Piper, G1, or Go2.
3. Obtain a reference image that represents the desired or comparison state.
4. Write a task description that matches the intended goal.
5. Call the critic service with all three inputs in one request.
6. Interpret the response as supervision for task control.

## Current Image Sources

The current observation can come from any robot in the fleet:

- Piper camera
- G1 camera
- Go2 camera

Choose the camera that best reflects task progress.

Examples:

- Piper wrist or workcell camera for tabletop manipulation
- G1 head or chest camera for humanoid interaction tasks
- Go2 forward camera for inspection or navigation-adjacent tasks

## Reference Image Sources

A reference image may come from:

- a successful prior run
- a user-provided target image
- a recorded frame from a known-good final state
- a memory/evidence image from `abotclaw-memory`

## Example Request

Get the correct host and port from `service.md`.

```bash
curl -s -X POST <VLAC_BASE_URL>/critic \
  -H 'Content-Type: application/json' \
  -d '{
    "image":"<current_frame>",
    "reference_image":"<reference_frame>",
    "task_description":"Put the bowl back into the white storage box."
  }'
```

## Input Preparation Rules

### `image`

Use the robot's current frame.

### `reference_image`

Use an image that represents the expected target state or a meaningful comparison state.

### `task_description`

Keep it concrete and visual.

Better:
- "Put the bowl back into the white storage box."
- "Place the bottle upright on the tray."

Worse:
- "Do the task correctly."
- "Finish it."

## How to Use the Result

Treat the critic output as task supervision, not absolute truth.

Possible uses:

- if the critic indicates strong completion -> stop or hand back success
- if the critic indicates partial progress -> continue
- if the critic indicates failure or no change -> retry, replan, or ask for help
- if the critic disagrees with sensor heuristics -> inspect evidence before acting further

## Multi-Robot Usage Pattern

The image source does not have to come from the same robot that originally planned the task.

Examples:

- Go2 scouts a scene, then the critic evaluates whether the inspected target matches expectation
- Piper manipulates an object, and its current frame is checked against a reference finish state
- G1 performs a human-environment interaction task, and the critic judges completion from G1's current view

## Integration with Memory

This skill works well with `abotclaw-memory`:

- use memory to retrieve a prior successful evidence image as `reference_image`
- use the critic to compare the current scene against remembered success state
- use critic output plus memory result to decide whether to navigate, manipulate, or stop

## Behavioral Rule

Use the critic to reduce ambiguity in real-world execution, especially when success is easier to see than to hand-code.

Do not pretend the critic is a robot controller. It is a supervisor and evaluator.
