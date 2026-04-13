---

## summary: "AbotClaw robot fleet reference"
read_when:
  - Bootstrapping a workspace manually

# ROBOT.md - About the Robot Fleet

This workspace is built for a **multi-robot fleet**, not a single embodiment.

## Fleet Connection Placeholders

Fill only the **Base URL** column. The guide and SDK URLs are derived from it.


| Robot       | Base URL           | Getting-started guide              | SDK reference                        |
| ----------- | ------------------ | ---------------------------------- | ------------------------------------ |
| Piper       | `<PIPER_BASE_URL>` | `<PIPER_BASE_URL>/docs/guide/html` | `<PIPER_BASE_URL>/code/sdk/markdown` |
| Unitree G1  | `<G1_BASE_URL>`    | `<G1_BASE_URL>/docs/guide/html`    | `<G1_BASE_URL>/code/sdk/markdown`    |
| Unitree Go2 | `<GO2_BASE_URL>`   | `<GO2_BASE_URL>/docs/guide/html`   | `<GO2_BASE_URL>/code/sdk/markdown`   |


### Recommended placeholders

- `PIPER_BASE_URL=http://<PIPER_HOST>:<PIPER_PORT>`
- `G1_BASE_URL=http://<G1_HOST>:<G1_PORT>`
- `GO2_BASE_URL=http://<GO2_HOST>:<GO2_PORT>`

## Fleet Overview

### 1. Piper

- Role: fixed-base manipulation
- Strengths: stable tabletop reach, repeatable grasping, station-based operation
- Best for: picking, placing, sorting, pressing, tool interaction near a fixed workcell
- Limits: no mobility, workspace constrained by mounting position and arm reach

### 2. Unitree G1

- Role: humanoid interaction and whole-body task execution
- Strengths: human-scale reach, upright embodiment, richer interaction possibilities
- Best for: tasks designed around human environments, upper-body interaction, demonstrations, teleop-assisted sequences
- Limits: balance, whole-body safety, more complex motion planning, higher execution risk than a fixed arm

### 3. Unitree Go2

- Role: mobile scouting and environmental coverage
- Strengths: locomotion, patrol, following, mobile sensing, inspection from place to place
- Best for: navigation, scene checks, route traversal, remote observation, bringing perception closer to a target area
- Limits: not a primary precision manipulator, mobility safety and terrain constraints must be considered

## Operating Principle

Treat these robots as complementary:

- Use **Go2** to go somewhere, inspect, or gather context
- Use **G1** when a task needs human-like embodiment or richer interaction
- Use **Piper** when the task can be brought to a stable manipulation station

## Important Differences from Single-Robot Setups

- Do not assume one shared SDK shape across all robots
- Do not assume one common camera arrangement
- Do not assume one common coordinate frame
- Do not assume one shared safety envelope

## SDK Discovery Rule

Do not duplicate SDK query logic here.

For any robot-facing coding task:

1. Fill in the correct base URL above
2. Use `abotclaw-sdk-discovery` to find the real SDK/docs/examples
3. Only then write code

## Required Local Notes

Fill these in before serious deployment:

- Control/API endpoint for Piper:
- Control/API endpoint for Unitree G1:
- Control/API endpoint for Unitree Go2:
- Auth method / API key details:
- Camera list for each robot:
- E-stop / recovery procedure for each robot:
- Teleoperation fallback path:

## Skill Design Reminder

Every robot-facing skill should state:

1. Which robot it targets
2. What assumptions it makes about sensors and actuators
3. What safety checks should happen before execution
4. Whether it can run unattended or needs supervision

