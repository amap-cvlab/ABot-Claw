---
name: abotclaw-robot-hardware
description: Hardware roles, embodiment boundaries, and task-fit guidance for Piper, Unitree G1, and Unitree Go2. Use when deciding which robot should handle a task, when reasoning about embodiment constraints, or when documenting robot-specific assumptions in a skill.
---

# AbotClaw Robot Hardware

## Piper

Use Piper for stable local manipulation, repeated pick/place, and workcell-style actions.

## Unitree G1

Use G1 for human-scale interaction, upright reach, and tasks that benefit from humanoid embodiment.

## Unitree Go2

Use Go2 for mobility, scouting, patrol, inspection, and scene acquisition.

## Decision Rule

Pick the robot whose embodiment naturally fits the task. Do not choose a more impressive robot when a simpler one is the safer and more reliable choice.

## Skill Requirement

Every robot-facing skill should state the intended robot and the assumptions it makes about sensors, workspace, and supervision.
