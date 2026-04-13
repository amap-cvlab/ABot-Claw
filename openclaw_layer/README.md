# AbotClaw OpenClaw Setup

This setup adapts OpenClaw into a **hardware-first skill agent** for the AbotClaw robot fleet. Instead of centering the workflow around a single mobile manipulator and a simulator, it assumes openclaw can operat multiple real robots with different strengths.

## Quick Start

```bash
# Merge AbotClaw workspace files into an existing OpenClaw setup
./setup.sh

# Or rebuild the OpenClaw workspace, then apply AbotClaw files
./setup.sh --fresh
```

## Modes

**Integrate mode** keeps your existing OpenClaw sessions, memory, and identity files, then layers AbotClaw-specific mission prompts and hardware notes on top.

**Fresh mode** rebuilds the workspace from a clean state, preserves auth/config where possible, and then installs the AbotClaw templates.

## What This Setup Changes

It will:

- Install OpenClaw if needed
- Copy AbotClaw mission and robot-profile files into `~/.openclaw/workspace/`
- Keep the workspace aligned with a **real-hardware, multi-robot workflow**
- Restart the OpenClaw gateway so the changes take effect

## Included Workspace Files

```text
workspace/
├── MISSION.md      # Multi-robot mission and decision rules
├── ROBOT.md        # Fleet overview: Piper + Unitree G1 + Unitree Go2
├── HEARTBEAT.md    # Periodic checks for skill progress and fleet readiness
├── skills/         # AbotClaw-specific hardware and skill workflows
└── docs/           # SDK discovery and operator reference notes
```

## Operational Philosophy

The agent should not assume one universal embodiment. It should decide:

- Should this task be done by Piper, G1, or Go2?
- Is the task a perception task, a locomotion task, a manipulation task, or a coordinated task?
- Is there already a skill that can be reused with minimal adaptation?
- Is this safe to run directly on hardware right now?

## How the Agent Should Query Robot SDKs

The agent should not vaguely "look for docs". It should try concrete queries.

### First try robot-hosted docs

```bash
curl -s -L http://<ROBOT_IP>:8880/docs/guide/html
curl -s -L http://<ROBOT_IP>:8880/code/sdk/markdown
```

These correspond to:

1. getting-started guide
2. SDK reference with methods/classes

If your robot uses another host or port, replace `<ROBOT_IP>:8880` accordingly.

### Expected Outcome

After setup, the agent should behave less like a generic coding assistant and more like a robot skill operator for your own hardware stack.