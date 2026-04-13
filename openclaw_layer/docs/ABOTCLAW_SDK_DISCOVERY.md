# ABOTCLAW_SDK_DISCOVERY.md

This note explains how OpenClaw should discover **how a robot is used** in the AbotClaw setup.

The goal is not to search everywhere blindly. The goal is to answer a practical question:

- How do I talk to this robot?
- How do I read its camera/state?
- How do I send actions safely?

## Core Rule

For AbotClaw, SDK discovery should start from the **known robot service address**, not from local filesystem guessing.

That means:

1. check `service.md`
2. get the robot base URL
3. query the robot guide
4. query the robot SDK reference
5. summarize the real usage pattern
6. only then write code

## Source of Truth

Use these files first:

- `service.md` — service host and ports
- `ROBOT.md` — robot list and base URL placeholders

Do not start with local repo scanning unless the remote docs are missing or incomplete.

## Standard Discovery Flow

For the target robot:

### 1. Identify the base URL

Find the robot base URL from `ROBOT.md` / `service.md`.

Examples:

- `PIPER_BASE_URL=http://<PIPER_HOST>:<PIPER_PORT>`
- `G1_BASE_URL=http://<G1_HOST>:<G1_PORT>`
- `GO2_BASE_URL=http://<GO2_HOST>:<GO2_PORT>`

### 2. Read the getting-started guide

```bash
curl -s -L <ROBOT_BASE_URL>/docs/guide/html
```

This should be the first read before writing robot code.

### 3. Read the SDK reference

```bash
curl -s -L <ROBOT_BASE_URL>/code/sdk/markdown
```

Use this to learn available methods, classes, and API shape.

### 4. Check basic reachability if needed

If the system exposes a health endpoint, check it before deeper calls.

Example:

```bash
curl -s <ROBOT_BASE_URL>/health
```

### 5. Extract the real control pattern

After reading the guide and SDK reference, summarize:

- how to connect
- how to read cameras
- how to read robot pose/state
- how to send actions or commands
- how to stop safely
- what the smallest safe smoke test is

## What Good SDK Discovery Produces

Before OpenClaw writes robot code, it should be able to say something like:

- "Piper should be accessed through `<PIPER_BASE_URL>`"
- "Camera frames are obtained using ..."
- "Robot state is read using ..."
- "Action commands are sent using ..."
- "The first safe smoke test should be ..."

If it cannot say that clearly, it is not ready to write code yet.

## When to Fall Back

Only fall back to broader searching when:

- the guide endpoint is missing
- the SDK reference endpoint is missing
- the docs are too incomplete to reveal the usage pattern

If that happens, use secondary sources such as:

- vendor docs
- existing robot-facing examples in your deployment
- operator-provided notes

But these are fallback paths, not the default.

## What Not to Do

Do not:

- invent SDK methods from intuition
- assume all robots share the same API shape
- skip the guide and go straight to writing code
- over-index on local repo/package scanning when the running robot already exposes docs

## Summary

In AbotClaw, SDK discovery should be simple and operational:

1. get base URL
2. read guide
3. read SDK reference
4. summarize usage
5. write minimal safe code

That is the default path.
