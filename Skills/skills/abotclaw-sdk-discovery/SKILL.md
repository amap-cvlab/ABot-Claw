---
name: abotclaw-sdk-discovery
description: Discover how a robot is actually used by reading its deployed guide and SDK reference before writing code. Use when starting robot work, when the API is unclear, when a new robot is added, or when OpenClaw must determine the real robot usage pattern instead of guessing.
---

# AbotClaw SDK Discovery

This skill owns one question:

**How should OpenClaw discover the real usage pattern of a robot before writing code?**

## Responsibility Boundary

This skill is the single place for:

- finding the robot base URL
- querying the robot guide
- querying the robot SDK reference
- summarizing how the robot is actually used

Other files should reference this skill instead of duplicating SDK discovery instructions.

## Default Discovery Path

For AbotClaw, discovery should start from known service and robot configuration, not from broad local searching.

Use this order:

1. read `service.md`
2. read `ROBOT.md`
3. identify the correct robot base URL
4. query the getting-started guide
5. query the SDK reference
6. summarize the usage pattern
7. only then write code

## Canonical Queries

### 1. Getting-started guide

```bash
curl -s -L <ROBOT_BASE_URL>/docs/guide/html
```

### 2. SDK reference

```bash
curl -s -L <ROBOT_BASE_URL>/code/sdk/markdown
```

### 3. Optional health check

If the deployment exposes a health endpoint, check it before deeper calls:

```bash
curl -s <ROBOT_BASE_URL>/health
```

## What to Extract

Before writing robot code, determine:

- how to connect
- how to read cameras
- how to read robot pose/state
- how to send actions or commands
- how to stop safely
- what the smallest safe smoke test should be

## Deliverable

Before writing code, summarize the real usage pattern in plain terms.

A good summary should answer:

- what is the correct base URL?
- how do I get the camera frame?
- how do I read state or pose?
- how do I send commands?
- how do I stop safely?
- what is the first safe test?

## Fallback Rule

Only use broader fallback sources when the guide or SDK reference is missing or clearly incomplete.

Fallback sources may include:

- vendor docs
- existing deployment examples
- operator-provided notes

Do not treat those as the default path.

## Rule

Do not invent methods because they sound plausible. If the deployed docs do not show a method, treat it as unknown until confirmed.
