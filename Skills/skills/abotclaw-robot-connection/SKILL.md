---
name: abotclaw-robot-connection
description: Robot connection and reachability guidance for the AbotClaw fleet. Use when identifying the correct robot base URL, auth method, health check, safest first connectivity test, or when debugging why Piper, Unitree G1, or Unitree Go2 cannot be reached.
---

# AbotClaw Robot Connection

This skill is about connection and reachability.

It does **not** own full SDK discovery. When you need to learn how a robot is actually used, use `abotclaw-sdk-discovery`.

## Robots Covered

- Piper
- Unitree G1
- Unitree Go2

## Responsibility Boundary

This skill owns:

- the correct base URL
- the auth method
- the health check or reachability test
- the safest first connectivity check
- debugging connection failures

This skill does not own:

- guide-reading workflow
- SDK method discovery
- example discovery
- detailed robot usage patterns

## What to Determine

For the target robot, determine:

1. the correct base URL
2. whether authentication is required
3. the available health or status endpoint
4. the safest first query to confirm connectivity
5. any stop/recovery or operator-facing safety note relevant to connection troubleshooting

## Default Workflow

1. Read `service.md`
2. Read `ROBOT.md`
3. Identify the correct robot base URL
4. Check whether a health endpoint exists
5. Run the safest possible connectivity test
6. If the robot is reachable, hand off to `abotclaw-sdk-discovery` for usage discovery

## Good Outcome

A good result from this skill is a short robot-specific connection summary:

- base URL
- auth requirement
- health or status check
- first safe connectivity test
- special notes for troubleshooting

## Rule

Do not assume every robot has the same endpoint layout. Confirm what is actually exposed before relying on it.
