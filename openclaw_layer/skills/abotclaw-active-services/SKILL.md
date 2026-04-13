---
name: abotclaw-active-services
description: External services and backend capabilities for AbotClaw robots — vision APIs, VLM/LLM endpoints, planners, speech services, grasping backends, and client SDKs. Use when a robot skill needs an external service, when checking what backend tools already exist, when debugging service reachability, or when deciding whether to build locally versus call a service.
---

# AbotClaw Active Services

Treat external services as optional accelerators, not assumptions.

## What to Look For

When a task may need outside help, check whether any of these already exist:

- vision / detection service
- segmentation / grounding service
- speech / ASR / TTS service
- planner or policy inference service
- grasp scoring service
- map / navigation backend
- robot-specific bridge service
- Python client SDK for any of the above

## Workflow

1. Identify the missing capability in the requested robot task.
2. Check local docs, repos, or service catalogs first.
3. If a service exists, prefer using its documented client SDK or official API shape.
4. Record the dependency explicitly in the skill so future edits know the skill is not standalone.
5. If no service exists, say so clearly instead of pretending one does.

## Rule

Do not invent service endpoints. Discover them from the actual system, docs, repo, config, or operator notes.
