---
summary: "Heartbeat tasks for AbotClaw hardware-first robot skill work"
read_when:
  - Bootstrapping a workspace manually
---

# HEARTBEAT.md

Rotate through these checks instead of repeating the same one every time:

### 1. Fleet Readiness
- Are Piper, G1, and Go2 endpoint notes filled in?
- Are there missing auth details, camera mappings, or safety notes that should be documented?
- Are there robot-specific assumptions still buried in old code or prompts?

### 2. Skill Backlog Review
- Check which requested tasks are still unowned
- For each task, decide whether it belongs to Piper, G1, Go2, or a multi-stage workflow
- Flag tasks that are underspecified and need hardware/API clarification
- If a robot interface is still vague, prioritize discovering the real SDK/docs/examples before suggesting implementation

### 3. Skill Refactor Opportunities
- Look for skills copied from single-robot assumptions
- Suggest how to split them into embodiment-agnostic logic plus robot-specific adapters
- Prefer structural cleanup over cosmetic rewrites

### 4. Hardware-First Safety Review
- Identify any skill idea that sounds unsafe to run directly on real hardware
- Call out missing preconditions, confirmation steps, or recovery paths
- Recommend supervised testing before unattended runs when needed

---

# If nothing above needs attention, reply HEARTBEAT_OK.
