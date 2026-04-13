---
name: abotclaw-bundle
description: Bundle an AbotClaw robot skill and its dependencies into a single executable Python file or portable submission artifact. Use when preparing robot code for execution environments that want one file, when flattening dependency chains, or when packaging a skill for deployment or review.
---

# AbotClaw Bundle

Bundle robot skill code into a single portable artifact when the runtime or review flow benefits from one-file submission.

## Use Cases

- execution endpoints that accept one script
- deployment handoff for a single robot task
- review / audit of the exact code that will run
- flattening `deps.txt` style dependency chains

## Expected Skill Layout

```text
skill-name/
├── SKILL.md
└── scripts/
    ├── main.py
    └── deps.txt
```

## Workflow

1. Read `scripts/deps.txt` recursively if present.
2. Resolve dependency order before concatenation.
3. Keep imports and helper definitions deduplicated.
4. Ensure the final script has one clear entrypoint.
5. Prefer predictable output over clever rewriting.

## Important Rule

Only bundle dependencies that are actually required by the target task. Do not drag in half the workspace just because it exists.

For implementation details, see `references/usage.md`.
