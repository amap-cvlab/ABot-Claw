# Usage

Typical examples:

```bash
python scripts/abotclaw-bundle.py my-skill
python scripts/abotclaw-bundle.py my-skill -o bundled.py
python scripts/abotclaw-bundle.py my-skill --call 'main()' -o bundled.py
```

Expected behavior:

1. Read target skill
2. Resolve dependency graph from `deps.txt`
3. Inline required code in dependency-safe order
4. Emit one runnable file

If an execution environment supports multiple files well, bundling is optional rather than mandatory.
