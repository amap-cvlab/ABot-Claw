#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FRESH=false
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=true ;;
        --help|-h)
            echo "Usage: setup.sh [--fresh]"
            echo ""
            echo "  (default)  Merge AbotClaw files into an existing OpenClaw workspace"
            echo "  --fresh    Rebuild workspace, sessions, and memory, then apply AbotClaw templates"
            exit 0 ;;
    esac
done

echo "=== AbotClaw OpenClaw Setup ==="
if $FRESH; then echo "Mode: fresh"; else echo "Mode: integrate"; fi
echo

if ! command -v openclaw &> /dev/null; then
    echo "Installing OpenClaw..."
    curl -fsSL https://openclaw.ai/install.sh | bash
    echo
fi

if [ ! -f ~/.openclaw/openclaw.json ]; then
    echo "Running OpenClaw onboarding..."
    openclaw onboard --install-daemon
    echo
fi

if $FRESH; then
    echo "Resetting workspace and memory..."
    rm -rf ~/.openclaw/workspace/ 2>/dev/null || true
    rm -rf ~/.openclaw/agents/main/sessions/ 2>/dev/null || true
    rm -f ~/.openclaw/memory/main.sqlite 2>/dev/null || true

    echo "Regenerating default workspace files..."
    TEMPLATE_DIR="$(npm root -g)/openclaw/docs/reference/templates"
    if [ -d "$TEMPLATE_DIR" ]; then
        mkdir -p ~/.openclaw/workspace
        for f in AGENTS.md SOUL.md TOOLS.md IDENTITY.md USER.md BOOTSTRAP.md; do
            if [ -f "$TEMPLATE_DIR/$f" ]; then
                sed '1{/^---$/!b};1,/^---$/d' "$TEMPLATE_DIR/$f" | sed '/./,$!d' > ~/.openclaw/workspace/"$f"
            fi
        done
        echo "  Copied default templates from OpenClaw package"
    else
        echo "  Warning: OpenClaw templates not found at $TEMPLATE_DIR"
    fi
fi

echo "Copying AbotClaw workspace files..."
mkdir -p ~/.openclaw/workspace ~/.openclaw/workspace/skills ~/.openclaw/workspace/docs
cp "$SCRIPT_DIR/workspace/MISSION.md" ~/.openclaw/workspace/
cp "$SCRIPT_DIR/workspace/ROBOT.md" ~/.openclaw/workspace/
cp "$SCRIPT_DIR/workspace/HEARTBEAT.md" ~/.openclaw/workspace/
if [ -d "$SCRIPT_DIR/workspace/skills" ]; then
    cp -R "$SCRIPT_DIR/workspace/skills/." ~/.openclaw/workspace/skills/
fi
if [ -d "$SCRIPT_DIR/workspace/docs" ]; then
    cp -R "$SCRIPT_DIR/workspace/docs/." ~/.openclaw/workspace/docs/
fi
echo "  Copied MISSION.md, ROBOT.md, HEARTBEAT.md, skills/, docs/"

echo "Patching AGENTS.md with AbotClaw session checklist..."
python3 << 'PATCHEOF'
import os, sys

path = os.path.expanduser("~/.openclaw/workspace/AGENTS.md")
if not os.path.exists(path):
    print("  Warning: AGENTS.md not found, skipping patch")
    sys.exit(0)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

if "Read `MISSION.md`" in content and "robot fleet" in content:
    print("  AGENTS.md already patched, skipping")
    sys.exit(0)

insert_block = (
    '3. Read `MISSION.md` — this defines how you operate as a skill agent for a real multi-robot fleet\n'
    '4. Read `ROBOT.md` — this describes the available embodiments: Piper, Unitree G1, and Unitree Go2\n'
    '5. Identify which robot best fits the current task before proposing code or actions\n'
)

content = content.replace(
    "2. Read `USER.md` — this is who you're helping\n3. Read `memory/",
    "2. Read `USER.md` — this is who you're helping\n" + insert_block + "6. Read `memory/"
)
content = content.replace(
    "4. **If in MAIN SESSION** (direct chat with your human): Also read `MEMORY.md`",
    "7. **If in MAIN SESSION** (direct chat with your human): Also read `MEMORY.md`"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("  Patched AGENTS.md")
PATCHEOF

echo "Restarting gateway..."
openclaw gateway restart 2>/dev/null || openclaw gateway start

echo
echo "=== Setup Complete ==="
echo "Open a chat or run: openclaw dashboard"
