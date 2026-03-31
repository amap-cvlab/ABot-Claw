#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Bundle an AbotClaw skill into a single file")
    parser.add_argument("skill_name")
    parser.add_argument("-o", "--output")
    parser.add_argument("--skills-dir", default=".")
    parser.add_argument("--call")
    args = parser.parse_args()

    skill_dir = Path(args.skills_dir) / args.skill_name
    main_py = skill_dir / "scripts" / "main.py"
    if not main_py.exists():
        raise SystemExit(f"main.py not found: {main_py}")

    content = main_py.read_text(encoding="utf-8")
    if args.call:
        content += f"\n\n# injected call\n{args.call}\n"

    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
