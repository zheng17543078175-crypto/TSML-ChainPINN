"""Check local documentation links and synchronized paper defaults."""
import json
from pathlib import Path
import re
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    for doc in root.rglob("*.md"):
        if any(p in {"build", ".pytest_cache", "outputs", ".venv"} for p in doc.parts):
            continue
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            target = link.split("#")[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (doc.parent/target).exists():
                errors.append(f"{doc.relative_to(root)}: broken link {target}")
    repo = json.loads((root/"configs/paper.json").read_text())
    bundled = json.loads((root/"src/tsml_chainpinn/defaults/paper.json").read_text())
    if repo != bundled:
        errors.append("Packaged defaults differ from configs/paper.json")
    required = ["LICENSE", "CITATION.cff", "README.md", "README_zh-CN.md", "CHANGELOG.md",
                ".github/workflows/tests.yml", "docs/VALIDATION.md"]
    errors += [f"Missing {p}" for p in required if not (root/p).is_file()]
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("Release checks passed: local links, paper defaults, repository metadata.")


if __name__ == "__main__":
    main()
