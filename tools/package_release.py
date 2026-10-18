"""Build a clean source ZIP; exclude caches, data, training outputs and weights."""
import hashlib
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    directories = ("src/tsml_chainpinn", "configs", "docs", "examples", "tests", "tools", ".github")
    excluded = {"__pycache__", ".pytest_cache", "build", "dist", ".git"}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix in {".md", ".toml", ".txt", ".cff", ".in"}]
    files += [root/"LICENSE", root/".gitignore"]
    for name in directories:
        files += [p for p in (root/name).rglob("*") if p.is_file()
                  and not any(part in excluded for part in p.relative_to(root).parts)
                  and p.suffix not in {".pyc", ".pyo", ".pt", ".pth", ".npz"}]
    dest = root.parent/"TSML-ChainPINN-open-source-v0.3.0.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(set(files)):
            archive.write(path, str(Path("TSML-ChainPINN")/path.relative_to(root)))
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix(".zip.sha256").write_text(f"{digest}  {dest.name}\n")
    print(f"{len(set(files))} files, {dest.stat().st_size:,} bytes: {dest}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
