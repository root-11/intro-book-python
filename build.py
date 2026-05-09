#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Stage and render the Python-edition book.

mdbook can only see files inside its `src` directory. Our canonical sources
live outside `book/` (in `concepts/` and `code/`), so this script stages
everything into `.mdbook/` with cross-link paths adjusted, then invokes the
locally-installed `mdbook` (in `.cargo/bin/`) to render `dist/`.

Run as:

    uv run build.py

To skip the mdbook render step (just stage), pass `--stage-only`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGING = ROOT / ".mdbook"
LOCAL_CARGO_BIN = ROOT / ".cargo" / "bin"

# (rel_src, rel_in_staging) — files copied verbatim from outside book/
EXTERNAL_FILES = [
    ("concepts/dag.md", "concepts/dag.md"),
    ("concepts/glossary.md", "concepts/glossary.md"),
    ("code/sim/SPEC.md", "code/sim/SPEC.md"),
]

# Path rewrites applied to staged markdown.
# Source files use the GitHub-friendly form (e.g. `../../concepts/dag.md`
# from `book/trunk/foo.md`); the staging tree flattens by one level so
# `concepts/` is a sibling of `trunk/`.
PATH_REWRITES = [
    ("../../concepts/", "../concepts/"),
    ("../../code/", "../code/"),
    ("../../.archive/", "../.archive/"),
]


def stage() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()

    # 1. Copy the entire book/ tree into staging.
    shutil.copytree(ROOT / "book", STAGING, dirs_exist_ok=True)

    # 2. Copy external sources into staging at their re-mapped locations.
    for src_rel, dst_rel in EXTERNAL_FILES:
        src = ROOT / src_rel
        dst = STAGING / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    # 3. Rewrite cross-link paths in every staged file.
    #    Source uses `../../X/` from `book/trunk/`; after staging,
    #    `trunk/` and `concepts/` are siblings, so `../X/` is correct.
    for md in STAGING.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for old, new in PATH_REWRITES:
            text = text.replace(old, new)
        md.write_text(text, encoding="utf-8")

    # 4. Adjust top-level book pages (SUMMARY.md, front_matter.md,
    #    nomenclature.md): source uses `../concepts/` and `../code/`
    #    (relative to book/, pointing at repo-root). After staging these
    #    files sit at the staging root, so the prefix becomes a no-op.
    for top_md in STAGING.glob("*.md"):
        text = top_md.read_text(encoding="utf-8")
        text = text.replace("../concepts/", "concepts/")
        text = text.replace("../code/", "code/")
        top_md.write_text(text, encoding="utf-8")

    n_md = sum(1 for _ in STAGING.rglob("*.md"))
    print(f"Staged {n_md} markdown file(s) to {STAGING.relative_to(ROOT)}")


def render() -> None:
    mdbook = LOCAL_CARGO_BIN / "mdbook"
    if not mdbook.exists():
        sys.exit(
            f"mdbook not found at {mdbook}.\n"
            f"Install with:\n"
            f"  cd {ROOT}\n"
            f'  CARGO_INSTALL_ROOT=$(pwd)/.cargo cargo install mdbook mdbook-mermaid --locked'
        )
    import os
    env = os.environ.copy()
    env["PATH"] = f"{LOCAL_CARGO_BIN}:{env.get('PATH', '')}"
    subprocess.run([str(mdbook), "build"], cwd=ROOT, env=env, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage-only", action="store_true", help="skip the mdbook render step")
    args = p.parse_args()
    stage()
    if not args.stage_only:
        render()


if __name__ == "__main__":
    main()
