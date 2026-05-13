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

    uv run build.py              # stage + regen README + render → dist/
    uv run build.py --stage-only # stage only (.mdbook/), no mdbook
    uv run build.py --no-readme  # skip the README regeneration step
"""
from __future__ import annotations

import argparse
import os
import re
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

# Callout type → (icon filename in book/icons/, display label).
# `> [!NOTE]` blocks in the source markdown get rewritten into HTML tables
# that load these icons. GitHub renders the source `> [!NOTE]` natively;
# mdbook (after our preprocessor) renders the HTML table with the mouse icon.
CALLOUTS = {
    "NOTE":    ("note_assumptions.webp",    "Note"),
    "TIP":     ("tip_simplify.webp",        "Tip"),
    "WARNING": ("warning_units.webp",       "Warning"),
}

_CALLOUT_RE = re.compile(
    r'^> \[!(?P<type>NOTE|TIP|WARNING)\][ \t]*\n'
    r'(?P<body>(?:^>(?:[ \t].*)?\n)*)',
    re.MULTILINE,
)


def _render_callout(match: re.Match[str], icons_prefix: str) -> str:
    ctype = match.group("type")
    icon_file, label = CALLOUTS.get(ctype, (None, None))
    if icon_file is None:
        return match.group(0)
    body_lines = match.group("body").splitlines()
    body = "\n".join(
        line[2:] if line.startswith("> ")
        else line[1:] if line.startswith(">")
        else line
        for line in body_lines
    ).strip()
    return (
        f'<table class="callout callout-{ctype.lower()}" '
        f'style="width: 100%; border-left: 3px solid var(--quote-border, #aaa); '
        f'background: var(--quote-bg, #f9f9f9); color: var(--fg, inherit); margin: 1em 0;">\n'
        f'<tr>\n'
        f'<td style="width: 110px; vertical-align: top; padding: 0.6em;">\n'
        f'<img src="{icons_prefix}icons/{icon_file}" alt="{label}" '
        f'style="max-width: 96px; max-height: 96px; display: block; margin: 0 auto;">\n'
        f'</td>\n'
        f'<td style="vertical-align: top; padding: 0.6em 0.8em;">\n\n'
        f'**{label}** — {body}\n\n'
        f'</td>\n'
        f'</tr>\n'
        f'</table>\n'
    )


def _icons_prefix_for(md_path: Path, staging_root: Path) -> str:
    rel = md_path.relative_to(staging_root)
    depth = len(rel.parts) - 1
    return "../" * depth


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

    # 2b. Copy whole directories that chapters link into. Non-.md files in
    #     these trees get carried through by mdbook to dist/ as static assets,
    #     so the cross-links from chapter prose resolve. Without this step,
    #     every link like `../../code/measurement/foo.py` 404s on the rendered
    #     site even though the file is in the repo.
    for src_dir_rel, dst_dir_rel in [
        ("code/measurement", "code/measurement"),
        (".archive/simlog",  ".archive/simlog"),
    ]:
        src_dir = ROOT / src_dir_rel
        dst_dir = STAGING / dst_dir_rel
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    # 3. Rewrite cross-link paths and callout blocks in every staged file.
    for md in STAGING.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for old, new in PATH_REWRITES:
            text = text.replace(old, new)
        prefix = _icons_prefix_for(md, STAGING)
        text = _CALLOUT_RE.sub(lambda m: _render_callout(m, prefix), text)
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
    env = os.environ.copy()
    env["PATH"] = f"{LOCAL_CARGO_BIN}:{env.get('PATH', '')}"
    subprocess.run([str(mdbook), "build"], cwd=ROOT, env=env, check=True)


# --- README generation ---------------------------------------------------
#
# README.md at repo root is regenerated from the same sources as the website.
# It is the entire book in one file: front matter + chapter files concatenated,
# with cross-paths rewritten so everything resolves from the repo root.
#
# The reader who lands on the source repo gets the full book inline, GitHub-
# /Forgejo-rendered. The reader who wants the polished experience follows
# the live URL in the README header.

README_BEGIN = "<!-- BOOK_BEGIN -->"
README_END   = "<!-- BOOK_END -->"
LIVE_URL     = "https://root-11.codeberg.page/intro-book-python/"


def _slugify(heading: str) -> str:
    s = heading.lower().strip()
    s = re.sub(r"[*`]", "", s)
    s = re.sub(r"[^\w\s\-]", "", s)
    s = re.sub(r"\s", "-", s)
    return s


def _parse_summary() -> list[dict]:
    text = (ROOT / "book" / "SUMMARY.md").read_text(encoding="utf-8")
    out: list[dict] = []
    for line in text.splitlines():
        m = re.match(r"\s*-?\s*\[([^\]]+)\]\(([^)]+)\)", line)
        if not m:
            continue
        title, path = m.group(1), m.group(2)
        if path.startswith("../") or "_solutions" in path:
            continue
        if path.startswith("trunk/"):
            out.append({"kind": "chapter", "title": title, "path": path})
        else:
            out.append({"kind": "intro", "title": title, "path": path})
    return out


def _build_anchor_map(chapters: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in chapters:
        if entry["kind"] != "chapter":
            continue
        path = ROOT / "book" / entry["path"]
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
        if m:
            out[path.stem] = _slugify(m.group(1))
    return out


def _resolve_url(url: str, source: Path) -> str:
    if not url or url.startswith(("http://", "https://", "#", "/", "mailto:")):
        return url
    base, sep, frag = url.partition("#")
    if not base:
        return url
    try:
        full = (source.parent / base).resolve()
        rel = full.relative_to(ROOT)
        return f"{rel}{sep}{frag}"
    except (ValueError, OSError):
        return url


_HTML_IMG_SRC_RE = re.compile(r'(<img\s+[^>]*?src=")([^"]+)(")', re.DOTALL)
_MD_LINK_RE      = re.compile(r'(!?\[(?:[^\[\]]|\[[^\]]*\])*\]\()([^()\s]+)(\))')
_CHAPTER_LINK_RE = re.compile(r"\[([^\]]+)\]\(([0-9]+_[a-z0-9_]+)\.md(#[\w-]+)?\)")


def _rewrite_chapter_links(text: str, anchor_map: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        label, stem, frag = m.group(1), m.group(2), m.group(3) or ""
        if stem.endswith("_solutions"):
            return f"[{label}]({LIVE_URL}trunk/{stem}.html{frag})"
        anchor = anchor_map.get(stem)
        if not anchor:
            return m.group(0)
        return f"[{label}](#{anchor}{frag})"
    return _CHAPTER_LINK_RE.sub(repl, text)


def _rewrite_paths(text: str, source: Path) -> str:
    text = _HTML_IMG_SRC_RE.sub(
        lambda m: m.group(1) + _resolve_url(m.group(2), source) + m.group(3),
        text,
    )
    text = _MD_LINK_RE.sub(
        lambda m: m.group(1) + _resolve_url(m.group(2), source) + m.group(3),
        text,
    )
    return text


_SKIP_RE             = re.compile(r"<!-- START_SKIP_FOR_README -->.*?<!-- STOP_SKIP_FOR_README -->\s*\n?", re.DOTALL)
_CONCEPT_NODE_RE     = re.compile(r"^> \*Concept node:[^\n]*\n", re.MULTILINE)
_REFERENCE_NOTES_RE  = re.compile(r"^Reference notes in \[[^\]]+\]\([^)]+\)\.[^\n]*\n", re.MULTILINE)


def _render_for_readme(path: Path, anchor_map: dict[str, str], strip_h1: bool) -> str:
    text = path.read_text(encoding="utf-8")
    text = _SKIP_RE.sub("", text)
    text = _CONCEPT_NODE_RE.sub("", text)
    text = _REFERENCE_NOTES_RE.sub("", text)
    text = _rewrite_chapter_links(text, anchor_map)
    text = _rewrite_paths(text, path)
    if strip_h1:
        text = re.sub(r"^# [^\n]*\n+", "", text, count=1)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip() + "\n"


def generate_readme() -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        print(f"README.md not found — skipping README generation")
        return
    text = readme.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        print(f"README.md missing {README_BEGIN}/{README_END} markers — skipping README generation")
        return

    entries = _parse_summary()
    chapters = [e for e in entries if e["kind"] == "chapter"]
    intros   = [e for e in entries if e["kind"] == "intro"]
    anchor_map = _build_anchor_map(chapters)

    parts = []
    for i, entry in enumerate(intros):
        parts.append(_render_for_readme(
            ROOT / "book" / entry["path"], anchor_map, strip_h1=(i == 0)
        ))
    for entry in chapters:
        parts.append(_render_for_readme(ROOT / "book" / entry["path"], anchor_map, strip_h1=False))

    body = "\n\n".join(parts).rstrip() + "\n"

    pre, _, rest = text.partition(README_BEGIN)
    _, _, post = rest.partition(README_END)
    new_text = (
        f"{pre}{README_BEGIN}\n\n"
        f"<!-- This block is generated by build.py — do not edit by hand. -->\n\n"
        f"{body}\n"
        f"{README_END}{post}"
    )
    readme.write_text(new_text, encoding="utf-8")
    print(f"Regenerated README.md ({len(chapters)} chapter(s) inserted)")


# --- Dist-side README (for Codeberg repo view + SEO) ---------------------

DIST = ROOT / "dist"

_DIST_README_LIVE_LINK_RE = re.compile(
    r"(\]\()((?:concepts|code)/[^)\s]+\.md(?:#[^)\s]*)?)(\))"
)


def stage_readme_in_dist() -> None:
    src = ROOT / "README.md"
    if not src.exists() or not DIST.exists():
        return
    text = src.read_text(encoding="utf-8")
    for sub in ("illustrations", "covers", "icons", "simlog"):
        text = text.replace(f'"book/{sub}/', f'"{sub}/')
        text = text.replace(f'](book/{sub}/', f']({sub}/')
    def to_live(m: re.Match[str]) -> str:
        path = m.group(2).replace(".md", ".html")
        return f"{m.group(1)}{LIVE_URL}{path}{m.group(3)}"
    text = _DIST_README_LIVE_LINK_RE.sub(to_live, text)
    (DIST / "README.md").write_text(text, encoding="utf-8")
    for name in ("LICENSE", "LICENSE-CC-BY-4.0", "LICENSE-MIT", "LICENSE-APACHE-2.0"):
        p = ROOT / name
        if p.exists():
            shutil.copy(p, DIST / name)
    print(f"Wrote dist/README.md")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage-only", action="store_true", help="skip the mdbook render step")
    p.add_argument("--no-readme",  action="store_true", help="skip README.md regeneration")
    args = p.parse_args()
    stage()
    if not args.no_readme:
        generate_readme()
    if not args.stage_only:
        render()
        stage_readme_in_dist()


if __name__ == "__main__":
    main()
