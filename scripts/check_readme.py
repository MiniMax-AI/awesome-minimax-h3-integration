"""Check README.md and README_cn.md before a pull request.

usage: python3 scripts/check_readme.py
  1. README.md and README_cn.md link to the same set of URLs
  2. every #anchor link in README.md, README_cn.md and docs/*.md resolves (GitHub slug rules)
"""
import re, sys, unicodedata
from pathlib import Path
from urllib.parse import unquote

root = Path(__file__).resolve().parent.parent
files = {p.resolve(): p.read_text() for p in [root / "README.md", root / "README_cn.md", *sorted((root / "docs").glob("*.md"))]}
URL = re.compile(r'https?://[^\s)"\'<>\]]+')


def slug(h):
    h = re.sub(r"<[^>]+>", "", h).strip().lower()
    return "".join("-" if c == " " else c for c in h if c in "-_ " or unicodedata.category(c)[0] in "LNM")


def anchors(text):
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    out, seen = set(re.findall(r'<a id="([^"]+)"', body)), {}
    for h in re.findall(r"^#{1,6}\s+(.+)$", body, re.M):
        s = slug(h); n = seen.get(s, 0); seen[s] = n + 1
        out.add(s if n == 0 else f"{s}-{n}")
    return out


bad = 0
urls = [{u.rstrip(".,;").rstrip("/") for u in URL.findall(files[(root / f).resolve()])} for f in ("README.md", "README_cn.md")]
for u in sorted(urls[0] - urls[1]):
    print("only in README.md:", u); bad += 1
for u in sorted(urls[1] - urls[0]):
    print("only in README_cn.md:", u); bad += 1

for p, text in files.items():
    for link in re.findall(r'\]\(([^)\s]*#[^)\s]+)\)', text) + re.findall(r'href="([^"]*#[^"]+)"', text):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, frag = link.partition("#")
        tp = (p.parent / target).resolve() if target else p
        if tp not in files:
            if not tp.exists():
                print(f"{p.relative_to(root)}: missing file {target}"); bad += 1
            continue
        if unquote(frag) not in anchors(files[tp]):
            print(f"{p.relative_to(root)}: missing anchor {link}"); bad += 1

print("OK" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
