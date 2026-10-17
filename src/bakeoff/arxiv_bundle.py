"""Build the arXiv source bundle for paper/main.tex.

    python -m bakeoff.arxiv_bundle

Needs paper/main.bbl from a fresh build (tectonic -X compile --keep-intermediates main.tex). Copies main.tex, the files it
inputs, the ACL style files, main.bbl and every figure it includes (into figures/, which main.tex's \\graphicspath searches
first) to paper/arxiv/src, and packs them as paper/arxiv/arxiv-source.tar.gz. arXiv compiles with pdflatex and uses
main.bbl as shipped, so references.bib is left out.
"""

from __future__ import annotations

import re
import shutil
import tarfile

from bakeoff.protocol import ROOT

PAPER = ROOT / "paper"
FIGURES = ROOT / "docs" / "figures"
OUT = PAPER / "arxiv"
SRC = OUT / "src"


def main() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bbl = PAPER / "main.bbl"
    if not bbl.exists() or bbl.stat().st_mtime < (PAPER / "references.bib").stat().st_mtime:
        raise SystemExit("paper/main.bbl is missing or older than references.bib: rebuild with --keep-intermediates first")
    inputs = [f"{name}.tex" for name in re.findall(r"\\input\{([^}]+)\}", tex)]
    texts = tex + "".join((PAPER / f).read_text(encoding="utf-8") for f in inputs)
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", texts)))
    if SRC.exists():
        shutil.rmtree(SRC)
    (SRC / "figures").mkdir(parents=True)
    for name in ["main.tex", *inputs, "acl.sty", "acl_natbib.bst", "main.bbl"]:
        shutil.copy2(PAPER / name, SRC / name)
    for name in figures:
        shutil.copy2(FIGURES / name, SRC / "figures" / name)
    archive = OUT / "arxiv-source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(SRC.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=f.relative_to(SRC).as_posix())
    files = sorted(f.relative_to(SRC).as_posix() for f in SRC.rglob("*") if f.is_file())
    print(f"{len(files)} files, {archive.stat().st_size / 1e6:.2f} MB -> {archive}", *files, sep="\n  ")


if __name__ == "__main__":
    main()
