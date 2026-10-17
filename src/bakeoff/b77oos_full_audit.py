"""POST HOC: label audit of every BANKING77-OOS split file against the original Banking77 labels, keyed by upstream line number.

    python -m bakeoff.b77oos_full_audit

Extends the run D audit (posthoc_round3b.label_audit, which covers the retrieval pool and run D's sampled items) to all files
of github.com/jianguoz/Few-Shot-Intent-Detection commit fa955331 under data/banking77-oos, for the notification to the data
set's authors. Each utterance is matched to legacy-datasets/banking77 after lower-casing and stripping, as in the run D audit.
The held-out intents are the 27 original labels in id-oos/*/label_original.

Writes artifacts/posthoc/b77oos_full_audit.csv (one row per flagged line) and artifacts/posthoc/b77oos_full_audit.json (counts).
"""

from __future__ import annotations

import csv
import io
import json

from bakeoff.posthoc_round3b import original_labels
from bakeoff.protocol import ROOT

DATA = ROOT / "data" / "banking77-oos"
OUT = ROOT / "artifacts" / "posthoc"
IN_SCOPE_FILES = ("train", "valid", "test")
HELD_OUT_FILES = ("id-oos/train", "id-oos/valid", "id-oos/test")


def lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    orig = {k: {label.lower() for label in v} for k, v in original_labels().items()}
    held = set()
    for split in HELD_OUT_FILES:
        if (DATA / split / "label_original").exists():
            held |= {label.strip().lower() for label in lines(DATA / split / "label_original")}
    rows, counts = [], {}
    for split in (*IN_SCOPE_FILES, *HELD_OUT_FILES):
        if not (DATA / split / "seq.in").exists():
            continue
        texts, given = lines(DATA / split / "seq.in"), [g.strip().lower() for g in lines(DATA / split / "label")]
        c = {"lines": len(texts), "matched": 0, "flagged": 0}
        for n, (text, g) in enumerate(zip(texts, given), start=1):
            true = orig.get(text.strip().lower())
            if true is None:
                continue
            c["matched"] += 1
            if split in IN_SCOPE_FILES:
                if g in true:
                    continue
                problem = "held-out intent under an in-scope label" if true & held else "wrong in-scope label"
            else:
                if true & held:
                    continue
                problem = "in-scope intent labelled oos"
            c["flagged"] += 1
            c[problem] = c.get(problem, 0) + 1
            rows.append({"file": f"data/banking77-oos/{split}/seq.in", "line": n, "text": text, "given_label": g, "banking77_label": "|".join(sorted(true)), "problem": problem})
        counts[split] = c
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["file", "line", "text", "given_label", "banking77_label", "problem"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (OUT / "b77oos_full_audit.csv").write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    summary = {
        "status": "POST HOC: label audit of all BANKING77-OOS files for the notification to the data set's authors; not pre-registered",
        "source": "github.com/jianguoz/Few-Shot-Intent-Detection commit fa955331, data/banking77-oos; matched to legacy-datasets/banking77 after lower-casing and stripping",
        "held_out_intents": sorted(held),
        "files": counts,
    }
    (OUT / "b77oos_full_audit.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
