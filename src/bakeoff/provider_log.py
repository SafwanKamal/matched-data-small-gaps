"""Import OpenRouter's own request log (the account's activity export) as provider-side evidence of when requests were sent.

    python -m bakeoff.provider_log <openrouter_activity_export.csv> [<later export.csv> ...]

Keeps only the columns needed to date and price each request (server timestamp in UTC, model, provider, tokens, cost,
finish reason) and drops generation ids, key names and app names. Several exports are merged, and a request that
appears in more than one is kept once (by generation id). Writes
artifacts/provider-log/openrouter_requests.csv.gz and artifacts/provider-log/summary.json.
"""

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from bakeoff.protocol import ROOT

OUT = ROOT / "artifacts" / "provider-log"
KEEP = ["created_at", "model_permaslug", "provider_name", "tokens_prompt", "tokens_completion", "tokens_reasoning", "cost_total", "finish_reason_normalized"]


def main(*srcs: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources, seen = [], {}
    for src in srcs:
        with open(src, "rb") as f:
            source_sha = hashlib.sha256(f.read()).hexdigest()
        with open(src, encoding="utf-8", newline="") as f:
            n = 0
            for r in csv.DictReader(f):
                seen.setdefault(r["generation_id"], {k: r[k] for k in KEEP})
                n += 1
        sources.append({"file": Path(src).name, "sha256": source_sha, "requests": n})
    rows = list(seen.values())
    rows.sort(key=lambda r: (r["created_at"], r["model_permaslug"], r["tokens_prompt"]))
    with gzip.open(OUT / "openrouter_requests.csv.gz", "wt", encoding="utf-8", newline="") as g:
        g.reconfigure(newline="\n") if hasattr(g, "reconfigure") else None
        w = csv.DictWriter(g, fieldnames=KEEP, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    by_day = defaultdict(lambda: {"requests": 0, "cost_usd": 0.0})
    for r in rows:
        d = by_day[(r["created_at"][:10], r["model_permaslug"])]
        d["requests"] += 1
        d["cost_usd"] += float(r["cost_total"] or 0)
    summary = {
        "source": "OpenRouter activity exports of the project's API key, exported by the owner; timestamps are OpenRouter's (UTC)",
        "sources": sources,
        "requests": len(rows),
        "first": rows[0]["created_at"],
        "last": rows[-1]["created_at"],
        "cost_usd": round(sum(float(r["cost_total"] or 0) for r in rows), 6),
        "by_day_and_model": {f"{d} {m}": {"requests": v["requests"], "cost_usd": round(v["cost_usd"], 6)} for (d, m), v in sorted(by_day.items())},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows)} requests, ${summary['cost_usd']:.2f}, {summary['first']} -> {summary['last']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
