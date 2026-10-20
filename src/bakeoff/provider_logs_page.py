"""Import a saved copy of OpenRouter's Logs page as provider-side evidence of when requests were sent.

    python -m bakeoff.provider_logs_page "<saved Logs _ OpenRouter.html>"

Used for run R, which OpenRouter's activity export did not yet include. The Logs page lists every request of the key
in a time window, newest first, and dates each row with OpenRouter's own clock (`data-created-at`, UTC, to the
millisecond). The page renders only the rows in view, so the saved copy holds the oldest rows of the window and the
total row count, not every row's cost. Writes artifacts/provider-log/logs-page-<date>.json with the window, the row
count, the oldest request and the rendered rows (time, model, provider, tokens, cost). The HTML itself is not kept.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from urllib.parse import parse_qs, urlparse

from bakeoff.protocol import ROOT

OUT = ROOT / "artifacts" / "provider-log"


def main(src: str) -> None:
    raw = open(src, "rb").read()
    text = raw.decode("utf-8", errors="replace")
    url = re.search(r"saved from url=\(\d+\)(\S+) -->", text).group(1)
    query = parse_qs(urlparse(url).query)
    last = re.search(r'data-index="(\d+)" data-last-row=""', text)
    rows = []
    for m in re.finditer(r'data-index="(\d+)" data-created-at="([^"]+)"', text):
        start = text.index(">", m.end()) + 1  # skip the rest of the row tag
        cells = re.sub(r"<[^>]+>", "|", text[start: start + 4000])
        cells = [c.strip() for c in html.unescape(cells).split("|") if c.strip()][:12]
        rows.append({"index": int(m.group(1)), "created_at": m.group(2), "model": cells[1], "provider": cells[2],
                     "tokens_prompt": int(cells[4].replace(",", "")), "tokens_completion": int(cells[6].replace(",", "")),
                     "cost_usd": float(cells[9])})
    rows.sort(key=lambda r: r["created_at"])
    record = {
        "source": "OpenRouter Logs page of the project's API key, saved from the browser by the authors; timestamps are OpenRouter's (UTC, data-created-at)",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "window_from": query["from"][0],
        "window_to": query["to"][0],
        "rows_in_window": int(last.group(1)) + 1 if last else None,
        "last_row_index": int(last.group(1)) if last else None,
        "oldest_request": next((r["created_at"] for r in rows if last and r["index"] == int(last.group(1))), None),
        "rendered_rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"logs-page-{record['window_from'][:10]}.json"
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{path.name}: window {record['window_from']} to {record['window_to']}, {record['rows_in_window']} rows, oldest {record['oldest_request']}, {len(rows)} rendered")


if __name__ == "__main__":
    main(sys.argv[1])
