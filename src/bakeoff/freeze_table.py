"""Build the freeze table: for each governing protocol, when it was frozen, what it froze, and when its replies and analysis appeared.

Writes release/FREEZE_TABLE.md, release/freeze_table.json and release/frozen-protocols/<protocol> (the text as first
committed), so the pre-registration order can be checked from the release without git history. Times are the author's
git commit timestamps, converted to UTC; the saved replies carry no server-side timestamps.
"""

import csv
import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone

from bakeoff.protocol import ROOT

RUNS = [
    ("Confirmatory runs A, B, C and I", "confirmatory_protocol.json", "artifacts/confirmatory"),
    ("Run D (retrieval suppression)", "retrieval_suppression_protocol.json", "artifacts/retrieval-suppression"),
    ("Jev re-run (J1, J2)", "jev_rerun_protocol.json", "artifacts/jev-rerun"),
    ("Run D2 (retrieval mechanism)", "retrieval_mechanism_protocol.json", "artifacts/retrieval-mechanism"),
    ("MASSIVE robustness", "massive_robustness_protocol.json", "artifacts/massive-robustness"),
    ("Runs G and H", "gh_protocol.json", "artifacts/gh"),
]
OUT = ROOT / "release"
PROVIDER_LOG = ROOT / "artifacts" / "provider-log" / "openrouter_requests.csv.gz"
NOT_ON_OPENROUTER = {"jev_rerun_protocol.json"}


def provider_times() -> list[datetime]:
    """OpenRouter's server timestamps (UTC) for every request of the project's key."""
    if not PROVIDER_LOG.exists():
        return []
    with gzip.open(PROVIDER_LOG, "rt", encoding="utf-8", newline="") as g:
        return sorted(datetime.fromisoformat(r["created_at"]).replace(tzinfo=timezone.utc) for r in csv.DictReader(g))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def blob(rev: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def utc(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def first_added(path: str) -> tuple[str, str]:
    """(commit, author time) of the commit that first added the path."""
    line = git("log", "--diff-filter=A", "--format=%H %aI", "--", path).strip().splitlines()[-1]
    rev, when = line.split()
    return rev, when


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build() -> list[dict]:
    rows = []
    logged = provider_times()
    for name, proto, run_dir in RUNS:
        rev, frozen_at = first_added(proto)
        frozen_text = blob(rev, proto)
        current_text = blob("HEAD", proto)
        (OUT / "frozen-protocols").mkdir(parents=True, exist_ok=True)
        (OUT / "frozen-protocols" / proto).write_bytes(frozen_text)
        amendments = [utc(t) for t in git("log", "--format=%aI", f"{rev}..HEAD", "--", proto).split()][::-1]

        items = []
        for m in re.finditer(r"(artifacts/[\w./-]+?items\.json)[^\"]{0,40}?SHA-256 ([0-9a-f]{8,64})", frozen_text.decode("utf-8")):
            path, recorded = m.group(1), m.group(2)
            actual = sha(blob("HEAD", path))
            items.append({"path": path, "recorded_sha256": recorded, "file_sha256": actual,
                          "match": actual.startswith(recorded), "added": utc(first_added(path)[1])})
        if not items:  # the confirmatory set is described without its path next to the hash
            m = re.search(r"SHA-256 ([0-9a-f]{64})", frozen_text.decode("utf-8"))
            if m:
                path = f"{run_dir}/items.json"
                actual = sha(blob("HEAD", path))
                items.append({"path": path, "recorded_sha256": m.group(1), "file_sha256": actual,
                              "match": actual == m.group(1), "added": utc(first_added(path)[1])})

        replies = [p for p in git("ls-files", run_dir).splitlines() if p.endswith(".jsonl")]
        reply_times = sorted(first_added(p)[1] for p in replies)
        analysis = f"{run_dir}/analysis.json"
        analysis_at = first_added(analysis)[1]
        provider = None
        if logged and proto not in NOT_ON_OPENROUTER:
            frozen_dt = datetime.fromisoformat(frozen_at).astimezone(timezone.utc)
            before = [t for t in logged if t < frozen_dt]
            after = [t for t in logged if t >= frozen_dt]
            provider = {"last_request_before_freeze": before[-1].strftime("%Y-%m-%d %H:%M:%S UTC") if before else None,
                        "first_request_after_freeze": after[0].strftime("%Y-%m-%d %H:%M:%S UTC") if after else None,
                        "seconds_after_freeze": round((after[0] - frozen_dt).total_seconds()) if after else None}
        rows.append({
            "run": name,
            "openrouter_log": provider,
            "protocol": proto,
            "frozen_at": utc(frozen_at),
            "frozen_sha256": sha(frozen_text),
            "current_sha256": sha(current_text),
            "later_amendment_commits": amendments,
            "items": items,
            "reply_files": len(replies),
            "first_reply_at": utc(reply_times[0]),
            "last_reply_at": utc(reply_times[-1]),
            "analysis_first_at": utc(analysis_at),
            "order_holds": frozen_at <= reply_times[0] <= analysis_at
            and datetime.fromisoformat(frozen_at) <= datetime.fromisoformat(reply_times[0]) <= datetime.fromisoformat(analysis_at),
        })
    return rows


def write(rows: list[dict]) -> None:
    (OUT / "freeze_table.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8", newline="\n")
    lines = [
        "# Freeze table",
        "",
        "For each protocol behind the paper: when it was frozen, what it froze, and when its saved replies and its analysis first appeared.",
        "",
        "- **Times** in the first table are the author's git commit timestamps in UTC, which git does not certify. The second table checks them against OpenRouter's own request log, which dates every hosted-model request with the provider's clock.",
        "- **What can be checked without git:**",
        "  - `frozen-protocols/` holds each protocol exactly as first committed, and its SHA-256 is listed below. The protocol in the repository root is the same text plus dated amendments and errata.",
        "  - Each frozen protocol records the SHA-256 of its item list, and the item list in the release matches it.",
        "  - `SHA256SUMS` covers every saved reply and analysis output.",
        "- **Order:** for every run, the protocol was committed before any of its saved replies, and the replies no later than the analysis. These are commit times; the saved replies do not record when each request was sent.",
        "- `python -m bakeoff.freeze_table` regenerates this file from the repository history.",
        "",
        "| Run | Protocol | Frozen (UTC) | Item list SHA-256 (recorded = file) | Replies first committed | Last replies committed | Analysis committed | Later amendments | Order |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        items = "; ".join(f"`{i['path']}` `{i['file_sha256'][:12]}…` ({'match' if i['match'] else 'MISMATCH'})" for i in r["items"]) or "reuses earlier item lists (see protocol)"
        lines.append(f"| {r['run']} | `{r['protocol']}` | {r['frozen_at']} | {items} | {r['first_reply_at']} | {r['last_reply_at']} "
                     f"| {r['analysis_first_at']} | {len(r['later_amendment_commits'])} | {'holds' if r['order_holds'] else 'FAILS'} |")
    lines += ["", "## Requests in OpenRouter's own log", "",
              "OpenRouter's request log for the project's key (`artifacts/provider-log/`, exported by the owner) dates every hosted-model request with the provider's own clock. "
              "For each protocol: the last request of any kind before its freeze commit, and the first request after it. Jev is not served through OpenRouter, so the Jev re-run has no entry.", "",
              "| Run | Frozen (UTC) | Last logged request before | First logged request after | Seconds after freeze |", "|---|---|---|---|---|"]
    for r in rows:
        o = r["openrouter_log"]
        if o:
            lines.append(f"| {r['run']} | {r['frozen_at']} | {o['last_request_before_freeze']} | {o['first_request_after_freeze']} | {o['seconds_after_freeze']} |")
        else:
            lines.append(f"| {r['run']} | {r['frozen_at']} | not on OpenRouter | | |")
    lines += ["", "Amendments made during a run, checked against the same log:",
              "- **Run I (confirmatory):** the amendment naming Kimi K3 and Claude Opus 5 was committed at 04:19:02 UTC. Before it, the log shows one probe request to each candidate model (04:14:46 to 04:15:13) and the 30-call plumbing check on development items for each of the two models (04:18:19 to 04:18:53), as the amendment itself records. The first session turn for either model was sent at 05:24:44 UTC.",
              "- **Run D (DeepSeek host):** the last Parasail request was at 08:43:32 UTC, the host amendment was committed at 08:43:56, and the first StreamLake request was at 08:44:07."]
    lines += ["", "## Frozen protocol hashes", "", "| Protocol | SHA-256 as frozen | SHA-256 now (with amendments) | Amendment dates (UTC) |", "|---|---|---|---|"]
    for r in rows:
        dates = ", ".join(d[:16] for d in r["later_amendment_commits"]) or "none"
        lines.append(f"| `{r['protocol']}` | `{r['frozen_sha256']}` | `{r['current_sha256']}` | {dates} |")
    (OUT / "FREEZE_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    table = build()
    write(table)
    for r in table:
        print(r["run"], r["frozen_at"], "->", r["first_reply_at"], "->", r["analysis_first_at"], "order", r["order_holds"],
              "items", [i["match"] for i in r["items"]])
