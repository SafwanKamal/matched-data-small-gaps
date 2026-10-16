"""Smoke test of OpenRouter models on Banking77 dev items. Not a result: it measures token use, cost,
parse failures and whether logprobs come back. Dev items only; nothing here touches a test split."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bakeoff.suite import load_suite  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "api-smoke-2"
N = 20
CAP_USD = 1.50
URL = "https://openrouter.ai/api/v1/chat/completions"
# model -> (modes, provider pin, top_logprobs, reasoning setting, max_tokens for probs mode)
CONFIGS = {
    "deepseek/deepseek-v4.1-flash": (("index", "probs"), "Together", 20, {"enabled": False}, 400),
    "qwen/qwen3.7-flash": (("index",), "Alibaba", 5, {"enabled": False}, 400),
    "qwen/qwen3.7-plus": (("index",), "Alibaba", 5, {"enabled": False}, 400),
    "stepfun/step-3.7-flash": (("index", "probs"), None, 5, {"effort": "low"}, 2500),
    "google/gemini-3.5-flash-lite": (("probs",), None, 0, {"effort": "low"}, 2500),
    "google/gemini-3.1-pro-preview": (("probs",), None, 0, {"effort": "low"}, 2500),
}


def key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        name, _, rest = line.partition("=")
        if name.strip() == "openrouter_key":
            return rest.strip().strip("\"'")
    raise RuntimeError("openrouter_key missing from .env")


def prompt_index(text: str, labels: list[str]) -> str:
    options = "\n".join(f"{i}: {label}" for i, label in enumerate(labels))
    return f"Message:\n{text}\n\nOptions:\n{options}\n\nReply with only the number of the best option."


def prompt_probs(text: str, labels: list[str]) -> str:
    options = "\n".join(f"{i}: {label}" for i, label in enumerate(labels))
    return (
        f"Message:\n{text}\n\nOptions:\n{options}\n\nGive a probability for each plausible option as JSON "
        'like {"3": 0.7, "12": 0.2}. Probabilities sum to 1. Only the JSON.'
    )


def call(client, model: str, content: str, mode: str) -> dict:
    _, pin, top_k, reasoning, probs_tokens = CONFIGS[model]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": probs_tokens,
        "reasoning": reasoning,
        "usage": {"include": True},
    }
    if mode == "index":
        body.update(max_tokens=6 if "enabled" in reasoning else 2500, logprobs=True, top_logprobs=top_k)
        body["provider"] = {"require_parameters": True}
    if pin:
        body["provider"] = {**body.get("provider", {}), "order": [pin], "allow_fallbacks": False}
    started = time.perf_counter()
    r = client.post(URL, json=body)
    ms = (time.perf_counter() - started) * 1000
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "ms": ms}
    d = r.json()
    ch = d["choices"][0]
    u = d.get("usage", {})
    return {
        "text": ch["message"].get("content") or "",
        "has_logprobs": bool((ch.get("logprobs") or {}).get("content")),
        "provider": d.get("provider"),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost": u.get("cost") or 0.0,
        "finish": ch.get("finish_reason"),
        "ms": ms,
    }


def parse_index(text: str, n: int):
    m = re.search(r"\d+", text)
    return int(m.group()) if m and int(m.group()) < n else None


def parse_probs(text: str, n: int):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = {int(k): float(v) for k, v in json.loads(m.group()).items()}
    except (ValueError, TypeError):
        return None
    d = {k: v for k, v in d.items() if 0 <= k < n}
    return d or None


def main() -> None:
    suite = load_suite(["banking77_full"])["banking77_full"]
    labels, items = suite["labels"], suite["dev"][:N]
    OUT.mkdir(parents=True, exist_ok=True)
    spent, summary = 0.0, []
    headers = {"Authorization": f"Bearer {key()}"}
    with httpx.Client(headers=headers, timeout=90) as client:
        for model, cfg in CONFIGS.items():
            for mode in cfg[0]:
                rows = []
                for it in items:
                    if spent > CAP_USD:
                        print("cap reached, stopping")
                        break
                    p = (prompt_index if mode == "index" else prompt_probs)(it.text, labels)
                    res = call(client, model, p, mode)
                    spent += res.get("cost", 0.0) or 0.0
                    if "error" not in res:
                        pred = parse_index(res["text"], len(labels)) if mode == "index" else parse_probs(res["text"], len(labels))
                        res["parsed"] = pred is not None
                        if mode == "index":
                            res["correct"] = pred == it.gold_index
                        elif pred:
                            res["correct"] = max(pred, key=pred.get) == it.gold_index
                            res["prob_sum"] = round(sum(pred.values()), 3)
                    res["item_id"] = it.item_id
                    rows.append(res)
                    if "error" in res and rows[0] is res:
                        break  # first call failed; do not repeat 19 more
                (OUT / f"{model.replace('/', '__')}__{mode}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
                ok = [r for r in rows if "error" not in r]
                s = {
                    "model": model, "mode": mode, "n": len(rows), "errors": len(rows) - len(ok),
                    "first_error": next((r["error"] for r in rows if "error" in r), None),
                    "parsed": sum(r.get("parsed", False) for r in ok),
                    "correct": sum(r.get("correct", False) for r in ok),
                    "logprobs": sum(r["has_logprobs"] for r in ok),
                    "prompt_tok": sum(r["prompt_tokens"] or 0 for r in ok) / max(len(ok), 1),
                    "compl_tok": sum(r["completion_tokens"] or 0 for r in ok) / max(len(ok), 1),
                    "reason_tok": sum(r["reasoning_tokens"] or 0 for r in ok) / max(len(ok), 1),
                    "cost": sum(r["cost"] for r in ok),
                    "providers": sorted({str(r["provider"]) for r in ok}),
                }
                summary.append(s)
                print(json.dumps(s), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"total spent ${spent:.4f}")


if __name__ == "__main__":
    main()
