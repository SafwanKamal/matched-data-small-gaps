"""Which frontier models answer with reasoning disabled? One Banking77 dev item each (plumbing only, not a result)."""
import json
import time

import httpx

from bakeoff import api_baseline as api
from bakeoff import confirmatory as cf
from bakeoff.protocol import ROOT

CANDIDATES_FIRST = [
    "anthropic/claude-opus-5.5",
    "anthropic/claude-fable-5.1",
    "anthropic/claude-sonnet-5.5",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-astra",
    "google/gemini-4-argon",
    "x-ai/grok-4.3",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.8-max-prime",
    "moonshotai/kimi-k3",
]
CANDIDATES = ["anthropic/claude-opus-5", "anthropic/claude-opus-4.8", "openai/gpt-5.5", "openai/gpt-5.4", "google/gemini-3.5-flash", "google/gemini-3.1-pro-preview", "qwen/qwen3.7-max", "qwen/qwen3.8-max-0902", "deepseek/deepseek-v4-pro-0813", "mistralai/mistral-medium-3-5"]
OUT = ROOT / "artifacts" / "confirmatory" / "check" / "frontier_probe.jsonl"

api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
budget = api.Budget(api_rules, run_cap=0.25)
d = cf.load()["banking77"]
job = cf.jobs_for(d, [("dev-in-0", d["dev_in"][0]["text"])], False)[0]
with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=120.0) as client, OUT.open("a", encoding="utf-8") as handle:
    for model in CANDIDATES:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": job[1]}, {"role": "user", "content": job[2]}],
            "temperature": 0,
            "max_tokens": 24,
            "usage": {"include": True},
            "reasoning": {"enabled": False},
        }
        started = time.perf_counter()
        payload = api._post(client, body)
        latency = (time.perf_counter() - started) * 1000
        row = {"model": model, "latency_ms": latency}
        if "error" in payload:
            row["error"] = str(payload["error"])[:300]
        else:
            usage = payload.get("usage") or {}
            budget.add(float(usage.get("cost") or 0))
            row.update({
                "text": payload["choices"][0]["message"].get("content"),
                "provider": payload.get("provider"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": usage.get("cost"),
            })
        handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
budget.save()
print(f"probe spend ${budget.run:.4f}, project ${budget.total:.3f}")
