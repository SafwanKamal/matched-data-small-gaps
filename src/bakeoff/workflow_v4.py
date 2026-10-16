"""Workflow v4: a size ladder of hosted models as escalation targets, de-escalation, and context carrying.

Frozen rules are in workflow_v4_protocol.json. API only; no GPU.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow as v1
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT, load_protocol

RULES_PATH = ROOT / "workflow_v4_protocol.json"
OUT = ROOT / "artifacts" / "workflow-v4"
SEED = 20261011
RATES = [0.0, 0.05, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------- requests


def _system_prompt(rules: dict, labels: list[str], form: str, write_notes: bool) -> str:
    request = rules["request"]
    options = "\n".join(f"{code}: {label}" for code, label in zip(api.codes_for(len(labels)), labels))
    with_repeat = form != "current_only"
    return request["system_prompt"].format(
        context_clause=request["context_clause"][form],
        repeat_item=request["repeat_item"] if with_repeat else "",
        options=options,
        repeat_line=request["repeat_line"] if with_repeat else "",
        notes_line=request["notes_line"] if write_notes else "",
    )


def _call(client, model: dict, system: str, user: str, max_tokens: int, budget: api.Budget, logprobs: bool = False) -> dict:
    row = {"error": None}
    if not budget.allowed():
        row["error"] = "budget cap reached"
        return row
    body = {
        "model": model["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},
        "provider": {"order": [model["provider"]], "allow_fallbacks": False},
    }
    if model["reasoning_off"]:
        body["reasoning"] = {"enabled": False}
    if logprobs:
        body["logprobs"] = True
        body["provider"]["require_parameters"] = True
    # Some hosts answer HTTP 200 with an upstream rate-limit error in the body, which api._post does not retry.
    for wait in (0, 3, 8, 20, 45):
        time.sleep(wait)
        started = time.perf_counter()
        payload = api._post(client, body)
        row["latency_ms"] = (time.perf_counter() - started) * 1000.0
        if "error" not in payload:
            break
    if "error" in payload:
        row["error"] = payload["error"]
        return row
    usage = payload.get("usage") or {}
    cost = float(usage.get("cost") or 0.0)
    budget.add(cost)
    details = usage.get("prompt_tokens_details") or {}
    row.update(
        {
            "text": payload["choices"][0]["message"].get("content") or "",
            "provider": payload.get("provider"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": cost,
        }
    )
    if logprobs:
        content = (payload["choices"][0].get("logprobs") or {}).get("content") or []
        row["token_logprobs"] = [[step["token"], step["logprob"]] for step in content]
    return row


def parse(text: str, n_labels: int) -> dict:
    scope = re.search(r"scope\s*:\s*\W*(yes|no)", text, re.I)
    intent = re.search(r"intent\s*:\s*\D{0,3}(\d{1,3})", text, re.I)
    repeat = re.search(r"repeat\s*:\s*\W*(yes|no)", text, re.I)
    notes = re.search(r"notes\s*:\s*(.*)", text, re.I | re.S)
    code = int(intent.group(1)) if intent else None
    out = {"parse_failed": scope is None or intent is None}
    if scope is not None:
        out["scope"] = scope.group(1).lower() == "yes"
    else:
        out["scope"] = bool(code)
    out["intent"] = code - 1 if code and 1 <= code <= n_labels else -1
    out["repeat"] = bool(repeat and repeat.group(1).lower() == "yes")
    out["notes"] = notes.group(1).strip() if notes else ""
    return out


def _read(path) -> dict:
    rows = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("error") is None:
                rows[row["id"]] = row
    return rows


def _collect(client, model, jobs: list[tuple[str, str, str]], max_tokens: int, path, budget, logprobs: bool = False) -> dict:
    """jobs: (id, system, user). Independent requests, eight at a time, resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        done = _read(path)
        todo = [job for job in jobs if job[0] not in done]
        if not todo or not budget.allowed():
            break

        def work(job):
            row = _call(client, model, job[1], job[2], max_tokens, budget, logprobs)
            row["id"] = job[0]
            return row

        with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=8) as pool:
            for count, row in enumerate(pool.map(work, todo), start=1):
                handle.write(json.dumps(row) + "\n")
                if count % 500 == 0:
                    handle.flush()
                    budget.save()
                    print(f"    {path.stem} {count}/{len(todo)} run ${budget.run:.3f} total ${budget.total:.3f}", flush=True)
        budget.save()
    done = _read(path)
    missing = [job[0] for job in jobs if job[0] not in done]
    if missing:
        raise RuntimeError(f"{path.name}: {len(missing)} requests still failing or over budget")
    return done


# ---------------------------------------------------------------- sessions and scoring


def long_sessions(sessions: list[list[dict]]) -> list[list[dict]]:
    chains = []
    for start in range(0, len(sessions), 4):
        chain, last = [], None
        for turn in (t for session in sessions[start : start + 4] for t in session):
            turn = dict(turn)
            if turn["in_scope"]:
                turn["repeat"] = None if last is None else bool(turn["intent"] == last)
                last = turn["intent"]
            chain.append(turn)
        chains.append(chain)
    return chains


def decide(sessions, answer_of, stated_repeat: bool = False) -> list[list[dict]]:
    """answer_of(turn) -> {'scope', 'intent', 'handled_by_model', optional 'repeat'}; rows fit v1.evaluate."""
    out = []
    for turns in sessions:
        rows, previous = [], None
        for turn in turns:
            answer = answer_of(turn)
            repeat = answer["repeat"] if stated_repeat else (previous is not None and answer["intent"] == previous)
            rows.append({"scope": bool(answer["scope"]), "intent": answer["intent"], "repeat": bool(repeat), "jev_calls": 0, "escalated": bool(answer.get("by_model")), "had_previous": previous is not None})
            if answer["scope"]:
                previous = answer["intent"]
        out.append(rows)
    return out


def _brief(result: dict) -> dict:
    keys = ("node_accuracy", "turn_success", "turn_success_ci", "session_success", "session_success_ci", "share_of_turns_escalated")
    return {key: result[key] for key in keys}


def _session_success(sessions, decisions) -> float:
    """The v1 session rule without the bootstrap, for the 200-draw random control."""
    ok = []
    for turns, rows in zip(sessions, decisions):
        good = True
        for turn, row in zip(turns, rows):
            good = good and row["scope"] == turn["in_scope"]
            if turn["in_scope"]:
                good = good and row["intent"] == turn["intent"] and (turn["repeat"] is None or row["repeat"] == turn["repeat"])
        ok.append(good)
    return float(np.mean(ok))


def _turn_right(turn, answer) -> bool:
    return answer["scope"] == turn["in_scope"] and (not turn["in_scope"] or answer["intent"] == turn["intent"])


def _specialists(turns):
    rules_v1 = json.loads((v1.OUT / "summary.json").read_text())["rules"]
    out = {}
    for kind in ("encoder", "decoder"):
        probs = apply_temperature(np.load(v1.OUT / f"{kind}__session_turns.npz")["scores"], rules_v1[kind]["temperature"])
        out[kind] = {
            "answers": {turn["id"]: {"scope": bool(row.max() >= rules_v1[kind]["tau"]), "intent": int(row.argmax()), "top": float(row.max())} for turn, row in zip(turns, probs)},
            "probs": {turn["id"]: row for turn, row in zip(turns, probs)},
            "escalation_tau": rules_v1[kind]["escalation_tau"],
        }
    return out


# ---------------------------------------------------------------- parts


def check(rules, labels, client, budget) -> dict:
    from bakeoff.data import load_items

    protocol = load_protocol()
    dev = load_items(ROOT / "artifacts" / "splits.json", protocol)["items"]["banking77"]["dev"][:8]
    _, _, _, _, oos_dev = v1._sources()
    cases = [(item.text, True, item.gold_index) for item in dev] + [(text, False, None) for text in oos_dev[:2]]
    system = _system_prompt(rules, labels, "current_only", False)
    report = {}
    for name, model in rules["models"].items():
        rows = [_call(client, model, system, rules["request"]["user"]["current_only"].format(text=text), rules["request"]["max_tokens"]["answer"], budget) for text, _, _ in cases]
        errors = [row["error"] for row in rows if row["error"]]
        good = [row for row in rows if not row["error"]]
        parsed = [parse(row["text"], len(labels)) for row in good]
        report[name] = {
            "errors": errors[:2],
            "n_ok": len(good),
            "parse_failed": sum(p["parse_failed"] for p in parsed),
            "max_completion_tokens": max((row["completion_tokens"] or 0 for row in good), default=None),
            "prompt_tokens": good[0]["prompt_tokens"] if good else None,
            "cached_tokens_last": good[-1]["cached_tokens"] if good else None,
            "cost_total": sum(row["cost"] for row in good),
            "providers": sorted({str(row["provider"]) for row in good}),
            "sample": good[0]["text"][:80] if good else None,
        }
        print(name, json.dumps(report[name]), flush=True)
    budget.save()
    (OUT / "check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def part_a(rules, labels, sessions, client, budget, names) -> dict:
    turns = [turn for session in sessions for turn in session]
    system = _system_prompt(rules, labels, "current_only", False)
    answers = {}
    for name in names:
        model = rules["models"][name]
        jobs = [(turn["id"], system, rules["request"]["user"]["current_only"].format(text=turn["text"])) for turn in turns]
        print(f"  part A {name}", flush=True)
        rows = _collect(client, model, jobs, rules["request"]["max_tokens"]["answer"], OUT / f"a__{name}.jsonl", budget)
        answers[name] = {key: {**parse(row["text"], len(labels)), "cost": row["cost"], "prompt_tokens": row["prompt_tokens"], "cached_tokens": row.get("cached_tokens"), "latency_ms": row["latency_ms"]} for key, row in rows.items()}
    return answers


def score_a(rules, sessions, answers, specialists) -> dict:
    turns = [turn for session in sessions for turn in session]
    rng = np.random.default_rng(SEED)
    out = {"specialists": {}, "models": {}}
    for kind, block in specialists.items():
        out["specialists"][kind] = _brief(v1.evaluate(sessions, decide(sessions, lambda turn: block["answers"][turn["id"]]), None))
    for name, model_answers in answers.items():
        cost = {turn["id"]: model_answers[turn["id"]]["cost"] for turn in turns}
        per_1000 = lambda ids: 1000.0 * sum(cost[i] for i in ids) / len(sessions)
        alone = v1.evaluate(sessions, decide(sessions, lambda turn: {**model_answers[turn["id"]], "by_model": True}), None)
        row = {
            "params_b": rules["models"][name]["params_b"],
            "family": rules["models"][name]["family"],
            "alone": {**_brief(alone), "usd_per_1000_sessions": per_1000(cost)},
            "parse_failed": sum(a["parse_failed"] for a in model_answers.values()),
            "prompt_tokens_mean": float(np.mean([a["prompt_tokens"] or 0 for a in model_answers.values()])),
            "cached_share": float(np.sum([a["cached_tokens"] or 0 for a in model_answers.values()]) / max(1, np.sum([a["prompt_tokens"] or 0 for a in model_answers.values()]))),
            "latency_p50_ms": float(np.percentile([a["latency_ms"] for a in model_answers.values()], 50)),
            "escalation": {},
        }
        for kind, block in specialists.items():
            base = block["answers"]
            order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
            helps = [(not _turn_right(turn, base[turn["id"]])) and _turn_right(turn, model_answers[turn["id"]]) for turn in turns]

            def run(chosen: set) -> dict:
                return v1.evaluate(sessions, decide(sessions, lambda turn: {**model_answers[turn["id"]], "by_model": True} if turn["id"] in chosen else base[turn["id"]]), None)

            curve = {}
            for rate in RATES:
                k = int(round(rate * len(turns)))
                chosen = {turn["id"] for turn in order[:k]}
                entry = {"by_confidence": {**_brief(run(chosen)), "usd_per_1000_sessions": per_1000(chosen)}}
                if 0.0 < rate < 1.0:
                    oracle = {turns[i]["id"] for i in np.argsort(-np.asarray(helps, dtype=np.float64), kind="stable")[:k]}
                    entry["oracle"] = {"session_success": run(oracle)["session_success"]}
                    draws = []
                    for _ in range(200):
                        drawn = {turns[i]["id"] for i in rng.choice(len(turns), size=k, replace=False)}
                        draws.append(_session_success(sessions, decide(sessions, lambda turn: model_answers[turn["id"]] if turn["id"] in drawn else base[turn["id"]])))
                    entry["random"] = {"session_success": float(np.mean(draws))}
                curve[str(rate)] = entry
            threshold = {turn["id"] for turn in turns if base[turn["id"]]["top"] < block["escalation_tau"]}
            curve["v1_threshold"] = {**_brief(run(threshold)), "usd_per_1000_sessions": per_1000(threshold)}
            row["escalation"][kind] = curve
        out["models"][name] = row
        best = max((rate for rate in RATES if rate < 1.0), key=lambda rate: row["escalation"]["encoder"][str(rate)]["by_confidence"]["session_success"])
        print(
            f"{name:24s} alone session {alone['session_success']:.3f} turn {alone['turn_success']:.3f} ${row['alone']['usd_per_1000_sessions']:.2f}/1000 | "
            f"encoder+escalate best rate {best} session {row['escalation']['encoder'][str(best)]['by_confidence']['session_success']:.3f} "
            f"${row['escalation']['encoder'][str(best)]['by_confidence']['usd_per_1000_sessions']:.2f}/1000 | parse failures {row['parse_failed']}",
            flush=True,
        )
    return out


def part_b(rules, labels, sessions, answers, specialists, client, budget) -> dict:
    spec = rules["part_b_de_escalation"]
    turns = [turn for session in sessions for turn in session]
    encoder = specialists["encoder"]
    out = {}
    for name in spec["models"]:
        if name not in answers:
            continue
        model = rules["models"][name]
        own = answers[name]
        out[name] = {}
        for form in ("blind", "informed"):
            jobs = []
            for turn in turns:
                if form == "blind":
                    user = spec["user"]["blind"].format(text=turn["text"])
                else:
                    probs = encoder["probs"][turn["id"]]
                    lines = "\n".join(f"- {labels[i]} ({100 * probs[i]:.0f}%)" for i in np.argsort(-probs)[:3])
                    user = spec["user"]["informed"].format(text=turn["text"], lines=lines)
                jobs.append((turn["id"], spec["system_prompt"], user))
            print(f"  part B {name} {form}", flush=True)
            rows = _collect(client, model, jobs, rules["request"]["max_tokens"]["triage"], OUT / f"b__{name}__{form}.jsonl", budget)
            action = {}
            for key, row in rows.items():
                found = re.search(r"\b(delegate|keep|out)\b", row["text"], re.I)
                action[key] = found.group(1).lower() if found else "keep"
            unparsed = sum(1 for row in rows.values() if not re.search(r"\b(delegate|keep|out)\b", row["text"], re.I))

            def answer(turn):
                choice = action[turn["id"]]
                if choice == "delegate":
                    return {"scope": True, "intent": encoder["answers"][turn["id"]]["intent"]}
                if choice == "out":
                    return {"scope": False, "intent": -1}
                return {**own[turn["id"]], "by_model": True}

            result = v1.evaluate(sessions, decide(sessions, answer), None)
            kept = {turn["id"] for turn in turns if action[turn["id"]] == "keep"}
            cost = sum(row["cost"] for row in rows.values()) + sum(own[i]["cost"] for i in kept)
            in_scope = [turn for turn in turns if turn["in_scope"]]
            encoder_right = lambda group: float(np.mean([encoder["answers"][t["id"]]["intent"] == t["intent"] for t in group])) if group else None
            marked_out = [turn for turn in turns if action[turn["id"]] == "out"]
            matched_order = sorted(turns, key=lambda turn: encoder["answers"][turn["id"]]["top"])
            matched = {turn["id"] for turn in matched_order[: len(kept)]}
            matched_result = v1.evaluate(sessions, decide(sessions, lambda turn: {**own[turn["id"]], "by_model": True} if turn["id"] in matched else encoder["answers"][turn["id"]]), None)
            out[name][form] = {
                **_brief(result),
                "usd_per_1000_sessions": 1000.0 * cost / len(sessions),
                "triage_usd_per_1000_sessions": 1000.0 * sum(row["cost"] for row in rows.values()) / len(sessions),
                "triage_prompt_tokens_mean": float(np.mean([row["prompt_tokens"] or 0 for row in rows.values()])),
                "shares": {choice: float(np.mean([action[turn["id"]] == choice for turn in turns])) for choice in ("delegate", "keep", "out")},
                "unparsed": unparsed,
                "encoder_accuracy_on_delegated_in_scope": encoder_right([t for t in in_scope if action[t["id"]] == "delegate"]),
                "encoder_accuracy_on_kept_in_scope": encoder_right([t for t in in_scope if action[t["id"]] == "keep"]),
                "out_precision": float(np.mean([not t["in_scope"] for t in marked_out])) if marked_out else None,
                "out_recall": float(np.mean([action[t["id"]] == "out" for t in turns if not t["in_scope"]])),
                "in_scope_wrongly_marked_out": float(np.mean([action[t["id"]] == "out" for t in in_scope])),
                "confidence_escalation_at_same_share": {**_brief(matched_result), "usd_per_1000_sessions": 1000.0 * sum(own[i]["cost"] for i in matched) / len(sessions)},
            }
            entry = out[name][form]
            print(
                f"{name:22s} {form:8s} session {entry['session_success']:.3f} ${entry['usd_per_1000_sessions']:.2f}/1000 shares {entry['shares']} "
                f"encoder acc delegated {entry['encoder_accuracy_on_delegated_in_scope']} kept {entry['encoder_accuracy_on_kept_in_scope']} "
                f"| same-share confidence escalation {entry['confidence_escalation_at_same_share']['session_success']:.3f}",
                flush=True,
            )
    return out


def _context_jobs(rules, chain, index: int, form: str, notes: str | None) -> str:
    user = rules["request"]["user"]
    if form == "raw_history":
        return v1._history(chain, index)
    if index == 0:
        return user["notes_first_turn"].format(text=chain[index]["text"])
    return user["notes"].format(notes=notes or "(empty)", text=chain[index]["text"])


def part_c(rules, labels, sessions, answers, specialists, client, budget) -> dict:
    spec = rules["part_c_context"]
    chains = long_sessions(sessions)
    request = rules["request"]
    names = {"small": spec["models"]["small"], "large": spec["models"]["large"]}
    stated: dict[str, dict[str, dict]] = {}

    def keep(row):
        return {**parse(row["text"], len(labels)), "cost": row["cost"], "prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"]}

    for role, name in names.items():
        model = rules["models"][name]
        stated[name] = {}
        # raw history: independent requests
        system = _system_prompt(rules, labels, "raw_history", False)
        jobs = [(turn["id"], system, _context_jobs(rules, chain, index, "raw_history", None)) for chain in chains for index, turn in enumerate(chain)]
        print(f"  part C {name} raw_history", flush=True)
        rows = _collect(client, model, jobs, request["max_tokens"]["answer"], OUT / f"c__{name}__raw_history.jsonl", budget)
        stated[name]["raw_history"] = {key: keep(row) for key, row in rows.items()}
        # own notes: sequential within a conversation
        system = _system_prompt(rules, labels, "notes", True)
        path = OUT / f"c__{name}__own_notes.jsonl"
        done = _read(path)
        print(f"  part C {name} own_notes", flush=True)

        def run_chain(chain):
            fresh, notes = [], None
            for index, turn in enumerate(chain):
                row = done.get(turn["id"])
                if row is None:
                    for _ in range(3):
                        row = _call(client, model, system, _context_jobs(rules, chain, index, "notes", notes), request["max_tokens"]["with_notes"], budget)
                        if row["error"] is None:
                            break
                    if row["error"] is not None:
                        raise RuntimeError(f"{name} own_notes {turn['id']}: {row['error']}")
                    row["id"] = turn["id"]
                    fresh.append(row)
                notes = parse(row["text"], len(labels))["notes"]
            return fresh

        with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=8) as pool:
            for count, fresh in enumerate(pool.map(run_chain, chains), start=1):
                for row in fresh:
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
                budget.save()
                if count % 25 == 0:
                    print(f"    {name} own_notes {count}/{len(chains)} conversations run ${budget.run:.3f}", flush=True)
        stated[name]["own_notes"] = {key: keep(row) for key, row in _read(path).items()}
    for role, name in names.items():
        other = names["large" if role == "small" else "small"]
        model = rules["models"][name]
        system = _system_prompt(rules, labels, "notes", False)
        jobs = []
        for chain in chains:
            for index, turn in enumerate(chain):
                notes = stated[other]["own_notes"][chain[index - 1]["id"]]["notes"] if index else None
                jobs.append((turn["id"], system, _context_jobs(rules, chain, index, "notes", notes)))
        print(f"  part C {name} other_notes (written by {other})", flush=True)
        rows = _collect(client, model, jobs, request["max_tokens"]["answer"], OUT / f"c__{name}__other_notes.jsonl", budget)
        stated[name]["other_notes"] = {key: keep(row) for key, row in rows.items()}

    positions = {"turns_1_5": range(0, 5), "turns_6_10": range(5, 10), "turns_11_20": range(10, 20)}

    def report(answer_of, stated_repeat: bool, costs=None) -> dict:
        decisions = decide(chains, answer_of, stated_repeat)
        result = _brief(v1.evaluate(chains, decisions, None))
        by_position = {}
        for label, span in positions.items():
            intent, repeat, tokens = [], [], []
            for chain, rows in zip(chains, decisions):
                for index in span:
                    turn, row = chain[index], rows[index]
                    if turn["in_scope"]:
                        intent.append(row["scope"] and row["intent"] == turn["intent"])
                        if turn["repeat"] is not None:
                            repeat.append(row["scope"] and row["repeat"] == turn["repeat"])
                    if costs:
                        tokens.append(costs[turn["id"]]["prompt_tokens"] or 0)
            by_position[label] = {"intent": float(np.mean(intent)), "repeat": float(np.mean(repeat)), "prompt_tokens_mean": float(np.mean(tokens)) if tokens else None}
        result["by_position"] = by_position
        if costs:
            result["usd_per_1000_turns"] = 1000.0 * float(np.mean([c["cost"] for c in costs.values()]))
            result["parse_failed"] = sum(c["parse_failed"] for c in costs.values())
        return result

    out = {"n_conversations": len(chains), "turns_per_conversation": len(chains[0]), "typed_state": {}, "models": {}}
    for kind, block in specialists.items():
        out["typed_state"][kind] = report(lambda turn: block["answers"][turn["id"]], False)
    for role, name in names.items():
        out["typed_state"][name] = report(lambda turn: answers[name][turn["id"]], False, answers[name])
        out["models"][name] = {}
        for condition, table in stated[name].items():
            out["models"][name][condition] = report(lambda turn: table[turn["id"]], True, table)
            if condition == "own_notes":
                words = [len(row["notes"].split()) for row in table.values()]
                out["models"][name][condition]["notes_words_mean"] = float(np.mean(words))
                out["models"][name][condition]["notes_words_p95"] = float(np.percentile(words, 95))
                out["models"][name][condition]["completion_tokens_mean"] = float(np.mean([row["completion_tokens"] or 0 for row in table.values()]))
        for condition in ("typed_state", "raw_history", "own_notes", "other_notes"):
            row = out["typed_state"][name] if condition == "typed_state" else out["models"][name][condition]
            print(
                f"{name:22s} {condition:12s} conversation {row['session_success']:.3f} turn {row['turn_success']:.3f} intent {row['node_accuracy']['intent']:.3f} repeat {row['node_accuracy']['repeat']:.3f} "
                f"repeat by position {[round(v['repeat'], 3) for v in row['by_position'].values()]} tokens {[v['prompt_tokens_mean'] and round(v['prompt_tokens_mean']) for v in row['by_position'].values()]}",
                flush=True,
            )
    return out


def main() -> None:
    import httpx

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    labels = load_protocol()["datasets"]["banking77"]["labels"]
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    budget = api.Budget(api_rules, run_cap=0.3 if stage == "check" else rules["run_cap_usd"])
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"protocol_id": rules["protocol_id"]}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=45.0) as client:
        if stage == "check":
            check(rules, labels, client, budget)
            print(f"CHECK_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)
            return
        turns = [turn for session in sessions for turn in session]
        specialists = _specialists(turns)
        dropped = set(summary.get("dropped_models", []))
        names = [name for name in rules["models"] if name not in dropped]
        names = names if len(sys.argv) < 3 else sys.argv[2].split(",")
        answers = part_a(rules, labels, sessions, client, budget, names)
        if stage in ("all", "a"):
            summary["part_a"] = score_a(rules, sessions, answers, specialists)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if stage in ("all", "b"):
            summary["part_b"] = part_b(rules, labels, sessions, answers, specialists, client, budget)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if stage in ("all", "c"):
            summary["part_c"] = part_c(rules, labels, sessions, answers, specialists, client, budget)
        summary["spend_usd"] = {"project_total": budget.total}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    budget.save()
    print(f"WORKFLOW_V4_DONE {stage} run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    main()
