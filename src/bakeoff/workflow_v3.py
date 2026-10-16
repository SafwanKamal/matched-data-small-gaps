"""Workflow v3: forced-rate escalation, comparing rankers. Frozen rules are in workflow_v3_protocol.json."""

from __future__ import annotations

import json

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v2 as v2
from bakeoff.data import load_items
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT, load_protocol

RULES_PATH = ROOT / "workflow_v3_protocol.json"
OUT = ROOT / "artifacts" / "workflow-v3"
SEED = 20261011


def _guess_block(rules_v2: dict, labels, probs: np.ndarray) -> str:
    top = np.argsort(-probs)[:3]
    return rules_v2["guesses_block"].format(lines="\n".join(f"- {labels[i]} ({100 * probs[i]:.0f}%)" for i in top))


def _shots(rules_v2: dict, labels, temperature: float) -> list[tuple[str, int]]:
    """Ten worked examples from dev: (prompt text, correct action)."""
    protocol = load_protocol()
    dev = load_items(ROOT / "artifacts" / "splits.json", protocol)["items"]["banking77"]["dev"]
    _, _, _, _, oos_dev = v1._sources()
    dev_probs = apply_temperature(np.load(v1.OUT / "encoder__dev_in_scope.npz")["scores"], temperature)
    oos_probs = apply_temperature(np.load(v1.OUT / "encoder__dev_out_of_scope.npz")["scores"], temperature)
    rng = np.random.default_rng(SEED)
    right = [i for i, item in enumerate(dev) if int(dev_probs[i].argmax()) == item.gold_index]
    wrong = [i for i, item in enumerate(dev) if int(dev_probs[i].argmax()) != item.gold_index]
    shots = []
    for index in rng.choice(right, size=4, replace=False).tolist():
        shots.append((rules_v2["prompt"].format(text=dev[index].text, guesses=_guess_block(rules_v2, labels, dev_probs[index])), 1))
    for index in rng.choice(wrong, size=4, replace=False).tolist():
        shots.append((rules_v2["prompt"].format(text=dev[index].text, guesses=_guess_block(rules_v2, labels, dev_probs[index])), 2))
    for index in rng.choice(len(oos_dev), size=2, replace=False).tolist():
        shots.append((rules_v2["prompt"].format(text=oos_dev[index], guesses=_guess_block(rules_v2, labels, oos_probs[index])), 4))
    order = rng.permutation(len(shots)).tolist()
    return [shots[i] for i in order]


def few_shot_probs(rules_v2: dict, sessions, labels, encoder_probs: dict, temperature: float) -> dict:
    import torch

    from bakeoff.readout import READOUT_PATH, IndexReadout
    from bakeoff.suite import load_suite_protocol

    cache = OUT / "few_shot_actions.json"
    if cache.exists():
        return json.loads(cache.read_text())
    shots = _shots(rules_v2, labels, temperature)
    (OUT / "few_shot_examples.json").write_text(json.dumps([{"prompt": text, "action": action} for text, action in shots], indent=1) + "\n", encoding="utf-8")
    system_spec = load_suite_protocol()["systems"]["qwen3.5-9b-4bit-index_prior"]
    spec = json.loads(READOUT_PATH.read_text())
    spec["model"] = {"hf_id": system_spec["hf_id"], "revision": system_spec["revision"], "load_in_4bit": True}
    readout = IndexReadout(spec)
    tokenizer = readout.tokenizer
    prefill = tokenizer.encode("Action: [", add_special_tokens=False)
    digits = [readout.digit_ids[d] for d in "1234"]
    history = []
    for text, action in shots:
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": f"Action: [{action}]"}]
    # Progress is saved every 250 turns so an interrupted run resumes instead of starting over.
    partial = OUT / "few_shot_actions.partial.json"
    out = json.loads(partial.read_text()) if partial.exists() else {}
    turns = [turn for session in sessions for turn in session]
    for index, turn in enumerate(turns, start=1):
        if turn["id"] in out:
            continue
        if index % 250 == 0:
            partial.write_text(json.dumps(out) + "\n")
        content = rules_v2["prompt"].format(text=turn["text"], guesses=_guess_block(rules_v2, labels, encoder_probs[turn["id"]]))
        chat = tokenizer.apply_chat_template(history + [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(chat, add_special_tokens=False) + prefill
        probs = torch.softmax(readout._next_log_probs([ids])[0, digits], dim=0).cpu().numpy()
        out[turn["id"]] = {"action": int(probs.argmax()) + 1, "probs": [round(float(p), 4) for p in probs]}
        if index % 250 == 0:
            print(f"  few-shot orchestrator {index}/{len(turns)}", flush=True)
    readout.close()
    cache.write_text(json.dumps(out) + "\n")
    return out


def _auroc(score: np.ndarray, positive: np.ndarray) -> float:
    ranks = score.argsort().argsort().astype(np.float64)
    # Average ranks over ties.
    for value in np.unique(score):
        tie = score == value
        ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v2 = json.loads(v2.RULES_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    labels = load_protocol()["datasets"]["banking77"]["labels"]
    rules_v1 = json.loads((v1.OUT / "summary.json").read_text())["rules"]
    turns = [turn for session in sessions for turn in session]
    local, encoder_probs = {}, {}
    for kind in ("encoder", "decoder"):
        probs = apply_temperature(np.load(v1.OUT / f"{kind}__session_turns.npz")["scores"], rules_v1[kind]["temperature"])
        local[kind] = {turn["id"]: {"top": float(row.max()), "intent": int(row.argmax())} for turn, row in zip(turns, probs)}
        if kind == "encoder":
            encoder_probs = {turn["id"]: row for turn, row in zip(turns, probs)}
    jev_typed = {}
    for line in (v1.OUT / "jev_typed_state.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        jev_typed[row["id"]] = row
    tokens = {key: row["input_tokens"] for key, row in jev_typed.items()}
    zero = json.loads((v2.OUT / "actions.json").read_text())["orchestrator_with_confidence"]
    few = few_shot_probs(rules_v2, sessions, labels, encoder_probs, rules_v1["encoder"]["temperature"])

    gate_out = {turn["id"] for turn in turns if int(np.argmax(zero[turn["id"]]["probs"])) == 3}
    pool = [turn for turn in turns if turn["id"] not in gate_out]

    def escalate_score(source: dict, turn_id: str) -> float:
        p = source[turn_id]["probs"]
        return (p[1] + p[2]) / max(p[0] + p[1] + p[2], 1e-9)

    scores = {
        "encoder_confidence": np.asarray([-local["encoder"][turn["id"]]["top"] for turn in pool]),
        "orchestrator_zero_shot": np.asarray([escalate_score(zero, turn["id"]) for turn in pool]),
        "orchestrator_few_shot": np.asarray([escalate_score(few, turn["id"]) for turn in pool]),
    }
    rank = lambda values: values.argsort().argsort().astype(np.float64)
    scores["combined"] = rank(scores["encoder_confidence"]) + rank(scores["orchestrator_few_shot"])
    accept_wrong = np.asarray([(not turn["in_scope"]) or local["encoder"][turn["id"]]["intent"] != turn["intent"] for turn in pool])
    summary = {
        "protocol_id": rules["protocol_id"],
        "gate": {"marked_out_of_scope": len(gate_out), "pool": len(pool), "accepting_encoder_wrong_in_pool": int(accept_wrong.sum())},
        "few_shot_action_shares": {str(a): float(np.mean([few[turn["id"]]["action"] == a for turn in turns])) for a in (1, 2, 3, 4)},
        "ranking_auroc": {name: _auroc(values, accept_wrong) for name, values in scores.items()},
        "results": {},
    }
    print("ranking AUROC for 'accepting the encoder would be wrong':", {k: round(v, 3) for k, v in summary["ranking_auroc"].items()}, flush=True)

    def run(chosen_ids: set[str], target: str) -> dict:
        action_of = {}
        for turn in turns:
            if turn["id"] in gate_out:
                action_of[turn["id"]] = 4
            elif turn["id"] in chosen_ids:
                action_of[turn["id"]] = 2 if target == "decoder" else 3
            else:
                action_of[turn["id"]] = 1
        return v1.evaluate(sessions, v2._decide(sessions, action_of, local, rules_v1, jev_typed), tokens)

    base = run(set(), "decoder")
    summary["no_escalation"] = {"turn_success": base["turn_success"], "session_success": base["session_success"], "session_success_ci": base["session_success_ci"]}
    print(f"no escalation: turn {base['turn_success']:.3f} session {base['session_success']:.3f}", flush=True)
    rng = np.random.default_rng(SEED)
    pool_ids = [turn["id"] for turn in pool]
    for target in rules["targets"]:
        def target_right(turn) -> bool:
            if target == "decoder":
                answer = local["decoder"][turn["id"]]
                scope, intent = answer["top"] >= rules_v1["decoder"]["tau"], answer["intent"]
            else:
                answer = jev_typed[turn["id"]]
                scope, intent = answer["scope"] >= 0.5, int(np.argmax(answer["intent_probs"]))
            return (scope == turn["in_scope"]) and (not turn["in_scope"] or intent == turn["intent"])

        helps = np.asarray([wrong and target_right(turn) for turn, wrong in zip(pool, accept_wrong)])
        for rate in rules["rates"]:
            k = int(round(rate * len(turns)))
            row = {}
            for name, values in scores.items():
                chosen = {pool_ids[i] for i in np.argsort(-values, kind="stable")[:k]}
                result = run(chosen, target)
                row[name] = {"turn_success": result["turn_success"], "session_success": result["session_success"], "session_success_ci": result["session_success_ci"]}
            oracle = {pool_ids[i] for i in np.argsort(-helps.astype(np.float64), kind="stable")[:k]}
            result = run(oracle, target)
            row["oracle"] = {"turn_success": result["turn_success"], "session_success": result["session_success"]}
            draws = [run({pool_ids[i] for i in rng.choice(len(pool), size=k, replace=False)}, target) for _ in range(200)]
            row["random"] = {"turn_success": float(np.mean([d["turn_success"] for d in draws])), "session_success": float(np.mean([d["session_success"] for d in draws]))}
            summary["results"][f"{target}@{rate}"] = row
            print(f"{target} rate {rate:.2f}: " + "  ".join(f"{name} {value['session_success']:.3f}" for name, value in row.items()), flush=True)
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("WORKFLOW_V3_DONE", flush=True)


if __name__ == "__main__":
    main()
