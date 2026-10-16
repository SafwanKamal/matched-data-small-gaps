"""Workflow v2: a local model routes each turn. Frozen rules are in workflow_v2_protocol.json."""

from __future__ import annotations

import json

import numpy as np

from bakeoff import workflow as v1
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT, load_protocol

RULES_PATH = ROOT / "workflow_v2_protocol.json"
OUT = ROOT / "artifacts" / "workflow-v2"


def _actions(rules: dict, sessions, labels, encoder_probs: dict) -> dict:
    """Orchestrator action per turn for each prompt variant. Loads the model once."""
    import torch

    from bakeoff.readout import READOUT_PATH, IndexReadout
    from bakeoff.suite import load_suite_protocol

    cache = OUT / "actions.json"
    if cache.exists():
        return json.loads(cache.read_text())
    system_spec = load_suite_protocol()["systems"]["qwen3.5-9b-4bit-index_prior"]
    spec = json.loads(READOUT_PATH.read_text())
    spec["model"] = {"hf_id": system_spec["hf_id"], "revision": system_spec["revision"], "load_in_4bit": True}
    readout = IndexReadout(spec)
    tokenizer = readout.tokenizer
    prefill = tokenizer.encode("Action: [", add_special_tokens=False)
    digits = [readout.digit_ids[d] for d in "1234"]
    out = {name: {} for name in rules["variants"]}
    turns = [turn for session in sessions for turn in session]
    for index, turn in enumerate(turns, start=1):
        probs = encoder_probs[turn["id"]]
        top = np.argsort(-probs)[:3]
        blocks = {
            "orchestrator_with_confidence": rules["guesses_block"].format(lines="\n".join(f"- {labels[i]} ({100 * probs[i]:.0f}%)" for i in top)),
            "orchestrator_labels_only": rules["guesses_block"].format(lines="\n".join(f"- {labels[i]}" for i in top)),
            "orchestrator_message_only": "",
        }
        for name, guesses in blocks.items():
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": rules["prompt"].format(text=turn["text"], guesses=guesses)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(chat, add_special_tokens=False) + prefill
            log_probs = readout._next_log_probs([ids])[0, digits]
            action_probs = torch.softmax(log_probs, dim=0).cpu().numpy()
            out[name][turn["id"]] = {"action": int(action_probs.argmax()) + 1, "probs": [round(float(p), 4) for p in action_probs]}
        if index % 250 == 0:
            print(f"  orchestrator {index}/{len(turns)}", flush=True)
    readout.close()
    cache.write_text(json.dumps(out) + "\n")
    return out


def _decide(sessions, action_of: dict, local: dict, rules_v1: dict, jev_typed: dict) -> list[list[dict]]:
    out = []
    for turns in sessions:
        rows = []
        previous = None
        for turn in turns:
            action = action_of[turn["id"]]
            calls, decoder_calls = 0, 0
            if action == 1:
                scope, intent = True, local["encoder"][turn["id"]]["intent"]
            elif action == 2:
                answer = local["decoder"][turn["id"]]
                scope, intent, decoder_calls = answer["top"] >= rules_v1["decoder"]["tau"], answer["intent"], 1
            elif action == 3:
                answer = jev_typed[turn["id"]]
                scope, intent, calls = answer["scope"] >= 0.5, int(np.argmax(answer["intent_probs"])), 1
            else:
                scope, intent = False, -1
            repeat = scope and previous is not None and intent == previous
            rows.append({"scope": bool(scope), "intent": intent, "repeat": bool(repeat), "jev_calls": calls, "decoder_calls": decoder_calls, "escalated": action in (2, 3), "action": action})
            if scope:
                previous = intent
        out.append(rows)
    return out


def _matched_actions(sessions, action_of: dict, local: dict, rules_v1: dict) -> dict:
    """Same number of Jev and decoder calls as the orchestrator, assigned by encoder confidence alone."""
    turns = [turn for session in sessions for turn in session]
    n_jev = sum(action_of[turn["id"]] == 3 for turn in turns)
    n_decoder = sum(action_of[turn["id"]] == 2 for turn in turns)
    order = sorted(turns, key=lambda turn: local["encoder"][turn["id"]]["top"])
    matched = {}
    for rank, turn in enumerate(order):
        if rank < n_jev:
            matched[turn["id"]] = 3
        elif rank < n_jev + n_decoder:
            matched[turn["id"]] = 2
        else:
            # Accepted turns keep the encoder's own scope rule from workflow v1.
            matched[turn["id"]] = 1 if local["encoder"][turn["id"]]["top"] >= rules_v1["encoder"]["tau"] else 4
    return matched


def _extras(sessions, decisions, local) -> dict:
    turns = [turn for session in sessions for turn in session]
    rows = [row for session in decisions for row in session]
    shares = {str(a): float(np.mean([row["action"] == a for row in rows])) for a in (1, 2, 3, 4)}
    marked = [turn for turn, row in zip(turns, rows) if row["action"] == 4]
    out_of_scope = [turn for turn in turns if not turn["in_scope"]]
    accepted = [turn for turn, row in zip(turns, rows) if row["action"] == 1]
    return {
        "action_shares": shares,
        "decoder_calls_per_session": sum(row["decoder_calls"] for row in rows) / len(sessions),
        "action4_precision": float(np.mean([not turn["in_scope"] for turn in marked])) if marked else None,
        "action4_recall": float(np.mean([row["action"] == 4 for turn, row in zip(turns, rows) if not turn["in_scope"]])) if out_of_scope else None,
        "accepted_encoder_right": float(np.mean([turn["in_scope"] and local["encoder"][turn["id"]]["intent"] == turn["intent"] for turn in accepted])) if accepted else None,
    }


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    labels = load_protocol()["datasets"]["banking77"]["labels"]
    v1_summary = json.loads((v1.OUT / "summary.json").read_text())
    rules_v1 = v1_summary["rules"]
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
    actions = _actions(rules, sessions, labels, encoder_probs)
    summary = {"protocol_id": rules["protocol_id"], "policies": {}, "from_workflow_v1": {name: v1_summary["policies"].get(name) for name in rules["comparators"]["from_workflow_v1"]}}
    for name in rules["variants"]:
        action_of = {key: value["action"] for key, value in actions[name].items()}
        for label, chosen in ((name, action_of), (f"{name}__matched_threshold", _matched_actions(sessions, action_of, local, rules_v1))):
            decisions = _decide(sessions, chosen, local, rules_v1, jev_typed)
            block = v1.evaluate(sessions, decisions, tokens)
            block.update(_extras(sessions, decisions, local))
            summary["policies"][label] = block
            print(
                f"{label:48s} turn {block['turn_success']:.3f} session {block['session_success']:.3f} [{block['session_success_ci'][0]:.3f},{block['session_success_ci'][1]:.3f}] "
                f"actions {' '.join(f'{k}:{v:.2f}' for k, v in block['action_shares'].items())} jev/session {block['jev_requests_per_session']:.2f} decoder/session {block['decoder_calls_per_session']:.2f}",
                flush=True,
            )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("WORKFLOW_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
