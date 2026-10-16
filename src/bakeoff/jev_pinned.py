"""Rerun the Jev rows whose version was not recorded, with the version pinned to jev-1.13.0.

The Jev answers of the workflow v1 sessions (2026-10-09) and of the handoff test (2026-10-12) were requested as
jev-latest without saving the version that answered. Every other saved Jev answer records jev-1.13.0. This reruns
those two sets of requests unchanged except for the pinned model string, and reports old against new.
"""

from __future__ import annotations

import json

import numpy as np

from bakeoff import handoff as h
from bakeoff import workflow as v1
from bakeoff.protocol import ROOT, load_protocol

OUT = ROOT / "artifacts" / "jev-pinned"
PINNED = "jev-1.13.0"


def _pinned_spec() -> dict:
    spec = json.loads(v1.WORKFLOW_PATH.read_text(encoding="utf-8"))
    spec["jev_request"]["model"] = PINNED
    return spec


def workflow_v1() -> dict:
    spec = _pinned_spec()
    labels = load_protocol()["datasets"]["banking77"]["labels"]
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    old_out = v1.OUT
    old = {}
    for name in ("jev_typed_state", "jev_raw_history"):
        old[name] = {json.loads(line)["id"]: json.loads(line) for line in (old_out / f"{name}.jsonl").open(encoding="utf-8")}
    v1.OUT = OUT / "workflow-v1"
    v1.OUT.mkdir(parents=True, exist_ok=True)
    try:
        typed = v1.run_jev(spec, sessions, labels, with_history=False)
        raw = v1.run_jev(spec, sessions, labels, with_history=True)
    finally:
        v1.OUT = old_out
    out = {}
    for name, new, before in (("jev_typed_state", typed, old["jev_typed_state"]), ("jev_raw_history", raw, old["jev_raw_history"])):
        result_new = v1.evaluate(sessions, v1.decide(name, sessions, raw, typed, {}, {}), None)
        result_old = v1.evaluate(sessions, v1.decide(name, sessions, old["jev_raw_history"], old["jev_typed_state"], {}, {}), None)
        agree_scope = float(np.mean([(new[k]["scope"] >= 0.5) == (before[k]["scope"] >= 0.5) for k in new]))
        agree_intent = float(np.mean([int(np.argmax(new[k]["intent_probs"])) == int(np.argmax(before[k]["intent_probs"])) for k in new]))
        out[name] = {"session_success_unrecorded_version": result_old["session_success"], "session_success_pinned": result_new["session_success"], "session_success_pinned_ci": result_new["session_success_ci"], "scope_agreement": agree_scope, "intent_agreement": agree_intent}
        print(f"workflow v1 {name:16s} session {100 * result_old['session_success']:.1f} (unrecorded) -> {100 * result_new['session_success']:.1f} (pinned) | agreement scope {100 * agree_scope:.1f}% intent {100 * agree_intent:.1f}%", flush=True)
    return out


def handoff() -> dict:
    rules = json.loads(h.RULES_PATH.read_text(encoding="utf-8"))
    labels = load_protocol()["datasets"]["banking77"]["labels"]
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    summary = json.loads((h.OUT / "summary.json").read_text())["sets"]
    pinned_protocol = OUT / "workflow_protocol_pinned.json"
    pinned_protocol.write_text(json.dumps(_pinned_spec(), indent=2) + "\n", encoding="utf-8")
    old_path, old_out = v1.WORKFLOW_PATH, h.OUT
    out = {}
    try:
        v1.WORKFLOW_PATH = pinned_protocol
        h.OUT = OUT / "handoff"
        h.OUT.mkdir(parents=True, exist_ok=True)
        for set_name, (group, limit) in h.SETS.items():
            convs = h.conversations(sessions, group, limit)
            ids = [f"{set_name}-{i}" for i in range(len(convs))]
            questions_of = {cid: h.build_questions(conv, labels, rules["seed"], i) for i, (cid, conv) in enumerate(zip(ids, convs))}
            sources = {"raw": {cid: "Customer messages:\n" + h._messages(conv) for cid, conv in zip(ids, convs)}}
            for writer in rules["models"]:
                for words in rules["note_budgets_words"]:
                    rows = {}
                    for line in (old_out / f"w__{set_name}__{writer}__{words}.jsonl").open(encoding="utf-8"):
                        row = json.loads(line)
                        if row.get("error") is None:
                            rows[row["id"]] = row
                    sources[f"{writer}__{words}"] = {cid: "Handoff notes:\n" + h._cut(rows[cid]["text"].strip(), words)[0] for cid in ids}
            out[set_name] = {}
            for tag, texts in sources.items():
                scored = h._jev_read(rules, f"{set_name}__{tag}", texts, questions_of)
                cell = h._cell([scored[cid] for cid in ids])
                before = summary[set_name]["raw"]["jev"] if tag == "raw" else summary[set_name]["cells"][f"{tag.replace('__', '@')}->jev"]
                out[set_name][tag] = {"accuracy_unrecorded_version": before["accuracy"], "accuracy_pinned": cell["accuracy"], "ci_pinned": cell["ci"]}
                print(f"handoff {set_name} {tag:28s} Jev reader {100 * before['accuracy']:.1f} (unrecorded) -> {100 * cell['accuracy']:.1f} (pinned)", flush=True)
    finally:
        v1.WORKFLOW_PATH, h.OUT = old_path, old_out
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"pinned_model": PINNED, "workflow_v1": workflow_v1()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["handoff"] = handoff()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("JEV_PINNED_DONE", flush=True)


if __name__ == "__main__":
    main()
