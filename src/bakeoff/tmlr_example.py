"""Worked example of the retrieval effect for the TMLR build, from run D2's saved replies (no API calls).

    python -m bakeoff.tmlr_example

Picks one held-out BANKING77-OOS request by a fixed rule: among the D2 held-out items that every general model rejects
zero-shot, accepts with 20 retrieved in-scope examples and still rejects with 20 random examples, take the one whose
retrieved examples concentrate most on one label (ties: the shortest text, then the item id). Its retrieved examples
are rebuilt with run D2's own prompt code. Writes paper/tmlr/worked_example.tex (input by the TMLR build's appendix)
and paper/tmlr/worked_example.json (kept out of artifacts/, which the release manifest hashes), which bakeoff.tmlr_number_check reads. Post hoc and descriptive: one item
chosen to illustrate the aggregate result, not evidence for it.
"""

from __future__ import annotations

import json
from collections import Counter

from bakeoff import retrieval_mechanism as rm
from bakeoff import workflow_v4 as v4
from bakeoff.jev_rerun import _read as jev_read
from bakeoff.protocol import ROOT

MODELS = ["gemma-4-31b", "qwen3.5-27b", "llama-3.3-70b", "deepseek-v4-pro"]
NAMES = {"gemma-4-31b": "Gemma 4 31B", "qwen3.5-27b": "Qwen3.5 27B", "llama-3.3-70b": "Llama 3.3 70B", "deepseek-v4-pro": "DeepSeek V4 Pro"}
ARMS = ["zero_shot", "retrieval", "random_examples", "rule_after_examples"]
ARM_NAMES = {"zero_shot": "Zero-shot", "retrieval": "Retrieval", "random_examples": "Random", "rule_after_examples": "Rule after"}
SPLIT = "banking77_oos"
OUT_JSON = ROOT / "paper" / "tmlr" / "worked_example.json"
OUT_TEX = ROOT / "paper" / "tmlr" / "worked_example.tex"


def tex_escape(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("#", r"\#").replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")


def answers(labels: list[str], held_out: list[dict]) -> dict:
    """answers[model][arm][item id] = (accepted, label name or None)."""
    out = {}
    for model in MODELS:
        out[model] = {}
        for arm in ARMS:
            rows = v4._read(rm._path(model, SPLIT, arm))
            got = {}
            for item in held_out:
                r = rows.get(item["id"])
                if r is None:
                    continue
                p = v4.parse(r["text"], len(labels))
                got[item["id"]] = (bool(p["scope"]), labels[p["intent"]] if p["scope"] and 0 <= p["intent"] < len(labels) else None)
            out[model][arm] = got
    jev = {}
    for arm in ARMS:
        rows = jev_read(rm.OUT / "jev" / f"{SPLIT}__{arm}.jsonl")
        jev[arm] = {k: float(v["scope"]) for k, v in rows.items()}
    out["jev"] = jev
    return out


def main() -> None:
    data = rm.load()
    block = data[SPLIT]
    labels = block["labels"]
    held_out = [i for i in block["items"] if not i["in_scope"]]
    got = answers(labels, held_out)

    def qualifies(i):
        k = i["id"]
        try:
            return all(not got[m]["zero_shot"][k][0] and got[m]["retrieval"][k][0] and not got[m]["random_examples"][k][0] for m in MODELS)
        except KeyError:
            return False

    candidates = [i for i in held_out if qualifies(i)]
    if not candidates:
        raise SystemExit("no held-out item meets the rule")
    seed = rm.rules()["seed"]
    prompts = {i["id"]: rm.jobs(data, SPLIT, "retrieval", [i], seed)[0][2] for i in candidates}
    code_to_label = dict(zip(__import__("bakeoff.api_baseline", fromlist=["codes_for"]).codes_for(len(labels)), labels))

    def retrieved(item_id):
        """The 20 retrieved examples, most similar first (the prompt lists them least similar first)."""
        lines = prompts[item_id].split("\n")[1:21]
        return [(line.rsplit(" => ", 1)[0], code_to_label[line.rsplit(" => ", 1)[1]]) for line in reversed(lines)]

    def concentration(i):
        return Counter(label for _, label in retrieved(i["id"])).most_common(1)[0][1]

    pick = sorted(candidates, key=lambda i: (-concentration(i), len(i["text"]), i["id"]))[0]
    ex = retrieved(pick["id"])
    counts = Counter(label for _, label in ex).most_common()
    record = {
        "rule": __doc__.split("\n\n")[2].strip(),
        "n_held_out": len(held_out),
        "n_candidates": len(candidates),
        "item": pick,
        "held_out_intent": pick["cluster"].removeprefix("held-").replace("_", " "),
        "retrieved": ex,
        "retrieved_label_counts": counts,
        "answers": {m: {a: got[m][a][pick["id"]] for a in ARMS} for m in MODELS},
        "jev_in_scope_probability": {a: got["jev"][a].get(pick["id"]) for a in ARMS},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)  # the release has no paper/tmlr folder
    OUT_JSON.write_text(json.dumps(record, indent=1), encoding="utf-8", newline="\n")

    top_label, top_n = counts[0]
    rows = []
    for m in MODELS:
        cells = []
        for a in ARMS:
            ok, lab = record["answers"][m][a]
            cells.append(r"\emph{reject}" if not ok else f"accept: {tex_escape(lab)}")
        rows.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    jp = record["jev_in_scope_probability"]
    rows.append("Jev 1.13.0 ($p$ in scope) & " + " & ".join(f"{jp[a]:.2f}" for a in ARMS) + r" \\")
    shown = "\n".join(f"  {tex_escape(t)} & {tex_escape(l)} \\\\" for t, l in ex[:8])
    others = ", ".join(f"{tex_escape(l)} ({n})" for l, n in counts)
    tex = rf"""\section{{A worked example of the retrieval effect}}
\label{{app:example}}

One held-out BANKING77-OOS request from run D2 shows the aggregate effect of \cref{{sec:operating}} on a single item. It was chosen post hoc by a fixed rule, coded in \texttt{{bakeoff.tmlr\_example}}. Among the {len(held_out)} held-out D2 items, {len(candidates)} are rejected zero-shot, accepted with retrieval and still rejected with random examples by all four general models. Of those, we show the one whose retrieved examples concentrate most on one label. It illustrates the effect and is not evidence for it.

The request, \emph{{``{tex_escape(pick['text'])}''}}, belongs to the held-out intent \emph{{{tex_escape(record['held_out_intent'])}}}, which the desk does not handle. BM25 retrieves 20 in-scope training examples for it, and {top_n} of the 20 carry the label \emph{{{tex_escape(top_label)}}}. The labels of all 20 are: {others}. \Cref{{tab:example-retrieved}} shows the eight most similar, and \cref{{tab:example-answers}} shows each model's answer under the four prompts.

\begin{{table}}[h]
\caption{{The eight retrieved in-scope examples most similar to the request (of 20), with their labels.}}
\label{{tab:example-retrieved}}
\vspace{{3pt}}
\centering\small
\begin{{tabular}}{{@{{}}p{{0.62\linewidth}}l@{{}}}}
\toprule
Retrieved example & Label \\
\midrule
{shown}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[h]
\caption{{Answers to the request under each run D2 prompt. \emph{{Random}}: 20 random in-scope examples. \emph{{Rule after}}: the scope rule repeated after the retrieved examples. Jev's in-scope probability is compared with its fixed 0.5 threshold.}}
\label{{tab:example-answers}}
\vspace{{3pt}}
\centering\small
\adjustbox{{max width=\linewidth}}{{%
\begin{{tabular}}{{@{{}}lllll@{{}}}}
\toprule
Model & Zero-shot & Retrieval & Random & Rule after \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""
    OUT_TEX.write_text(tex, encoding="utf-8", newline="\n")
    print(f"{len(candidates)} candidates; picked {pick['id']}: {pick['text']!r} ({record['held_out_intent']}); top label {top_label} {top_n}/20")
    for m in MODELS:
        print(" ", m, record["answers"][m])
    print("  jev", jp)


if __name__ == "__main__":
    main()
