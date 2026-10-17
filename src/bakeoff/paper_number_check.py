"""Positional, signed number check of paper/main.tex against the analysis artifacts.

Each check: (claim, anchor, printed, value, decimals). It passes when
  - the anchor (a short phrase) occurs in main.tex,
  - the printed string occurs within WINDOW characters after the anchor, and
  - the artifact value, rounded to the printed precision, equals the printed number (sign included).
Writes docs/number-check-draft3.md.
"""

from __future__ import annotations

import json
import re

from bakeoff.protocol import ROOT

A = ROOT / "artifacts"
WINDOW = 700


def j(*parts):
    return json.loads(A.joinpath(*parts).read_text(encoding="utf-8"))


def num(printed: str) -> float:
    s = printed.replace(",", "").replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    m = re.search(r"[-+]?\d*\.?\d+", s.replace("−", "-"))
    return float(m.group(0))


def jev_output_tokens() -> list[int]:
    """Billed output tokens of every answered Jev call in the frozen runs (last row per id; plumbing checks excluded)."""
    out = []
    for run in ("jev-rerun/J1", "jev-rerun/J2", "retrieval-mechanism/jev", "massive-robustness/jev"):
        for f in sorted(A.joinpath(run).glob("*.jsonl")):
            last = {}
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("error") is None and (r.get("usage") or {}).get("output_tokens") is not None:
                    last[r["id"]] = r["usage"]["output_tokens"]
            out.extend(last.values())
    return out


def checks():
    conf = j("confirmatory", "analysis.json")["sets"]
    b3 = j("posthoc", "round3b.json")
    r4 = j("posthoc", "round4.json")
    rd = j("retrieval-suppression", "analysis.json")["groups"]
    jev = j("jev-rerun", "analysis.json")
    d2 = j("retrieval-mechanism", "analysis.json")
    mr = j("massive-robustness", "analysis.json")["draws"]
    gh = j("gh", "analysis.json")
    g = "gemma-4-31b"
    c = []
    # Table 2 rows (anchor: set name at the start of the row).
    for s, label in (("banking77", "Banking77 & 92.7"), ("massive", "MASSIVE & 78.3"), ("clinc150", "CLINC150 & 93.1")):
        sy = conf[s]["systems"]["prior_0.2"]
        for key in (g, "qwen3.5-27b", f"{g}: combination", f"{g}: cascade_0.4"):
            c.append((f"Table 2 {s} {key}", label, f"{100 * sy[key]['weighted']:.1f}", 100 * sy[key]["weighted"], 1))
        c.append((f"Table 2 {s} Jev", label, f"{100 * jev['J2'][s]['systems']['jev retrieval']['weighted']:.1f}", 100 * jev["J2"][s]["systems"]["jev retrieval"]["weighted"], 1))
        t = conf[s]["tests"][f"specialist vs {g}"]
        c.append((f"Table 2 {s} specialist - Gemma", label, f"${t['difference_points']:+.2f}$", t["difference_points"], 2))
        f = b3["freshness"]["sets"][s]["tests_never_seen"][f"specialist vs {g}"]
        c.append((f"Table 2 {s} never-seen", label, f"${f['difference_points']:+.2f}$", f["difference_points"], 2))
    # Section 4 text.
    c.append(("CLINC150 Holm p", "ahead on CLINC150 by 1.9 points", "$p=.007$", conf["clinc150"]["tests"][f"specialist vs {g}"]["holm_p"], 3))
    c.append(("TOST lower bound", "lower bound is", "$-2.51$", b3["tost_noise"]["lower_bound_mean"], 2))
    c.append(("MASSIVE MDD", "cannot detect differences below about", "9 points", b3["mdd"]["massive"]["mdd_80pct_points"], 0))
    for k, dr in enumerate(("draw_1", "draw_2", "draw_3")):
        v = r4["jev_vs_gemma"]["massive_draws_qwen_minus_specialist"][dr]
        c.append((f"Qwen - specialist {dr}", "Qwen is 2.6 points ahead", "1.9 to 3.4", v, 1) if False else None)
    c = [x for x in c if x]
    q = r4["jev_vs_gemma"]["massive_draws_qwen_minus_specialist"]
    c.append(("Qwen - specialist draws min", "Qwen is 2.6 points ahead", "1.9", min(q.values()), 1))
    c.append(("Qwen - specialist draws max", "Qwen is 2.6 points ahead", "3.4", max(q.values()), 1))
    jd = r4["jev_vs_gemma"]["massive_draws_jev_minus_gemma"]
    c.append(("Jev - Gemma draw 1", "but not on the three further draws", "$+0.5$", jd["draw_1"], 1))
    c.append(("Jev - Gemma draw 2", "but not on the three further draws", "$-0.8$", jd["draw_2"], 1))
    c.append(("Jev - Gemma draw 3", "but not on the three further draws", "$-3.5$", jd["draw_3"], 1))
    c.append(("Jev MASSIVE confirmatory", "4.0 points ahead on the confirmatory MASSIVE draw", "4.0", r4["jev_vs_gemma"]["massive"]["difference_points"], 1))
    c.append(("G range CLINC", "changed its weighted accuracy by at most", "0.16", gh["G"]["clinc150"]["weighted_range_points"], 2))
    c.append(("G range MASSIVE", "changed its weighted accuracy by at most", "0.41", gh["G"]["massive"]["weighted_range_points"], 2))
    m = b3["massive"]["out_of_scope_rejected_by_domain"]
    c.append(("Gemma play", "Gemma keeps 97", "5\\%", 100 * m["play"][g], 0))
    c.append(("Gemma email", "Gemma keeps 97", "35\\%", 100 * m["email"][g], 0))
    c.append(("specialist play", "The specialist rejects", "78\\%", 100 * m["play"]["specialist (3 seeds, majority)"], 0))
    c.append(("general quirky", "the specialist is 6.1 points ahead", "6.1", b3["massive"]["specialist_vs_gemma_without_general_quirky"]["difference_points"], 1))
    for k, dr in enumerate(("draw_1", "draw_2", "draw_3")):
        t = mr[dr]["tests"]["M1 specialist vs gemma-4-31b no_rows"]
        c.append((f"M1 {dr}", "specialist minus Gemma is", f"${t['difference_points']:+.1f}$".replace("+", "-" if t["difference_points"] < 0 else "+"), t["difference_points"], 1))
    c.append(("MASSIVE source clustering (Section 4)", "clustering its out-of-scope turns by source label", "$p=.55$", r4["massive_source_clustering"]["specialist_minus_gemma"]["p"], 2))
    dense = [v["dense_minus_bm25"]["difference_points"] for v in j("workflow-retrieval", "summary_dense.json")["domains"].values()]
    c.append(("dense minus BM25 low", "a dense retriever (all-mpnet-base-v2)", "$-2.2$", min(dense), 1))
    c.append(("dense minus BM25 high", "a dense retriever (all-mpnet-base-v2)", "$+2.5$", max(dense), 1))
    c.append(("MASSIVE source clustering","Clustering out-of-scope turns by their source label", "$p=.55$", r4["massive_source_clustering"]["specialist_minus_gemma"]["p"], 2))
    fc = r4["freshness_components"]["clinc150"]
    c.append(("CLINC in-scope never seen", "Split by component", "$-2.19$", fc["in-scope, never seen"]["difference_points"], 2))
    c.append(("CLINC OOS never seen", "The change comes from the out-of-scope turns", "$+4.0$", fc["out-of-scope, never seen"]["difference_points"], 1))
    c.append(("CLINC OOS seen", "The change comes from the out-of-scope turns", "$-7.3$", fc["out-of-scope, seen before"]["difference_points"], 1))
    be = r4["cascade_break_even"]
    c.append(("cascade break-even low", "pays for hosting the specialist only above about", "79,000", be["clinc150"]["host $50/month"], -3))
    c.append(("cascade break-even high", "pays for hosting the specialist only above about", "177,000", be["massive"]["host $50/month"], -3))
    c.append(("specialist break-even low", "hosting it costs less than calling Gemma on every turn", "47,000", be["clinc150"]["specialist_alone_host_$50"], -3))
    lat = r4["latency"]
    c.append(("latency Qwen p50", "(p50 / p95 / p99, seconds)", "2.3 / 10.7", lat["qwen3.5-27b"]["p50_ms"] / 1000, 1))
    c.append(("latency Kimi p50", "(p50 / p95 / p99, seconds)", "2.8 / 5.3", lat["kimi-k3"]["p50_ms"] / 1000, 1))
    c.append(("latency Opus p50", "(p50 / p95 / p99, seconds)", "1.2 / 2.0", lat["claude-opus-5"]["p50_ms"] / 1000, 1))
    # Appendix B per-set latency table (anchor: row label).
    for key, row in ((g, "Gemma 4 31B, no rows (run A)"), (f"{g} with rows", "Gemma 4 31B, with rows (run C)"), ("qwen3.5-27b", "Qwen3.5 27B (run B)"), ("kimi-k3", "Kimi K3 (first 100 sessions) &"), ("claude-opus-5", "Claude Opus 5 (first 100 sessions) &")):
        for s in ("banking77", "massive", "clinc150"):
            for q in (50, 95):
                v = conf[s]["costs"][key][f"latency_ms_p{q}"] / 1000
                places = 1 if v >= 10 else 2
                printed = f"{v:.{places}f}"
                c.append((f"latency {key} {s} p{q}", row, ("& " if q == 50 else "/ ") + printed, v, places))
    jt = jev_output_tokens()
    c.append(("Jev output tokens low", "its API bills", "113", min(jt), 0))
    c.append(("Jev output tokens high", "its API bills", "1,259", max(jt), 0))
    cost = b3["cost"]["sets"]
    c.append(("Kimi multiple low", "were not significantly more accurate on 100 sessions per set, at", "17", cost["clinc150"]["kimi-k3"]["multiple_of_gemma_uncached"], 0))
    c.append(("Opus multiple high", "were not significantly more accurate on 100 sessions per set, at", "44", cost["banking77"]["claude-opus-5"]["multiple_of_gemma_uncached"], 0))
    # Section 5.
    for model, row in (("gemma-4-31b", "Gemma 4 31B & 59.5"), ("qwen3.5-27b", "Qwen3.5 27B & 82.2"), ("llama-3.3-70b", "Llama 3.3 70B & 50.7"), ("deepseek-v4-pro", "DeepSeek V4 Pro & 60.8")):
        arms = rd["banking77_oos"]["models"][model]
        c.append((f"Table 3 {model} retrieval", row, f"{100 * arms['retrieval']['held_out_rejected']:.1f}", 100 * arms["retrieval"]["held_out_rejected"], 1))
        lf = b3["leak"]["models"][model]["never-leaking intents, label-clean items"]
        c.append((f"Table 3 {model} leak-free", row, f"{100 * lf['rejected_retrieval']:.1f}", 100 * lf["rejected_retrieval"], 1))
        c.append((f"Table 3 {model} instruction", row, f"{100 * arms['mitigation_instruction']['held_out_rejected']:.1f}", 100 * arms["mitigation_instruction"]["held_out_rejected"], 1))
    c.append(("Table 3 Jev retrieval", "Jev 1.13.0 & 70.7", "25.3", 100 * jev["J1"]["banking77_oos"]["arms"]["retrieval"]["held_out_rejected"], 1))
    c.append(("CLINC dev-unseen specialist", "Specialist (reference)", "37.9", 100 * b3["mitigations_unseen"]["clinc_single_domain"]["specialist"]["rejected_unseen"], 1))
    sa = r4["sibling_absorption"]["models"]
    c.append(("sibling among retrieved low", "the chosen label is among the 20 retrieved labels for", "92", 100 * min(v["chosen_label_among_retrieved"] for v in sa.values()), 0))
    ident = [(label, n) for intent, label, n in sa["gemma-4-31b"]["all_mappings"] if "identity" in intent]
    c.append(("Gemma identity requests accepted", "Gemma, for example, maps all", "33", sum(n for _, n in ident), 0))
    c.append(("Gemma identity requests to why verify identity", "Gemma, for example, maps all", "33", sum(n for label, n in ident if label == "why verify identity"), 0))
    c.append(("sibling among retrieved high", "the chosen label is among the 20 retrieved labels for", "99", 100 * max(v["chosen_label_among_retrieved"] for v in sa.values()), 0))
    qc = sa["qwen3.5-27b"]
    c.append(("Qwen concentrated drop", "concentrate on one label (for Qwen", "29 points", 100 * (qc["drop_concentrated"][0] - qc["drop_concentrated"][1]), 0))
    c.append(("Qwen diffuse drop", "concentrate on one label (for Qwen", "against 8", 100 * (qc["drop_diffuse"][0] - qc["drop_diffuse"][1]), 0))
    c.append(("Jev AUROC as run", "AUROC unchanged", "$+0.003$", jev["J1"]["banking77_oos"]["tests"]["HJ2 retrieval minus zero-shot"]["auroc_change"], 3))
    c.append(("closeness leak removed", "Retrieval closeness does not explain", "0.90", b3["leak"]["closeness_dense_auroc"]["leaked rows removed"], 2))
    for model, row in (("gemma-4-31b", "Gemma 4 31B & 74.0"), ("qwen3.5-27b", "Qwen3.5 27B & 91.5"), ("llama-3.3-70b", "Llama 3.3 70B & 59.8"), ("deepseek-v4-pro", "DeepSeek V4 Pro & 68.0"), ("jev-1.13.0", "Jev 1.13.0 & 78.4")):
        for arm in ("retrieval", "random_examples", "rule_after_examples"):
            v = 100 * d2["banking77_oos"][model][arm]["held_out_rejected"]
            c.append((f"Table 4 {model} {arm}", row, f"{v:.1f}", v, 1))
    c.append(("H3 Gemma", "Gemma does not move", "$+1.5$", d2["tests"]["H3 gemma-4-31b"]["difference_points"], 1))
    ra = r4["rule_after_works"]["llama-3.3-70b"]
    c.append(("rule-after Llama recovered", "this is a mitigation for Llama only", "79\\%", 100 * ra["share_recovered"], 0))
    c.append(("rule-after Llama cost", "this is a mitigation for Llama only", "$-0.6$", ra["in_scope_right_change"]["difference_points"], 1))
    c.append(("H4 Llama", "Holm-significant only for Llama", "$-29$", d2["tests"]["H4 llama-3.3-70b"]["difference_points"], 0))
    c.append(("run C rows effect", "and its weighted accuracy rose by", "13.3", conf["massive"]["tests"]["rows effect: gemma with rows vs without"]["difference_points"], 1))
    c.append(("run C source clustering", "its rows effect has", "$p=.03$", r4["massive_source_clustering"]["rows_effect"]["p"], 2))
    mrr = r4["massive_rows_rejection"]
    gains = [mrr[d][mm]["gain_test"]["difference_points"] for d in ("draw_1", "draw_2", "draw_3") for mm in ("gemma-4-31b", "qwen3.5-27b")]
    c.append(("cross rows rejection low", "but raised rejection in all 6 cells", "$+2.4$", min(gains), 1))
    c.append(("cross rows rejection high", "but raised rejection in all 6 cells", "$+8.8$", max(gains), 1))
    c.append(("run D overlap", "462 of its 1,077 items", "462", r4["run_d_overlap"]["run_D_banking77_oos_items_in_confirmatory_turns"], 0))
    # round 5 (post hoc values from artifacts/posthoc/round5.json; pre-registered values from the analysis files)
    cfa = j("confirmatory", "analysis.json")["sets"]
    c.append(("TOST bound, fixed seed", r"pre-registered 90\% interval's lower bound is", "$-2.53$", cfa["banking77"]["tests"]["specialist vs gemma-4-31b"]["ci90"][0], 2))
    jg = j("jev-rerun", "analysis.json")["J2"]
    gate = "secondary: jev gate (retrieval) vs specialist"
    c.append(("Jev gate Banking77", "The Jev protocol's gate", "1.1 points", jg["banking77"]["tests"][gate]["difference_points"], 1))
    c.append(("Jev gate CLINC150", "The Jev protocol's gate", "$+0.9$", jg["clinc150"]["tests"][gate]["difference_points"], 1))
    c.append(("Jev gate MASSIVE", "The Jev protocol's gate", "$-0.7$", jg["massive"]["tests"][gate]["difference_points"], 1))
    r5 = j("posthoc", "round5.json")
    alt = [v for k, v in r5["massive_inscope_only_threshold"]["massive"].items() if "in-scope-only" in k]
    c.append(("MASSIVE in-scope-only rejection low", "set from in-scope development data alone, it rejects", "24 to", min(v["rejected"] for v in alt), 0))
    c.append(("MASSIVE in-scope-only rejection high", "set from in-scope development data alone, it rejects", r"54\%", max(v["rejected"] for v in alt), 0))
    c.append(("MASSIVE in-scope-only weighted low", "set from in-scope development data alone, it rejects", "73.9", min(v["weighted"] for v in alt), 1))
    c.append(("MASSIVE in-scope-only weighted high", "set from in-scope development data alone, it rejects", "79.3", max(v["weighted"] for v in alt), 1))
    sp = r5["run_d_specialist"]
    c.append(("run D specialist leak-free", "Specialist (reference) &", "68.2", sp["mean"]["leak_free_rejected"], 1))
    c.append(("run D specialist as run", "Specialist (reference) &", "49.6", sp["mean"]["as_run_rejected"], 1))
    c.append(("run D specialist seed low", "Specialist: mean of three seeds (", "35 to", min(v["as_run_rejected"] for v in sp["seeds"].values()), 0))
    c.append(("run D specialist seed high", "Specialist: mean of three seeds (", r"63\%", max(v["as_run_rejected"] for v in sp["seeds"].values()), 0))
    c.append(("run D specialist in-scope as run", "its in-scope accuracy is", "85.8", sp["mean"]["as_run_inscope_right"], 1))
    c.append(("run D specialist in-scope leak-free", "its in-scope accuracy is", "91.8", sp["mean"]["leak_free_inscope_right"], 1))
    jm = r5["jev_matched_point"]["banking77_oos"]
    c.append(("Jev matched zero-shot", r"with 95\% of in-scope requests kept, retrieval raises it from", r"61\%", jm["zero_shot"]["kept_95"]["rejected"], 0))
    c.append(("Jev matched retrieval", r"with 95\% of in-scope requests kept, retrieval raises it from", r"82\%", jm["retrieval"]["kept_95"]["rejected"], 0))
    sb = [v["zero_shot"]["share_in_retrieved_labels"] for v in r5["sibling_baseline"].values()]
    c.append(("sibling baseline low", "of accepted held-out requests; zero-shot,", "83 to", min(sb), 0))
    c.append(("sibling baseline high", "of accepted held-out requests; zero-shot,", r"94\%", max(sb), 0))
    ov = r5["run_d_overlap"]
    c.append(("run D overlap BANKING77-OOS", "items not used in the exploratory sessions (", "462", ov["banking77_oos"]["were_confirmatory_turns"], 0))
    c.append(("run D overlap CLINC", "items not used in the exploratory sessions (", "332", ov["clinc_banking"]["were_confirmatory_turns"] + ov["clinc_credit_cards"]["were_confirmatory_turns"], 0))
    ft = [r["openrouter_log"]["seconds_after_freeze"] for r in json.loads((ROOT / "release" / "freeze_table.json").read_text(encoding="utf-8")) if r["openrouter_log"]]
    c.append(("freeze to first logged request, low", "dates the first hosted request after each freeze", "4 to", min(ft), 0))
    c.append(("freeze to first logged request, high", "dates the first hosted request after each freeze", "17 seconds", max(ft), 0))
    return c


def main() -> None:
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    rows, failed = [], 0
    for claim, anchor, printed, value, dec in checks():
        pos = tex.find(anchor)
        in_place = pos >= 0 and printed in tex[pos : pos + WINDOW + len(anchor)]
        ok_value = abs(round(value, dec) - num(printed)) < 10 ** (-dec) / 2 + 1e-9
        failed += not (in_place and ok_value)
        rows.append(f"| {claim} | `{anchor[:40]}` | {printed} | {value:.4f} | {'yes' if ok_value else 'NO'} | {'yes' if in_place else 'NO'} |")
    doc = ["# Number check of paper draft 3 (positional, signed)", "", "Generated by `src/bakeoff/paper_number_check.py`. Each number must occur within 700 characters after its anchor phrase, and the artifact value, rounded to the printed precision, must equal the printed number including its sign.", "", f"{len(rows)} checks, {failed} failed.", "", "| Claim | Anchor | Printed | Artifact value | Value matches | At its place |", "|---|---|---|---|---|---|", *rows, ""]
    (ROOT / "docs" / "number-check-draft3.md").write_text("\n".join(doc), encoding="utf-8")
    print(f"{len(rows)} checks, {failed} failed")
    for row in rows:
        if "| NO" in row:
            print(row)


if __name__ == "__main__":
    main()
