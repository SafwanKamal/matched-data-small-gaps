# Freeze table

For each protocol behind the paper: when it was frozen, what it froze, and when its saved replies and its analysis first appeared.

- **Times** in the first table are the author's git commit timestamps in UTC, which git does not certify. The second table checks them against OpenRouter's own request log, which dates every hosted-model request with the provider's clock.
- **What can be checked without git:**
  - `frozen-protocols/` holds each protocol exactly as first committed, and its SHA-256 is listed below. The protocol in the repository root is the same text plus dated amendments and errata.
  - Each frozen protocol records the SHA-256 of its item list, and the item list in the release matches it.
  - `SHA256SUMS` covers every saved reply and analysis output.
- **Order:** for every run, the protocol was committed before any of its saved replies, and the replies no later than the analysis. These are commit times; the saved replies do not record when each request was sent.
- `python -m bakeoff.freeze_table` regenerates this file from the repository history.

| Run | Protocol | Frozen (UTC) | Item list SHA-256 (recorded = file) | Replies first committed | Last replies committed | Analysis committed | Later amendments | Order |
|---|---|---|---|---|---|---|---|---|
| Confirmatory runs A, B, C and I | `confirmatory_protocol.json` | 2026-10-16 03:50:57 UTC | `artifacts/confirmatory/items.json` `73a968d6d81b…` (match) | 2026-10-16 04:15:25 UTC | 2026-10-16 05:42:47 UTC | 2026-10-16 05:42:47 UTC | 4 | holds |
| Run D (retrieval suppression) | `retrieval_suppression_protocol.json` | 2026-10-16 06:14:14 UTC | `artifacts/retrieval-suppression/items.json` `85410e088899…` (match) | 2026-10-16 09:39:59 UTC | 2026-10-16 09:39:59 UTC | 2026-10-16 09:39:59 UTC | 2 | holds |
| Jev re-run (J1, J2) | `jev_rerun_protocol.json` | 2026-10-16 10:07:34 UTC | `artifacts/retrieval-suppression/items.json` `85410e088899…` (match); `artifacts/confirmatory/items.json` `73a968d6d81b…` (match) | 2026-10-16 10:15:43 UTC | 2026-10-16 10:15:43 UTC | 2026-10-16 10:15:43 UTC | 1 | holds |
| Run D2 (retrieval mechanism) | `retrieval_mechanism_protocol.json` | 2026-10-16 12:59:22 UTC | `artifacts/retrieval-mechanism/items.json` `04c55578a642…` (match) | 2026-10-16 14:49:29 UTC | 2026-10-16 14:49:29 UTC | 2026-10-16 14:49:29 UTC | 1 | holds |
| MASSIVE robustness | `massive_robustness_protocol.json` | 2026-10-16 15:10:13 UTC | `artifacts/massive-robustness/items.json` `88e6051264a4…` (match) | 2026-10-16 15:59:42 UTC | 2026-10-16 15:59:42 UTC | 2026-10-16 15:59:42 UTC | 0 | holds |
| Runs G and H | `gh_protocol.json` | 2026-10-16 16:30:12 UTC | reuses earlier item lists (see protocol) | 2026-10-16 16:56:21 UTC | 2026-10-16 16:56:21 UTC | 2026-10-16 16:56:21 UTC | 0 | holds |
| Run E (option count) | `option_count_protocol.json` | 2026-10-19 07:49:27 UTC | `artifacts/option-count/items.json` `53d7ffa2a033…` (match) | 2026-10-19 09:00:21 UTC | 2026-10-19 09:00:21 UTC | 2026-10-19 09:00:21 UTC | 0 | holds |
| Run R (reasoning on) | `reasoning_on_protocol.json` | 2026-10-20 04:12:26 UTC | `artifacts/retrieval-mechanism/items.json` `04c55578a642…` (match) | 2026-10-20 05:13:57 UTC | 2026-10-20 05:13:57 UTC | 2026-10-20 05:13:57 UTC | 3 | holds |

## Requests in OpenRouter's own log

OpenRouter's request log for the project's key (`artifacts/provider-log/`, exported by the owner) dates every hosted-model request with the provider's own clock. For each protocol: the last request of any kind before its freeze commit, and the first request after it. Jev is not served through OpenRouter, so the Jev re-run has no entry. Where the export does not yet include a run, a saved copy of OpenRouter's Logs page stands in (`artifacts/provider-log/logs-page-*.json`). It lists every request of the key in a window, dated to the millisecond, so it shows the first request after the freeze and that none came earlier in the window.

| Run | Frozen (UTC) | Last logged request before | First logged request after | Seconds after freeze |
|---|---|---|---|---|
| Confirmatory runs A, B, C and I | 2026-10-16 03:50:57 UTC | 2026-10-15 07:16:31 UTC | 2026-10-16 03:51:14 UTC | 17 |
| Run D (retrieval suppression) | 2026-10-16 06:14:14 UTC | 2026-10-16 05:39:12 UTC | 2026-10-16 06:14:25 UTC | 11 |
| Jev re-run (J1, J2) | 2026-10-16 10:07:34 UTC | not on OpenRouter | | |
| Run D2 (retrieval mechanism) | 2026-10-16 12:59:22 UTC | 2026-10-16 09:37:38 UTC | 2026-10-16 12:59:26 UTC | 4 |
| MASSIVE robustness | 2026-10-16 15:10:13 UTC | 2026-10-16 14:47:58 UTC | 2026-10-16 15:10:23 UTC | 11 |
| Runs G and H | 2026-10-16 16:30:12 UTC | 2026-10-16 15:57:47 UTC | 2026-10-16 16:30:17 UTC | 5 |
| Run E (option count) | 2026-10-19 07:49:27 UTC | 2026-10-16 16:53:46 UTC | 2026-10-19 07:49:38 UTC | 11 |
| Run R (reasoning on) | 2026-10-20 04:12:26 UTC | none since 2026-10-20 03:16:29 UTC (Logs page window) | 2026-10-20 04:12:36 UTC (saved Logs page) | 11 |

Amendments made during a run, checked against the same log:
- **Run I (confirmatory):** the amendment naming Kimi K3 and Claude Opus 5 was committed at 04:19:02 UTC. Before it, the log shows one probe request to each candidate model (04:14:46 to 04:15:13) and the 30-call plumbing check on development items for each of the two models (04:18:19 to 04:18:53), as the amendment itself records. The first session turn for either model was sent at 05:24:44 UTC.
- **Run D (DeepSeek host):** the last Parasail request was at 08:43:32 UTC, the host amendment was committed at 08:43:56, and the first StreamLake request was at 08:44:07.

## Frozen protocol hashes

| Protocol | SHA-256 as frozen | SHA-256 now (with amendments) | Amendment dates (UTC) |
|---|---|---|---|
| `confirmatory_protocol.json` | `778dad6cf9a1c9ac6592b42e7be73d924df31a9de7899f85358dd1534f31a25b` | `674313b6a0983e751dd10fbfa96eda04a3d7394be17a4b86a21adcc9ab17fb5a` | 2026-10-16 04:19, 2026-10-16 04:22, 2026-10-16 05:42, 2026-10-16 12:50 |
| `retrieval_suppression_protocol.json` | `df1365e27dcf03e00282f633cd485c45f8861c3d1fe376194e96697273c8a854` | `24906b74d28e36acffe3f64e0a5e65a9d07b49f782cccf5bd71ceba25879a729` | 2026-10-16 08:43, 2026-10-16 12:50 |
| `jev_rerun_protocol.json` | `188fab524831b912361beec99cfb9a081c554e86a6d26baf228c56c758150a60` | `3b1632fd7ccf77ea4c37f143cb49f1695e5f20901561f7314e623427574039fc` | 2026-10-16 12:50 |
| `retrieval_mechanism_protocol.json` | `a37c24807954bf5a7ca7cf956a9b1df491db83e62b69ec7f9c3cfa5f7a2f4a2e` | `6866889a872a98d84ed2e98db1bd4064e759604797c7bc0eec5ab38c2e69d0ad` | 2026-10-16 14:00 |
| `massive_robustness_protocol.json` | `c3f7b945f95030d0187ab9d94b890b08b4c8b1e343f6ca10464774960d69e68d` | `c3f7b945f95030d0187ab9d94b890b08b4c8b1e343f6ca10464774960d69e68d` | none |
| `gh_protocol.json` | `a75932c8cf455f5032ceff1eaeb42c409f3b344bf7966f13b5956ca64db0c560` | `a75932c8cf455f5032ceff1eaeb42c409f3b344bf7966f13b5956ca64db0c560` | none |
| `option_count_protocol.json` | `b6570b3eaee40a3488c843653630804efdb96d19cd00a64768a7c841cc288a5a` | `b6570b3eaee40a3488c843653630804efdb96d19cd00a64768a7c841cc288a5a` | none |
| `reasoning_on_protocol.json` | `fe7c905203894acb5d075341e2f4767b12eb44b49276a4a0cc14dade5652f8e7` | `b5f4a471647b36872389f9a7ee3b19d15bbec295bc30b9477678fab6e270065b` | 2026-10-20 04:45, 2026-10-20 05:01, 2026-10-20 05:13 |
