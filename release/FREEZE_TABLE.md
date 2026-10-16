# Freeze table

For each protocol behind the paper: when it was frozen, what it froze, and when its saved replies and its analysis first appeared.

- **Times** are the author's git commit timestamps in UTC. They are self-reported: git does not certify them, and the saved replies carry no server-side timestamps.
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

## Frozen protocol hashes

| Protocol | SHA-256 as frozen | SHA-256 now (with amendments) | Amendment dates (UTC) |
|---|---|---|---|
| `confirmatory_protocol.json` | `778dad6cf9a1c9ac6592b42e7be73d924df31a9de7899f85358dd1534f31a25b` | `674313b6a0983e751dd10fbfa96eda04a3d7394be17a4b86a21adcc9ab17fb5a` | 2026-10-16 04:19, 2026-10-16 04:22, 2026-10-16 05:42, 2026-10-16 12:50 |
| `retrieval_suppression_protocol.json` | `df1365e27dcf03e00282f633cd485c45f8861c3d1fe376194e96697273c8a854` | `24906b74d28e36acffe3f64e0a5e65a9d07b49f782cccf5bd71ceba25879a729` | 2026-10-16 08:43, 2026-10-16 12:50 |
| `jev_rerun_protocol.json` | `188fab524831b912361beec99cfb9a081c554e86a6d26baf228c56c758150a60` | `3b1632fd7ccf77ea4c37f143cb49f1695e5f20901561f7314e623427574039fc` | 2026-10-16 12:50 |
| `retrieval_mechanism_protocol.json` | `a37c24807954bf5a7ca7cf956a9b1df491db83e62b69ec7f9c3cfa5f7a2f4a2e` | `6866889a872a98d84ed2e98db1bd4064e759604797c7bc0eec5ab38c2e69d0ad` | 2026-10-16 14:00 |
| `massive_robustness_protocol.json` | `c3f7b945f95030d0187ab9d94b890b08b4c8b1e343f6ca10464774960d69e68d` | `c3f7b945f95030d0187ab9d94b890b08b4c8b1e343f6ca10464774960d69e68d` | none |
| `gh_protocol.json` | `a75932c8cf455f5032ceff1eaeb42c409f3b344bf7966f13b5956ca64db0c560` | `a75932c8cf455f5032ceff1eaeb42c409f3b344bf7966f13b5956ca64db0c560` | none |
