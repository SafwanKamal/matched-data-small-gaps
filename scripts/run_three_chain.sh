#!/usr/bin/env bash
# Queues the rest of the three experiments of 2026-10-14 so that API jobs run one at a time (one spend file)
# and GPU jobs run one at a time:
#   API: near-domain hosted (already started) -> MultiDoGO hosted
#   GPU: local orchestrator replication (already started) -> near-domain encoder -> MultiDoGO encoders
#   then the two reports.
set -uo pipefail
cd "$(dirname "$0")/.."
wait_for() { until grep -q "$2" "$1" 2>/dev/null; do sleep 60; done; }
(
  wait_for logs/near-oos.log "NEAR_OOS_EXIT_hosted_"
  bash scripts/run_multidogo.sh hosted
) &
(
  wait_for logs/v2-domains.log "V2_DOMAINS_EXIT_"
  bash scripts/run_near_oos.sh train
  bash scripts/run_multidogo.sh train
) &
wait
bash scripts/run_near_oos.sh report
bash scripts/run_multidogo.sh report
echo "THREE_CHAIN_DONE" >> logs/multidogo.log
