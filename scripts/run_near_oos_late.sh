#!/usr/bin/env bash
# The near-domain hosted stage failed on a key error before sending any request; it reruns here after the
# MultiDoGO hosted stage so that only one job writes the spend file, then the near-domain report is rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "MULTIDOGO_EXIT_hosted_" logs/multidogo.log 2>/dev/null; do sleep 60; done
bash scripts/run_near_oos.sh hosted
until grep -q "THREE_CHAIN_DONE" logs/multidogo.log 2>/dev/null; do sleep 60; done
bash scripts/run_near_oos.sh report
echo "NEAR_OOS_LATE_DONE" >> logs/near-oos.log
