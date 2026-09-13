#!/bin/bash
# Overnight experiment battery. Sequential — one Ollama model in RAM at a time.
# Resumable: a run whose RUN_END is already logged is skipped.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=/tmp/overnight.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

done_already() {
  grep -q '"RUN_END"' "research_logs/$1/notable_events.jsonl" 2>/dev/null
}

run_arm() {
  local id=$1 cfg=$2 model=$3 cycles=$4
  if done_already "$id"; then say "SKIP $id (already complete)"; return 0; fi
  say "START $id  cfg=$cfg model=$model cycles=$cycles"
  rm -rf "research_logs/$id" world
  $PY scripts/init_run.py --run-id "$id" --condition BASELINE --cycles "$cycles" --config "$cfg" >/dev/null 2>&1
  $PY -u -m controller.main --run-id "$id" --condition BASELINE --cycles "$cycles" \
      --config "$cfg" --model "$model" >"/tmp/${id}.log" 2>&1
  local rc=$?
  if done_already "$id"; then say "DONE  $id"; else say "FAIL  $id (rc=$rc) — see /tmp/${id}.log"; fi
}

say "=== overnight battery starting ==="

# Arm C at 7b first: needs no download and is directly comparable to today's
# ARM_PRESCRIBED / ARM_MINIMAL runs. Runs while 14b is still downloading.
run_arm STRUCT_7B  experiments/structural.yaml  qwen2.5:7b-instruct  4

say "waiting for qwen2.5:14b-instruct..."
waited=0
until ollama list 2>/dev/null | grep -q "qwen2.5:14b-instruct"; do
  sleep 20; waited=$((waited+20))
  if [ $waited -ge 3600 ]; then
    say "ABORT: 14b never arrived after 60 min; 7b results are still valid"
    exit 1
  fi
done
say "14b present"

# The 14b battery
run_arm PRESCRIBED_14B experiments/prescribed.yaml qwen2.5:14b-instruct 4
run_arm MINIMAL_14B    experiments/minimal.yaml    qwen2.5:14b-instruct 4
run_arm STRUCT_14B     experiments/structural.yaml qwen2.5:14b-instruct 4

say "=== battery complete ==="
ls -d research_logs/*/ | tee -a "$LOG"
