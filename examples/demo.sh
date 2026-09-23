#!/usr/bin/env bash
# template -> scan -> dry-run -> run -> kill -> resume
# Runnable from any directory after `pip install .` (or PYTHONPATH=src).
set -euo pipefail
EX="$(cd "$(dirname "$0")" && pwd)"
OUT="$EX/_demo"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
rm -rf "$OUT"
mkdir -p "$OUT/jobs" "$OUT/results"

python -m fullFold template \
  --template "$EX/receptor.json" \
  --records "$EX/ligands.smi" \
  --output-dir "$OUT/jobs"

python -m fullFold template \
  --template "$EX/receptor.json" \
  --records "$EX/binders.fasta" \
  --type protein \
  --output-dir "$OUT/jobs"

python -m fullFold scan --input-dir "$OUT/jobs" --output-dir "$OUT/results"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No nvidia-smi; skipping dry-run/run. Demonstrating kill -> resume via markers."
  python - "$OUT/jobs" "$OUT/results" <<'PY'
import os, signal, sys, time
from pathlib import Path
from fullFold.config import Config
from fullFold.jobs import claim, scan

cfg = Config(input_dir=Path(sys.argv[1]), output_dir=Path(sys.argv[2]))
jobs, _ = scan(cfg)
assert jobs, 'expected generated jobs'
job = jobs[0]
pid = os.fork()
if pid == 0:
    with claim(job, cfg):
        time.sleep(60)
    os._exit(0)
time.sleep(0.2)
os.kill(pid, signal.SIGKILL)
os.waitpid(pid, 0)
print(f'killed pid {pid}; lock leftover={ (job.state_dir(cfg)/"lock").is_file() }')
again, _ = scan(cfg)
print(f'scan after kill: {len(again)} remaining (lock reclaimable if host+pid dead)')
with claim(job, cfg):
    print('resume claim succeeded')
print('done.json written; next scan skips this job')
left, _ = scan(cfg)
print(f'scan after done: {len(left)} remaining')
PY
  echo "Jobs written to $OUT/jobs"
  exit 0
fi

python -m fullFold run \
  --input-dir "$OUT/jobs" \
  --output-dir "$OUT/results" \
  --model-dir "$MODEL_DIR" \
  --dry-run

echo "Starting run, then sending SIGTERM to demonstrate resume..."
python -m fullFold run \
  --input-dir "$OUT/jobs" \
  --output-dir "$OUT/results" \
  --model-dir "$MODEL_DIR" &
RUN_PID=$!
sleep 8
kill -TERM "$RUN_PID" || true
wait "$RUN_PID" || true
echo "Killed. Rerunning the same command resumes remaining jobs."
python -m fullFold run \
  --input-dir "$OUT/jobs" \
  --output-dir "$OUT/results" \
  --model-dir "$MODEL_DIR"
echo "Jobs written to $OUT/jobs"
