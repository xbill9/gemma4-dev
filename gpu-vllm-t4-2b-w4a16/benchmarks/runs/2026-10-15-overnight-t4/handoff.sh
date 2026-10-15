#!/bin/bash
# Wait for trimmed.sh to finish its W4A16 drafter job, stop it and whatever it started next,
# then run trimmed2.sh.
cd "$(dirname "$0")"
T=$(pgrep -f '^/bin/bash ./trimmed.sh$')
until grep -q 'END q4ct-e4b' trimmed.log; do sleep 3; done
sleep 2   # its summarize.py
kill $T
sleep 2
for p in $(pgrep -f 'python3 conc.py'); do kill $p; done
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done
while pgrep -f 'vllm.entrypoints.openai.api_server' >/dev/null; do sleep 2; done
echo "$(date -u +%FT%TZ) trimmed.sh stopped after q4ct-e4b; trimmed2.sh takes over with the cool-down" >> trimmed.log
exec ./trimmed2.sh
