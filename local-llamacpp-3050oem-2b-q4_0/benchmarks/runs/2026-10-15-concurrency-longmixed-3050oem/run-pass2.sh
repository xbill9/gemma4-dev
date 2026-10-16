#!/bin/bash
# Finishes pass 2 after run.sh's pass-2 loop stopped advancing (its index was the global i that
# lib.sh's serve() also sets): arms 6..0 in reverse order, arm 7 (e4b kvu q4x-n2) having run.
set -u
cd "$(dirname "$0")"; HERE=$PWD
eval "$(sed -n '/^\. \.\.\/2026/,/^}/p' run.sh)"
DRIVER=$HERE/driver.log
state "PASS 2 RESUME"
for ((k=6; k>=0; k--)); do run_arm 2 ${ARMS[$k]}; done
restore_live
state "RUN DONE"
commit_push $REL "3050oem: long mixed-prompt concurrency, pass 2 in reverse order (raw results)"
