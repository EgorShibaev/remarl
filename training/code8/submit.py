"""One-shot submit into existing authorized holder, not a new allocation."""
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path(__file__).resolve().parent
REMOTE = '/data/horse/ws/arma647i-ldlm/egor-code-nfe8-i8026-20261009-v2'
assert (BASE/'DEPLOYED.json').exists() and not (BASE/'SUBMITTED.json').exists()
command = ('set -eu; b='+shlex.quote(REMOTE)+'; test -f "$b/runtime/READY"; '
    'test ! -f "$b/workflow-launch.pid"; '
    'test "$(squeue -h -j 2356156 -o %T)" = RUNNING; '
    'test "$(squeue -h -j 2356156 -o %N)" = i8026; '
    'nohup srun --jobid=2356156 --overlap -N1 -n1 -w i8026 '
    '--cpus-per-task=32 --gres=gpu:8 --time=5-00:00:00 '
    '--job-name=kill_it_when_node_needed bash "$b/run.sh" '
    '> "$b/workflow.log" 2>&1 < /dev/null & '
    'task_pid=$!; printf "%s\n" "$task_pid" > "$b/workflow-launch.pid"; printf "%s\n" "$task_pid"')
result = subprocess.run(['ssh','-o','BatchMode=yes','-l','arma647i','1.alpha',command],
                        capture_output=True,text=True,check=True)
(BASE/'SUBMITTED.json').write_text(json.dumps(dict(time=time.time(),holder=2356156,node='i8026',
    run_name='kill_it_when_node_needed',gpus=8,cpus=32,step_walltime='5-00:00:00',
    launcher_output=result.stdout,stderr=result.stderr,remote=REMOTE,
    status='SUBMITTED: CPU/runtime/GPU gates must pass before production'),indent=2))
print(result.stdout,result.stderr)
