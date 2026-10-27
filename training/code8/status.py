"""Bounded status/receipt collection; no automatic resubmission."""
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
REMOTE = '/data/horse/ws/arma647i-ldlm/egor-code-nfe8-i8026-20261009-v2'
names = ['RUNTIME_VALIDATED.json','sandbox-validation.json','reference-progress.json',
         'reference-audit-failure.json','source/data-audit.json','source/validation/VALIDATED.json',
         'source/validation/context-gates.json','source/validation/resume-comparison.json',
         'source/production/resolved-recipe.json','source/production/metrics.jsonl',
         'source/production/COMPLETE.json']
dest = BASE/'receipts'; dest.mkdir(exist_ok=True)
for name in names:
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-l','arma647i','1.alpha',
            'test ! -f '+REMOTE+'/'+name+' || cat '+REMOTE+'/'+name],capture_output=True,text=True,timeout=30)
    if result.returncode==0 and result.stdout:
        p=dest/name; p.parent.mkdir(parents=True,exist_ok=True);p.write_text(result.stdout)
probe = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-l','arma647i','1.alpha',
    'squeue --steps -w i8026 -o "%i|%j|%M"; tail -10 '+REMOTE+'/workflow.log'],
    capture_output=True,text=True,timeout=30)
(BASE/'STATUS.json').write_text(json.dumps(dict(time=time.time(),stdout=probe.stdout,stderr=probe.stderr),indent=2))
print(probe.stdout,probe.stderr)
