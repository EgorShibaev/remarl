"""Persist bounded static/fake-backbone test receipt before dataset execution."""
import json
from pathlib import Path
import subprocess
import sys
import time

from cohort_hashes import sha
BASE=Path(__file__).resolve().parent
command=[sys.executable,'-m','pytest','-c','/dev/null','-p','no:cacheprovider',
         '--confcutdir='+str(BASE), 'test_batch_grpo.py','test_worker.py',
         'test_evaluate_p2.py','test_paper.py',str(BASE/'test_training.py'),'-q']
result=subprocess.run(command,cwd=BASE/'source',capture_output=True,text=True)
receipt=dict(time=time.time(),passed=result.returncode==0,stdout=result.stdout,stderr=result.stderr,
             kind='Static/mock/fake-backbone tests; no dataset programs executed',
             training_source_manifest_sha256=sha(BASE/'training-source.sha256'))
(BASE/'CPU_PREFLIGHT.json').write_text(json.dumps(receipt,indent=2))
print(result.stdout,result.stderr,flush=True)
raise SystemExit(result.returncode)
