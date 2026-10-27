"""Freeze and stage one isolated training workflow, without submitting it."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
REMOTE = '/data/horse/ws/arma647i-ldlm/egor-code-nfe8-i8026-20261009-v2'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (BASE/'SUBMITTED.json').exists(), 'Never modify a submitted workflow'
if (BASE/'DEPLOYED.json').exists():
    (BASE/'DEPLOYED.json').rename(BASE/f'DEPLOYED-{int(time.time())}.json')
def usable(p): return p.is_file() and not p.name.startswith('._') and '__pycache__' not in p.parts
source_files = sorted(p for p in (BASE/'source').rglob('*') if usable(p) and p.name!='source.sha256')
(BASE/'source/source.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(BASE/'source'))+'\n'
                                               for p in source_files))
files = sorted(p for p in BASE.glob('*') if usable(p) and p.suffix in ('.py','.sh','.json'))
files += sorted(p for folder in ('source','protocol','cohort','compatibility-evidence')
                for p in (BASE/folder).rglob('*') if usable(p))
files += [BASE/'parent-source.sha256',BASE/'runtime/checkpoint_manifest.json',BASE/'runtime/sota-selector.pt',BASE/'runtime/uv']
files = sorted(set(files))
(BASE/'training-source.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(BASE))+'\n' for p in files))
(BASE/'deployment-files.txt').write_text('\n'.join(str(p.relative_to(BASE)) for p in files)
                                       +'\ntraining-source.sha256\n')
subprocess.run(['rsync','-a','--files-from='+str(BASE/'deployment-files.txt'),str(BASE)+'/',
                'arma647i@1.alpha:'+REMOTE+'/'],check=True)
subprocess.run(['ssh','-o','BatchMode=yes','-l','arma647i','1.alpha',
                'cd '+REMOTE+' && sha256sum --check training-source.sha256 >/dev/null'],check=True)
(BASE/'DEPLOYED.json').write_text(json.dumps(dict(time=time.time(),node='i8026',holder=2356156,
    remote=REMOTE,training_source_manifest_sha256=sha(BASE/'training-source.sha256'),
    source_manifest_sha256=sha(BASE/'source/source.sha256'),not_training_yet=True),indent=2))
print('DEPLOYED',REMOTE,flush=True)
