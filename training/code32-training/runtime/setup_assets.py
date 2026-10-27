"""Only download the 14 previously pinned public backbone assets onto node-local SSD."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

root, control = map(Path, sys.argv[1:])
assert root.parent == Path('/tmp') and '-remask-' in root.name
assert root.stat().st_uid == os.getuid()
manifest = json.loads((control / 'checkpoint_manifest.json').read_text())
files = [f for f in manifest['files'] if f['local_path'].startswith('models/')]
assert len(files) == 14 and sum(f['bytes'] for f in files) < 17_000_000_000
assert shutil.disk_usage(root).free > 30 * 1024**3

def fetch(entry):
    relative = Path(entry['local_path'])
    assert relative.parts[:2] == ('models', 'LLaDA-8B-Instruct') and len(relative.parts) == 3
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    url = f"https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/resolve/{manifest['backbone_snapshot']}/{relative.name}"
    count, digest = 0, hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as source, temporary.open('xb') as target:
        while chunk := source.read(8 * 1024**2):
            count += len(chunk)
            assert count <= entry['bytes']
            digest.update(chunk)
            target.write(chunk)
    assert count == entry['bytes'] and digest.hexdigest() == entry['sha256']
    temporary.rename(destination)
    print('VERIFIED', relative.name, count, flush=True)
    return entry

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    verified = list(executor.map(fetch, files))
(control / f'backbone-{os.uname().nodename}.json').write_text(json.dumps(verified, indent=2))
print('BACKBONE_READY', root, flush=True)
