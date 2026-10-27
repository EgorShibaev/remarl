"""Fetch status/receipts and newly completed result shards without recomputation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from deploy import BASE, ROOT, common

SCRIPT = r'''
import gzip,hashlib,io,json,sys,tarfile
from pathlib import Path
b=Path(sys.argv[1]);raw=sys.argv[2]=='raw';existing=json.load(sys.stdin);saved={};shards={}
def capture(p):
 saved[str(p.relative_to(b))]=p.read_bytes()
for pattern in ('STAGED.json','STARTED.json','LAUNCHED.json','STATUS.json','COMPLETE.json',
                'REUSED.json','UPSTREAM.json','GPU_PROBE.json','DECODER_TESTS.json',
                'GRID.json','INPUTS.json','QUEUED.json','source.sha256',
                'GPU_TINY_TESTS.json','GPU_REAL_TESTS.json','QUEUE_STATUS.json',
                'FAILED-*.json','worker*-progress.json','worker*-ready.json'):
 for p in b.glob(pattern):capture(p)
for kind in ('math','code'):
 p=b/kind/'STATUS.json'
 if not p.exists():continue
 capture(p);status=json.loads(saved[str(p.relative_to(b))])
 p=b/kind/'shared/rows.json'
 if raw and p.exists():
  capture(p)
  name=str(p.relative_to(b))
  if existing.get(name)==hashlib.sha256(saved[name]).hexdigest():del saved[name]
 for cell in status['cells']:
  folder=b/kind/'results'/cell['name'];p=folder/'summary.json';capture(p)
  assert json.loads(saved[str(p.relative_to(b))])==cell
  if raw:
   for name,digest in cell['files'].items():
    assert Path(name).name==name
    rel=str((folder/name).relative_to(b))
    if existing.get(rel)!=digest:shards[rel]=digest
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed, tarfile.open(fileobj=compressed,mode='w|') as tar:
 for name in sorted(set(saved)|set(shards)):
  data=saved[name] if name in saved else (b/name).read_bytes()
  if name in shards:assert hashlib.sha256(data).hexdigest()==shards[name]
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
  tar.addfile(info,io.BytesIO(data))
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--suite', choices=('baseline', 'revision', 'extra'), default='baseline')
    args = parser.parse_args()
    if args.suite != 'baseline':
        local_name, remote_name = {
            'revision': ('2026-10-24-revision-adapters', 'revision-adapters-20261024'),
            'extra': ('2026-10-25-extra-baselines', 'extra-baselines-20261025'),
        }[args.suite]
        BASE = BASE.parent / local_name
        ROOT = str(Path(ROOT).parent / remote_name)
    command = 'python3 -c ' + shlex.quote(SCRIPT) + ' ' + shlex.join([ROOT, 'raw' if args.raw else 'status'])
    existing = {}
    if args.raw:
        for kind in ('math', 'code'):
            paths = list((BASE / kind / 'results').glob('*/shard-*.jsonl'))
            rows = BASE / kind / 'shared/rows.json'
            if rows.exists():
                paths.append(rows)
            for path in paths:
                existing[str(path.relative_to(BASE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = subprocess.run(common.SSH + [command], input=json.dumps(existing).encode(),
                            capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as tar:
        for member in tar.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            path = BASE / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
    print((BASE / 'STATUS.json').read_text() if (BASE / 'STATUS.json').exists() else 'No status file yet.')
