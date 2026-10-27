"""Stage isolated adapter sources and run a bounded validation on our holder."""
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[2]
UPSTREAM=PROJECT/'tmp/reeval-20261024/upstream'
ROOT='/data/horse/ws/egsh587h-ldlm/remasking/revision-adapters-20261024'
spec=importlib.util.spec_from_file_location('existing_deploy',BASE.parent/'2026-10-24-baseline-reevaluation/deploy.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
SSH=common.common.SSH


def stage():
    archive=io.BytesIO();manifest={}
    files={n:BASE/n for n in ('adapters.py','test_adapters.py')}
    files['native_model.py']=PROJECT/'outputs/vllm-rollout-study/vllm/vllm/experimental/llada/model.py'
    for repo,names in {
        'WINO':['LLaDA/decoding.py','LLaDA/modeling_llada.py','LLaDA/configuration_llada.py','README.md'],
        'Uncode':['src/generate.py','README.md'],
        'ReMDM':['diffusion.py','README.md'],
    }.items():
        manifest[repo]={'commit':subprocess.check_output(['git','-C',str(UPSTREAM/repo),'rev-parse','HEAD'],text=True).strip()}
        for name in names:
            files['sources/'+repo+'/'+name]=UPSTREAM/repo/name
    with tarfile.open(fileobj=archive,mode='w:gz') as tar:
        for name,p in files.items():tar.add(p,arcname=name)
        data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('UPSTREAM.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    script='''import ast,io,json,os,sys,tarfile
from pathlib import Path
b=Path(sys.argv[1]);assert os.getuid()==2224008
assert not (b/'VALIDATING').exists()
b.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  p=b/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  data=tar.extractfile(m).read()
  if p.suffix=='.py':ast.parse(data)
  p.write_bytes(data)
print(str(b))
'''
    r=subprocess.run(SSH+['python3 -c '+shlex.quote(script)+' '+shlex.quote(ROOT)],input=archive.getvalue(),capture_output=True,timeout=120)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    print(r.stdout.decode())


def validate(real=False):
    # A short diagnostic only. No evaluation workers or training are stopped.
    command=['srun','--jobid=2362070','--overlap','--exact','-N1','-n1','-c2',
             '--mem=32G','--gres=gpu:1','--time=00:10:00','-w','i8022','bash','-s','--',ROOT,'real' if real else 'tiny']
    script=r'''set -euo pipefail
cd "$1"
test "$(id -un)" = egsh587h
test "$(hostname -s)" = i8022
test "$SLURM_JOB_ID" = 2362070
test ! -e VALIDATING
touch VALIDATING
trap 'rm -f VALIDATING' EXIT
module --force purge
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runtime=/tmp/egsh587h-remask-code32-20261022.Rnk1Tx
if test "$2" = real; then
 "$runtime/venv/bin/python" -c 'import torch;free,total=torch.cuda.mem_get_info();assert free>21*1024**3,(free,total)'
 "$runtime/venv/bin/python" -B test_adapters.py --gpu --checkpoint "$runtime/models/LLaDA-8B-Instruct" | tee GPU_REAL_TESTS.json
else
 "$runtime/venv/bin/python" -B test_adapters.py | tee CPU_TESTS.json
 "$runtime/venv/bin/python" -B test_adapters.py --gpu | tee GPU_TINY_TESTS.json
fi
'''
    r=subprocess.run(SSH+[shlex.join(command)],input=script,text=True,capture_output=True,timeout=650)
    print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError(f'validation failed: {r.returncode}')


if __name__=='__main__':
    stage()
    validate('--real' in sys.argv)
