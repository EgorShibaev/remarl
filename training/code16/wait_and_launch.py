"""One-shot login-node handoff from successful code8 to fresh code16.

No LLM scheduler, allocation request, cancellation or automatic retry. The
compute entrypoint independently checks GPU idleness and validates the runtime.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
PARENT = Path('/data/horse/ws/egsh587h-ldlm/remasking/code-nfe8-i8027-20261010-v2')
HOLDER = '2357570'
STEP = '2357570.31'
NODE = 'i8027'
FAILED = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=45).stdout.strip()


def save(name, value):
    path = BASE / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def holder_remaining(detail, now=None):
    assert 'JobState=RUNNING ' in detail and f'NodeList={NODE} ' in detail
    assert 'UserId=egsh587h(' in detail
    end = detail.split('EndTime=', 1)[1].split()[0]
    return dt.datetime.fromisoformat(end).timestamp() - (time.time() if now is None else now) - 600


def parent_state(accounting):
    row = next((line.split('|') for line in accounting.splitlines()
                if line.startswith(STEP + '|')), None)
    assert row is not None, 'Parent step is absent from Slurm accounting'
    state = row[1].split()[0].rstrip('+')
    if state in FAILED:
        raise RuntimeError('Code NFE8 did not finish successfully: ' + str(row))
    if state == 'COMPLETED':
        assert row[2] == '0:0', 'Parent completed with nonzero exit'
        return True
    assert state in {'RUNNING', 'COMPLETING', 'PENDING'}, 'Unexpected parent state: ' + state
    return False


def active_payloads(text):
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().endswith(('.batch', '.extern'))]


def validate_parent(parent=PARENT):
    root = parent / 'source/production'
    receipt = read(root / 'COMPLETE.json')
    assert receipt['accepted_groups'] == 32768 and receipt['optimizer_updates'] == 8192
    assert not list(parent.glob('recovery-failure-*.json'))
    assert not list(root.glob('failure-*.json'))
    metrics = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    assert len(metrics) == 2048
    previous = read(root / 'resolved-recipe.json')['initial_policy_sha256']
    attempted = 0
    for index, metric in enumerate(metrics, 1):
        folder = root / 'batches' / f'{index:06d}'
        assert read(folder / 'complete.json') == metric
        assert metric['batch'] == index and metric['accepted_groups'] == index * 16
        assert metric['optimizer_updates'] == index * 4 and metric['lr'] == 1e-4
        assert metric['behavior_hash'] == previous and metric['replay_error'] <= .003
        previous = metric['new_policy_hash']
        attempts = [json.loads(line) for line in (folder / 'attempts.jsonl').read_text().splitlines()]
        assert len(attempts) == metric['attempted_this_batch']
        rewards = [reward for entry in attempts for reward in entry['rewards']]
        assert len(rewards) == 8 * len(attempts) and all(r in (0, 1) for r in rewards)
        assert abs(sum(rewards) / len(rewards) - metric['all_attempt_accuracy']) < 1e-12
        selected = [entry['question_id'] for entry in attempts if 0 < sum(entry['rewards']) < 8]
        assert len(selected) == len(set(selected)) == 16
        assert [update['pass_index'] for update in metric['updates']] == [1, 2, 3, 4]
        assert all(update['trajectories'] == 128 and update['optimizer_steps'] == 1
                   and set(update['question_ids']) == set(selected) for update in metric['updates'])
        attempted += len(attempts)
        assert attempted == metric['attempted_groups']
    assert receipt['attempted_groups'] == attempted
    tests = {}
    for split, count in (('test_humaneval', 164), ('test_mbpp', 500)):
        for seed in (101, 202, 303):
            name = f'{split}-032768-seed{seed}'
            folder = root / 'evaluations' / name
            result = read(folder / 'summary.json')
            assert result['complete'] and result['held_out'] and result['split'] == split
            assert result['accepted_groups'] == 32768 and result['optimizer_updates'] == 8192
            assert result['nfe'] == 8 and result['seed'] == seed
            assert result['policy_sha256'] == previous
            assert result['count'] == count and result['no_benchmark_exclusions']
            assert result['position_temperature'] == 1 and result['token_temperature'] == 0
            assert result['answer_length'] == 256
            assert result['training_source_sha256'] == sha(parent / 'training-source.sha256')
            rows = [json.loads(line) for line in (folder / 'answers.jsonl').read_text().splitlines()]
            assert [row['index'] for row in rows] == list(range(count))
            assert all(row['reward'] in (0, 1) and len(row['tokens']) == 256 for row in rows)
            assert sum(row['reward'] for row in rows) == result['correct']
            assert abs(result['accuracy'] - result['correct'] / count) < 1e-12
            assert sha(folder / 'answers.jsonl') == result['answers_sha256']
            assert receipt['final_test_counts'][f'{split}-{seed}'] == count
            tests[name] = dict(count=count, correct=result['correct'],
                summary_sha256=sha(folder / 'summary.json'), answers_sha256=result['answers_sha256'])
    assert all(path.with_name('summary.json').exists()
               for path in (root / 'evaluations').glob('*/answers.jsonl')), 'Partial evaluation remains'
    return dict(complete=True, parent_step=STEP, receipt=receipt, policy_sha256=previous,
                metrics_sha256=sha(root / 'metrics.jsonl'), latest_checkpoint_sha256=sha(root / 'latest.pt'),
                final_checkpoint_sha256=sha(root / 'selector-032768.pt'), tests=tests)


def main():
    assert os.getuid() != 0 and command(['id', '-un']) == 'egsh587h'
    assert BASE.name == 'code-nfe16-i8027-20261011'
    fd = os.open(BASE / 'launch.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    deadline = time.monotonic() + 48 * 3600
    try:
        assert not (BASE / 'LAUNCH_REQUESTED.json').exists()
        assert not (BASE / 'source/production').exists()
        assert not (BASE / 'PRODUCTION_STARTED.json').exists()
        command(['sha256sum', '--check', str(BASE / 'launcher.sha256')])
        command(['sha256sum', '--check', str(BASE / 'training-source.sha256')])
        while True:
            remaining = holder_remaining(command(['scontrol', 'show', 'job', HOLDER, '--oneliner']))
            assert remaining > 24 * 3600, 'Less than one day remains in the holder'
            complete = parent_state(command(['sacct', '-j', STEP, '--noheader', '--parsable2',
                                             '--format=JobID,State,ExitCode']))
            payload = active_payloads(command(['squeue', '--steps', '-j', HOLDER,
                                               '--noheader', '--format=%i']))
            save('LAUNCHER_STATUS.json', dict(state='WAITING_FOR_SUCCESS_AND_IDLE_NODE',
                 time=time.time(), pid=os.getpid(), parent_step=STEP, parent_completed=complete,
                 active_payloads=payload, no_automatic_retry=True, poll_seconds=120))
            if complete and not payload:
                gate = validate_parent()
                gate.update(time=time.time(), slurm_exit='0:0')
                save('CODE8_HANDOFF_VALIDATED.json', gate)
                # Recheck identity and idleness after the potentially lengthy audit.
                remaining = holder_remaining(command(['scontrol', 'show', 'job', HOLDER, '--oneliner']))
                assert remaining > 24 * 3600
                assert not active_payloads(command(['squeue', '--steps', '-j', HOLDER,
                                                    '--noheader', '--format=%i']))
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Code NFE8 handoff not ready within 48 hours')
            time.sleep(120)
        wall_seconds = int(min(4 * 86400, remaining))
        days, rest = divmod(wall_seconds, 86400)
        hours, rest = divmod(rest, 3600)
        minutes, seconds = divmod(rest, 60)
        wall = f'{days}-{hours:02}:{minutes:02}:{seconds:02}'
        argv = ['srun', '--jobid=' + HOLDER, '--overlap', '--exact', '-N1', '-w', NODE, '-n1',
                '--gres=gpu:8', '--cpus-per-task=32', '--mem=900G', '--time=' + wall,
                '--job-name=code16', 'bash', str(BASE / 'run.sh')]
        save('LAUNCH_REQUESTED.json', dict(time=time.time(), command=argv, node=NODE,
             owner='egsh587h', holder=HOLDER, parent_step=STEP,
             handoff_receipt_sha256=sha(BASE / 'CODE8_HANDOFF_VALIDATED.json'),
             source_manifest_sha256=sha(BASE / 'training-source.sha256'), no_automatic_retry=True))
        with (BASE / 'training.log').open('x') as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
            save('LAUNCHER_STATUS.json', dict(state='SLURM_STEP_REQUESTED', pid=os.getpid(),
                 srun_pid=process.pid, time=time.time(), no_automatic_retry=True))
            code = process.wait()
        save('LAUNCHER_STATUS.json', dict(state='COMPLETED' if code == 0 else 'FAILED',
             exit_code=code, time=time.time(), no_automatic_retry=True))
        if code:
            raise RuntimeError('NFE16 validation/training failed: ' + str(code))
    except BaseException as exc:
        save('LAUNCHER_FAILURE.json', dict(type=type(exc).__name__, message=str(exc),
                                         time=time.time(), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
