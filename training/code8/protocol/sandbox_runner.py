"""Trusted bwrap supervisor; candidate execution runs in a killable child.

Pinned HF unsafe_execute and its three-second timer are unchanged. Like
check_correctness, the parent independently enforces timeout + one second.
Supervisor/startup failures stay fatal; unexplained exits are not zero rewards.
"""
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time


def execute_child(program, connection, timeout):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        from execute import unsafe_execute
        connection.send_bytes(json.dumps({'phase': 'ready'}).encode())
        results = []
        cleanup_error = None
        try:
            unsafe_execute(program, results, timeout)
        except BaseException as exc:
            if not results:
                raise
            cleanup_error = type(exc).__name__ + ': ' + str(exc)
        if len(results) != 1:
            raise RuntimeError('Official executor returned an invalid verdict count')
        connection.send_bytes(json.dumps(dict(phase='verdict', passed=results[0] == 'passed',
            result=results[0], official_cleanup_error=cleanup_error)).encode())
    except BaseException as exc:
        connection.send_bytes(json.dumps(dict(phase='infrastructure_error',
            error=type(exc).__name__ + ': ' + str(exc))).encode())
    finally:
        connection.close()


def supervise(program, timeout=3.0, worker=execute_child):
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=worker, args=(program, sender, timeout))
    ready = False
    verdict = None
    error = None
    expired = False
    started = time.monotonic()
    try:
        child.start()
        sender.close()
        deadline = started + timeout + 1.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                expired = True
                break
            if receiver.poll(min(.05, remaining)):
                try:
                    message = json.loads(receiver.recv_bytes(1024**2))
                except EOFError:
                    break
                phase = message.get('phase')
                if phase == 'ready' and not ready:
                    ready = True
                elif phase == 'verdict' and ready:
                    verdict = {k: v for k, v in message.items() if k != 'phase'}
                    break
                elif phase == 'infrastructure_error':
                    error = message['error']
                    break
                else:
                    raise RuntimeError('Invalid sandbox child protocol')
            elif not child.is_alive():
                break
    finally:
        if child.pid is not None:
            child.join(.05)
            if child.is_alive():
                child.kill()
                child.join(1)
            if child.is_alive():
                raise RuntimeError('Sandbox child could not be reaped')
        receiver.close()
        sender.close()
    if error is not None:
        raise RuntimeError('Sandbox worker infrastructure failure: ' + error)
    if verdict is not None:
        if (type(verdict.get('passed')) is not bool
                or not isinstance(verdict.get('result'), str)
                or verdict['passed'] != (verdict['result'] == 'passed')
                or not (verdict['result'] in ('passed', 'timed out')
                        or verdict['result'].startswith('failed:'))):
            raise RuntimeError('Invalid official verdict')
        return dict(**verdict, supervisor_timeout=False, child_exitcode=child.exitcode)
    if ready and expired:
        # Controlled parent watchdog, not an unexplained killed supervisor.
        return dict(passed=False, result='timed out', supervisor_timeout=True,
                    child_exitcode=child.exitcode, official_cleanup_error=None)
    raise RuntimeError(f'Sandbox worker exited without verdict: ready={ready}, '
                       f'exit={child.exitcode}')


if __name__ == '__main__':
    sys.path.insert(0, '/runner')
    sys.path.insert(0, '/opt/extra')
    print(json.dumps(supervise(Path('/work/program.py').read_text())))
