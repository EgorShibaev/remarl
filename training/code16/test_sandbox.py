"""No candidate/reference programs run locally; only trusted command/mocking tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('isolated_seccomp_sandbox',BASE/'protocol/sandbox.py')
sandbox=importlib.util.module_from_spec(spec);spec.loader.exec_module(sandbox)

def test_no_unfiltered_command():
    with pytest.raises(TypeError):sandbox.command(Path('/work'))
    for fd in (None, -1, 0, 1, 2):
        with pytest.raises(AssertionError):sandbox.command(Path('/work'),fd)

def test_keeps_all_other_isolation_and_requires_filter():
    argv=sandbox.command(Path('/scratch/one'),7)
    assert argv[:6]==['bwrap','--unshare-all','--share-net','--unshare-user','--seccomp','7']
    assert argv[argv.index('--cap-drop')+1]=='ALL'
    assert '--new-session' in argv and '--die-with-parent' in argv
    assert '--clearenv' in argv and '--proc' in argv and '--tmpfs' in argv
    assert not any(arg in ('/home','/data','/root') for arg in argv)

def test_missing_filter_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(sandbox,'BASE',tmp_path)
    with pytest.raises(FileNotFoundError):sandbox.filter_path()

def test_corrupt_filter_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(sandbox,'BASE',tmp_path)
    (tmp_path/'network-deny.bpf').write_bytes(b'corrupt')
    (tmp_path/'network-deny.json').write_text(json.dumps(dict(architecture='x86_64',no_fallback=True,bpf_sha256='wrong')))
    with pytest.raises(AssertionError):sandbox.filter_path()

@pytest.mark.parametrize('returncode,stdout,raises',[
    (0,b'{"passed":true,"result":"passed"}',False),
    (1,b'',True),(0,b'not a verdict',True),
    (0,b'{"passed":true,"result":"timed out"}',True),
])
def test_fd_hygiene_and_infrastructure_errors(tmp_path,monkeypatch,returncode,stdout,raises):
    monkeypatch.setattr(sandbox,'BASE',tmp_path);(tmp_path/'scratch').mkdir()
    filt=tmp_path/'filter';filt.write_bytes(b'fake test filter')
    monkeypatch.setattr(sandbox,'filter_path',lambda:filt)
    def popen(argv,**kwargs):
        assert kwargs['stdin']==subprocess.DEVNULL and kwargs['close_fds']
        assert len(kwargs['pass_fds'])==1 and kwargs['pass_fds'][0]>=3
        assert argv[argv.index('--seccomp')+1]==str(kwargs['pass_fds'][0])
        return SimpleNamespace(returncode=returncode,communicate=lambda **kw:(stdout,b'error'))
    monkeypatch.setattr(sandbox.subprocess,'Popen',popen)
    if raises:
        with pytest.raises(Exception):sandbox.run('trusted test only')
        assert list((tmp_path/'execution-failures').glob('*/program.py'))
    else:assert sandbox.run('trusted test only')['passed']

def test_network_and_alternate_execution_syscalls_are_mandatory():
    spec=importlib.util.spec_from_file_location('filter_builder',BASE/'protocol/build_seccomp.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    assert {'socket','socketpair','connect','sendto','sendmsg','sendmmsg',
        'io_uring_setup','io_uring_enter','io_uring_register','pidfd_getfd','setns','unshare'}<=set(builder.DENIED)
