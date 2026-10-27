"""Bounded generation/scoring pipeline with dispatch-order acceptance.

No ML/backend imports: the scheduler can be tested without CUDA. A reservation
is one question-group, never a trajectory. At most ``target - accepted`` groups
are outstanding, so even if every outstanding group is informative we never
generate beyond the exact prefix needed by the synchronous collector.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import threading
import time


class IncompleteBatch(RuntimeError):
    """Retry limit reached; a partial optimizer batch must not be used."""


class UndrainedCollection(RuntimeError):
    """Work did not stop after a failure; the caller must not update/sync policy."""


@dataclass
class Stats:
    reserved: int = 0
    committed: int = 0
    accepted: int = 0
    peak_pending: int = 0
    peak_generation: int = 0
    peak_scoring: int = 0
    generation_work_seconds: float = 0.
    scoring_work_seconds: float = 0.
    wall_seconds: float = 0.

    def to_dict(self):
        return asdict(self)


class Stage:
    def __init__(self, function, *args):
        self.function, self.args = function, args
        self.started = self.finished = None
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.started = time.monotonic()
        try:
            return self.function(*self.args)
        finally:
            with self.lock:
                self.finished = time.monotonic()

    def timed_out(self, now, limit):
        with self.lock:
            return self.started is not None and self.finished is None and now - self.started > limit

    def seconds(self):
        with self.lock:
            return 0. if self.finished is None else self.finished - self.started


@dataclass
class Record:
    ordinal: int
    item: object
    group: object = None
    answers: object = None
    scored: bool = False


def collect_ordered(*, workers, generation_pool, reserve, generate, score, commit,
                    target=16, max_attempts=1024, max_pending=16, score_workers=1,
                    generation_timeout=300., scoring_timeout=600., drain_timeout=300.,
                    stats=None):
    """Return accepted records after ALL generation/scoring work has drained.

    ``reserve`` is called only in the coordinator, in canonical sample order.
    ``commit(record)`` runs in that same order and returns a strict bool. Neither
    worker latency nor scorer latency can choose which questions train. Generation
    for each worker is exclusive; one separate scoring thread is the safe default
    for an existing stateful scorer (its internal sandbox concurrency is unchanged).

    On any exception no result is returned. Outstanding work is cancelled/drained;
    a timeout during drain raises UndrainedCollection. Never update the behavior
    policy concurrently with this function, including its exception cleanup.
    """
    if not workers or target < 1 or max_attempts < target or max_pending < 1:
        raise ValueError("Require workers and positive bounds; attempts must cover the target")
    if score_workers < 1 or min(generation_timeout, scoring_timeout, drain_timeout) <= 0:
        raise ValueError("Invalid scoring concurrency or timeout")
    stats = Stats() if stats is None else stats
    if stats != Stats():
        raise ValueError("Stats must be fresh for each collection")
    started = time.monotonic()
    pending, futures, accepted = {}, {}, []
    free = list(range(len(workers)))
    next_commit = 0
    scoring_pool = ThreadPoolExecutor(max_workers=score_workers, thread_name_prefix="reward")

    def submit(pool, phase, record, worker, function, *args):
        stage = Stage(function, *args)
        futures[pool.submit(stage)] = (phase, record, worker, stage)

    try:
        while len(accepted) < target:
            # Every outstanding reservation might be informative. This bound
            # avoids speculative samples, discarded RNG draws and latency bias.
            while (free and len(pending) < min(max_pending, target - len(accepted))
                   and stats.reserved < max_attempts):
                item = reserve()
                record = Record(stats.reserved, item)
                pending[record.ordinal] = record
                stats.reserved += 1
                worker = free.pop(0)
                submit(generation_pool, "generate", record, worker, generate, workers[worker], item)
                stats.peak_pending = max(stats.peak_pending, len(pending))
                stats.peak_generation = max(stats.peak_generation,
                    sum(p == "generate" for p, *_ in futures.values()))
            if not futures:
                raise IncompleteBatch("Could not fill informative batch; no partial update allowed")
            done, _ = wait(futures, timeout=.05, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            for phase, record, _, stage in futures.values():
                limit = generation_timeout if phase == "generate" else scoring_timeout
                if stage.timed_out(now, limit):
                    raise TimeoutError(f"{phase} timed out at reservation {record.ordinal}")
            # Observe ALL completed failures before committing further records.
            for future in done:
                future.result()
            for future in sorted(done, key=lambda f: futures[f][1].ordinal):
                phase, record, worker, stage = futures.pop(future)
                if phase == "generate":
                    stats.generation_work_seconds += stage.seconds()
                    record.group = future.result()
                    free.append(worker)
                    submit(scoring_pool, "score", record, None, score, record.item, record.group)
                    stats.peak_scoring = max(stats.peak_scoring,
                        sum(p == "score" for p, *_ in futures.values()))
                else:
                    stats.scoring_work_seconds += stage.seconds()
                    record.answers, record.scored = future.result(), True
            while next_commit in pending and pending[next_commit].scored:
                record = pending.pop(next_commit)
                take = commit(record)
                if type(take) is not bool:
                    raise TypeError("commit must return bool")
                stats.committed += 1
                if take:
                    accepted.append(record)
                    stats.accepted += 1
                next_commit += 1
        # No old-policy tasks may survive to optimizer.step / replica.sync.
        assert not pending and not futures
        assert stats.reserved == stats.committed and stats.accepted == target
        return accepted, stats
    finally:
        for future in futures:
            future.cancel()
        scoring_pool.shutdown(wait=False, cancel_futures=True)
        # shutdown(cancel_futures=True) removes queued calls without a worker
        # promoting CANCELLED to CANCELLED_AND_NOTIFIED. Exclude them explicitly:
        # concurrent.futures.wait can otherwise wait the entire drain timeout.
        unfinished = [f for f in futures if not f.cancelled()]
        if unfinished:
            _, running = wait(unfinished, timeout=drain_timeout)
            if running:
                stats.wall_seconds = time.monotonic() - started
                raise UndrainedCollection(f"{len(running)} tasks still running; policy must stay frozen")
        stats.wall_seconds = time.monotonic() - started
