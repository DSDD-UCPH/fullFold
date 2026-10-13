"""Scan a folder of AF3 JSONs and claim output dirs with atomic markers."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fullFold.config import Config, host_id, sanitise, sha256_bytes, write_atomic
from fullFold.tokens import (
    bucket_for, count_tokens, count_tokens_exact, near_boundary,
)


@dataclass(frozen=True)
class Job:
    path: Path
    name: str
    sha256: str
    tokens: int
    bucket: int
    seeds: tuple[int, ...]
    exact: bool

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    @property
    def job_id(self) -> str:
        return sanitise(self.name)

    def state_dir(self, cfg: Config) -> Path:
        return cfg.output_dir / self.job_id / '.af3sched'


# A lock with unreadable content is broken, unless it is being written right now
# (create and write are two steps). This long is more than that takes.
_UNREADABLE_GRACE_S = 5.0

# Job ids that would collide with the scheduler's own files.
_RESERVED_IDS = frozenset({'.', '..', '_af3sched'})

# Locks held by this process, keyed by path -> owning pid. A daemon thread
# refreshes their mtime so that ``stale_lock_seconds`` measures "no heartbeat",
# not "job is old": a job may legitimately run for hours.
_HELD: dict[Path, int] = {}
_HELD_LOCK = threading.Lock()
_BEAT = {'pid': 0, 'interval': 60.0}
_WAKE = threading.Event()  # rouses the heartbeat when its interval shrinks


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:  # exists, but belongs to another user
        return True
    except OSError:
        return False
    return True


def _beat_interval(cfg: Config) -> float:
    return max(0.05, min(60.0, cfg.stale_lock_seconds / 4))


def _heartbeat() -> None:
    me = os.getpid()
    while True:
        _WAKE.wait(_BEAT['interval'])
        _WAKE.clear()
        if _BEAT['pid'] != me:  # forked child: the parent's thread is gone
            return
        with _HELD_LOCK:
            mine = [p for p, owner in _HELD.items() if owner == me]
        for path in mine:
            try:
                os.utime(path)
            except OSError:  # evicted or removed under us
                with _HELD_LOCK:
                    _HELD.pop(path, None)


def _hold(lock: Path, cfg: Config) -> None:
    me = os.getpid()
    with _HELD_LOCK:
        _HELD[lock] = me
        interval = _beat_interval(cfg)
        if _BEAT['pid'] == me:
            if interval < _BEAT['interval']:
                _BEAT['interval'] = interval
                _WAKE.set()
            return
        _BEAT.update(pid=me, interval=interval)
    threading.Thread(target=_heartbeat, name='fullFold-lock-heartbeat',
                     daemon=True).start()


def _unhold(lock: Path) -> None:
    with _HELD_LOCK:
        _HELD.pop(lock, None)


def lock_reclaimable(lock_path: Path, cfg: Config) -> bool:
    """True when the lock's owner is gone: dead pid on this host, or no heartbeat.

    Unreadable content is reclaimable after a short grace period, so a lock that
    is being written right now is not mistaken for a broken one.
    """
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return True
    if age > cfg.stale_lock_seconds:
        return True
    try:
        data = json.loads(lock_path.read_text())
        same_host = data.get('host') == host_id()
        pid = int(data.get('pid') or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return age > min(cfg.stale_lock_seconds, _UNREADABLE_GRACE_S)
    return same_host and not _pid_alive(pid)


def _evict_stale(lock: Path, cfg: Config) -> None:
    """Remove a dead owner's lock so that only one claimant can win.

    The lock is renamed away first (atomic: one contender wins the rename), then
    checked to be the file that was judged stale. If another claimant replaced it
    in between, it is put back.
    """
    try:
        seen = lock.read_bytes()
    except OSError:
        return
    if not lock_reclaimable(lock, cfg):
        return
    grave = lock.with_name(f'lock.stale.{os.getpid()}.{time.time_ns()}')
    try:
        os.rename(lock, grave)
    except OSError:
        return  # another contender evicted it first
    try:
        moved = grave.read_bytes()
    except OSError:
        moved = None
    if moved != seen:
        try:
            os.link(grave, lock)
        except OSError:
            pass
    grave.unlink(missing_ok=True)


def _skip_reason(job: Job, cfg: Config) -> str | None:
    d = job.state_dir(cfg)
    done, failed, lock = d / 'done.json', d / 'failed.json', d / 'lock'
    if done.is_file():
        try:
            prev = json.loads(done.read_text())
        except (OSError, json.JSONDecodeError):
            prev = {}
        if prev.get('sha256') == job.sha256:
            return 'done'
    if failed.is_file() and not cfg.retry_failed:
        return 'failed'
    if lock.is_file() and not lock_reclaimable(lock, cfg):
        return 'locked'
    return None


def _parse(path: Path, cfg: Config) -> Job:
    data = path.read_bytes()  # one read: the hash is of the bytes that were parsed
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise ValueError('top-level JSON must be an object')
    name = raw.get('name')
    if not name:
        raise ValueError('missing name')
    job_id = sanitise(str(name))
    if not job_id or job_id.lower() in _RESERVED_IDS:
        raise ValueError(
            f'name {name!r} is not usable as an output directory name '
            f'(it becomes {job_id!r} after sanitising)')
    seeds = raw.get('modelSeeds')
    if not isinstance(seeds, list) or not seeds:
        raise ValueError('missing modelSeeds')
    if cfg.num_seeds is not None:
        if len(seeds) != 1:
            raise ValueError('--num_seeds requires exactly one modelSeeds entry')
        base = int(seeds[0])
        seeds = [base + i for i in range(int(cfg.num_seeds))]
    n, exact = count_tokens(raw)
    escalate = (
        cfg.exact_tokens or not exact
        or near_boundary(n, cfg.buckets, cfg.bucket_margin)
    )
    if escalate:
        try:
            n, exact = count_tokens_exact(path), True
        except Exception:
            exact = False  # keep the estimate; the caller reports ``exact=False`` jobs
    return Job(
        path=path, name=str(name), sha256=sha256_bytes(data),
        tokens=n, bucket=bucket_for(n, cfg.buckets),
        seeds=tuple(int(s) for s in seeds), exact=exact,
    )


def _input_paths(cfg: Config) -> tuple[list[Path], list[str]]:
    if cfg.json_path is not None:
        path = Path(cfg.json_path)
        if not path.is_file():
            return [], [f'{path}: not a file']
        return [path], []
    root = Path(cfg.input_dir)
    return [p for p in sorted(root.glob('*.json')) if p.is_file()], []


def scan(cfg: Config) -> tuple[list[Job], list[str]]:
    """Parse the inputs; return the jobs still to run and the rejected inputs.

    Two inputs whose names sanitise to the same output directory (compared
    case-insensitively, as AlphaFold 3 lower-cases names) would overwrite each
    other. The first file in sorted order keeps the id; later ones are rejected.
    """
    jobs, rejected = [], []
    paths, missing = _input_paths(cfg)
    rejected.extend(missing)
    owner: dict[str, str] = {}
    for path in paths:
        try:
            job = _parse(path, cfg)
        except Exception as e:
            rejected.append(f'{path.name}: {e}')
            continue
        first = owner.setdefault(job.job_id.lower(), path.name)
        if first != path.name:
            rejected.append(
                f'{path.name}: job id {job.job_id!r} (name {job.name!r}) collides '
                f'with {first}; names must be unique after sanitising, ignoring case')
            continue
        if _skip_reason(job, cfg):
            continue
        jobs.append(job)
    jobs.sort(key=lambda j: (j.bucket, j.tokens, j.job_id, j.sha256))
    return jobs, rejected


def begin_claim(job: Job, cfg: Config) -> dict:
    d = job.state_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    lock = d / 'lock'
    if lock.is_file():
        _evict_stale(lock, cfg)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, json.dumps({
            'host': host_id(), 'pid': os.getpid(), 'started_at': time.time(),
        }).encode())
    finally:
        os.close(fd)
    _hold(lock, cfg)
    return {}


def try_begin_claim(job: Job, cfg: Config) -> tuple[dict | None, str | None]:
    """Claim ``job``, or ``(None, reason)`` if it is done, failed, or locked."""
    reason = _skip_reason(job, cfg)
    if reason:
        return None, reason
    try:
        return begin_claim(job, cfg), None
    except FileExistsError:
        return None, 'locked'


def release_claim(job: Job, cfg: Config) -> None:
    """Give a claim back without a verdict (the job never ran or was interrupted)."""
    lock = job.state_dir(cfg) / 'lock'
    _unhold(lock)
    lock.unlink(missing_ok=True)


def end_claim(job: Job, cfg: Config, extra: dict, error: BaseException | None = None) -> None:
    d = job.state_dir(cfg)
    lock, done, failed = d / 'lock', d / 'done.json', d / 'failed.json'
    try:
        if error is None:
            write_atomic(done, json.dumps({
                'sha256': job.sha256, 'tokens': job.tokens, 'bucket': job.bucket,
                'seeds': list(job.seeds), **extra,
            }, indent=2))
            failed.unlink(missing_ok=True)
        else:
            tb = traceback.format_exc()[-2000:]
            if not tb.strip() or tb.strip() == 'NoneType: None':
                tb = str(error)
            write_atomic(failed, json.dumps({
                'sha256': job.sha256, 'error': str(error), 'traceback': tb,
            }, indent=2))
    finally:
        _unhold(lock)
        lock.unlink(missing_ok=True)


@contextlib.contextmanager
def claim(job: Job, cfg: Config) -> Iterator[dict]:
    extra = begin_claim(job, cfg)
    err: BaseException | None = None
    try:
        yield extra
    except BaseException as e:
        err = e
        raise
    finally:
        end_claim(job, cfg, extra, error=err)
