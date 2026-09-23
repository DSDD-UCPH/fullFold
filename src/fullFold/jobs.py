"""Scan a folder of AF3 JSONs and claim output dirs with atomic markers."""

from __future__ import annotations

import contextlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fullFold.config import Config, host_id, sanitise, sha256_file, write_atomic
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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def lock_reclaimable(lock_path: Path, cfg: Config) -> bool:
    if not lock_path.is_file():
        return True
    try:
        data = json.loads(lock_path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    age = time.time() - lock_path.stat().st_mtime
    if age > cfg.stale_lock_seconds:
        return True
    return data.get('host') == host_id() and not _pid_alive(int(data.get('pid') or 0))


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
    text = path.read_text()
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError('top-level JSON must be an object')
    name = raw.get('name')
    if not name:
        raise ValueError('missing name')
    seeds = raw.get('modelSeeds')
    if not isinstance(seeds, list) or not seeds:
        raise ValueError('missing modelSeeds')
    n, exact = count_tokens(raw)
    escalate = (
        cfg.exact_tokens or not exact
        or near_boundary(n, cfg.buckets, cfg.bucket_margin)
    )
    if escalate:
        try:
            n, exact = count_tokens_exact(path), True
        except Exception:
            exact = False
    return Job(
        path=path, name=str(name), sha256=sha256_file(path),
        tokens=n, bucket=bucket_for(n, cfg.buckets),
        seeds=tuple(int(s) for s in seeds), exact=exact,
    )


def scan(cfg: Config) -> tuple[list[Job], list[str]]:
    jobs, rejected = [], []
    root = Path(cfg.input_dir)
    for path in sorted(root.glob('*.json')):
        if not path.is_file():
            continue
        try:
            job = _parse(path, cfg)
        except Exception as e:
            rejected.append(f'{path.name}: {e}')
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
    if lock.is_file() and lock_reclaimable(lock, cfg):
        lock.unlink(missing_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, json.dumps({
            'host': host_id(), 'pid': os.getpid(), 'started_at': time.time(),
        }).encode())
    finally:
        os.close(fd)
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
