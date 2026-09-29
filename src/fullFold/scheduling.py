"""Shape-selecting DP for AF3 GPU assignment.

Cost of compiling shape L and running k inferences at L:

    compilation_us(R, L) + k * inference_us(S, L)

R and S come from the 1024-token probe. Per-bucket cost uses a timing
reference through bucket 5216. ``standard`` reads ``data/reference_timings.csv``.
``fast`` reads ``data/reference_timing_fast.csv`` (Anthropic fast-mode compile
and inference modifiers). Inference above 5216 uses the tokamax cubic; compile
above 5216 uses that reference's last CSV compile modifier. Integer
microseconds only.
"""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path

INF = 10**18
Work = tuple[int, int, str]  # tokens, n_seeds, job_id
Gpu = tuple[int, int]        # compile_us_1024, infer_us_1024
FREE_ALIGN = 8  # --bucket-mode=free extras above CSV_MAX round up to this
SHAPE_REF = 1024
_F_A, _F_B, _F_C = 0.33, 0.42, 0.27


@dataclass(frozen=True)
class TimingRef:
    """Compile and inference modifiers for one measured reference."""

    name: str
    inf: dict[int, float]
    comp: dict[int, float]
    buckets: tuple[int, ...]

    @property
    def csv_max(self) -> int:
        return self.buckets[-1]

    @property
    def bucket_set(self) -> frozenset[int]:
        return frozenset(self.buckets)


def _load_reference(filename: str, name: str) -> TimingRef:
    path = Path(__file__).parent / 'data' / filename
    inf: dict[int, float] = {}
    comp: dict[int, float] = {}
    with path.open(newline='') as fh:
        for row in csv.DictReader(fh):
            bucket = int(row['Bucket_size'])
            inf[bucket] = float(row['Inference_modifier'])
            comp[bucket] = float(row['Compilation_modifier'])
    return TimingRef(name, inf, comp, tuple(sorted(inf)))


STANDARD = _load_reference('reference_timings.csv', 'standard')
FAST = _load_reference('reference_timing_fast.csv', 'fast')
_BY_NAME = {STANDARD.name: STANDARD, FAST.name: FAST}

# Standard-table aliases. Callers that pass ``ref`` see the selected table.
CSV_BUCKETS = STANDARD.buckets
CSV_MAX = STANDARD.csv_max


def reference_for(name: str) -> TimingRef:
    """``standard`` or ``fast`` (Anthropic fast-mode modifiers)."""
    ref = _BY_NAME.get(name)
    if ref is None:
        known = ', '.join(sorted(_BY_NAME))
        raise ValueError(f'unknown timing reference {name!r}; expected {known}')
    return ref


def shape_factor(bucket: int) -> float:
    """f(b/1024). Clamped positive so tiny synthetic test shapes stay ordered."""
    u = bucket / SHAPE_REF
    f = 1.0 + _F_A * (u - 1.0) + _F_B * (u * u - 1.0) + _F_C * (u * u * u - 1.0)
    return f if f > 1e-6 else 1e-6


def _snap_csv(bucket: int, ref: TimingRef = STANDARD) -> int:
    i = bisect.bisect_left(ref.buckets, bucket)
    if i < len(ref.buckets):
        return ref.buckets[i]
    return ref.csv_max


def inference_modifier(bucket: int, ref: TimingRef = STANDARD) -> float:
    if bucket in ref.inf:
        return ref.inf[bucket]
    if bucket > ref.csv_max:
        return shape_factor(bucket)
    return ref.inf[_snap_csv(bucket, ref)]


def compilation_modifier(bucket: int, ref: TimingRef = STANDARD) -> float:
    if bucket in ref.comp:
        return ref.comp[bucket]
    if bucket > ref.csv_max:
        return ref.comp[ref.csv_max]
    return ref.comp[_snap_csv(bucket, ref)]


def inference_us(
    infer_us_1024: int, bucket: int, ref: TimingRef = STANDARD,
) -> int:
    """Predicted inference microseconds at ``bucket`` for this GPU."""
    if infer_us_1024 <= 0:
        return 0
    return max(1, int(round(infer_us_1024 * inference_modifier(bucket, ref))))


def compilation_us(
    compile_us_1024: int, bucket: int, ref: TimingRef = STANDARD,
) -> int:
    """Predicted compile microseconds at ``bucket`` for this GPU."""
    if compile_us_1024 <= 0:
        return 0
    return max(0, int(round(compile_us_1024 * compilation_modifier(bucket, ref))))


@dataclass(frozen=True)
class Group:
    shape: int
    job_ids: tuple[str, ...]
    shared: bool = False


def _ceil_multiple(n: int, k: int = FREE_ALIGN) -> int:
    if n <= 0:
        return 0
    return ((n + k - 1) // k) * k


def _cover(tokens: int, mode: str, ref: TimingRef = STANDARD) -> int:
    if tokens <= ref.csv_max:
        return _snap_csv(tokens, ref)
    if mode == 'free':
        return _ceil_multiple(tokens)
    return tokens


def candidate_shapes(
    work: list[Work], ladder: tuple[int, ...] | list[int], mode: str,
    ref: TimingRef = STANDARD,
) -> list[int]:
    tokens = [w[0] for w in work]
    if not tokens:
        return []
    in_range = [t for t in tokens if t <= ref.csv_max]
    over = [t for t in tokens if t > ref.csv_max]
    out: set[int] = set()
    if in_range:
        lo, hi = _snap_csv(min(in_range), ref), _snap_csv(max(in_range), ref)
        if mode == 'ladder':
            out.update(s for s in ladder if s in ref.bucket_set and lo <= s <= hi)
            for t in in_range:
                cover = _snap_csv(t, ref)
                if cover not in out:
                    out.add(cover)
        else:
            out.update(s for s in ref.buckets if lo <= s <= hi)
    for t in over:
        out.add(_cover(t, mode, ref))
    return sorted(out)


def _shape_for(
    tokens: int, shapes: list[int], ref: TimingRef = STANDARD,
) -> int:
    for s in shapes:
        if s >= tokens:
            return s
    return _cover(tokens, 'free', ref) if tokens <= ref.csv_max else tokens


def _hist(work: list[Work], shapes: list[int]) -> list[int]:
    return [sum(w[1] for w in work if w[0] <= s) for s in shapes]


def _assign(work: list[Work], compiled: list[int]) -> list[Group]:
    groups, lo = [], 0
    for s in compiled:
        ids = tuple(w[2] for w in work if lo < w[0] <= s or (lo == 0 and w[0] <= s))
        if ids:
            groups.append(Group(shape=s, job_ids=ids))
        lo = s
    return groups


def plan_gpu(
    work: list[Work], gpu: Gpu, shapes: list[int], ref: TimingRef = STANDARD,
) -> tuple[int, list[Group]]:
    compile_us_1024, infer_us_1024 = gpu
    if not work:
        return 0, []
    if infer_us_1024 <= 0 or compile_us_1024 < 0:
        return INF, []
    mx = max(w[0] for w in work)
    mn = min(w[0] for w in work)
    if not any(s >= mx for s in shapes):
        extra = _snap_csv(mx, ref) if mx <= ref.csv_max else mx
        shapes = sorted(set(shapes) | {extra})
    else:
        shapes = sorted(set(shapes))
    m, hist = len(shapes), _hist(work, shapes)
    inf_at = [inference_us(infer_us_1024, s, ref) for s in shapes]
    compile_at = [compilation_us(compile_us_1024, s, ref) for s in shapes]
    dp = [INF] * m
    parent = [-1] * m
    for j, s in enumerate(shapes):
        if s < mn or hist[j] == 0:
            continue
        dp[j] = compile_at[j] + hist[j] * inf_at[j]
        parent[j] = -1
        for i in range(j):
            n_new = hist[j] - hist[i]
            if dp[i] >= INF or n_new == 0:
                continue
            cost = dp[i] + compile_at[j] + n_new * inf_at[j]
            if cost < dp[j]:
                dp[j], parent[j] = cost, i
    best_j, best = min(
        ((j, dp[j]) for j, s in enumerate(shapes) if s >= mx),
        key=lambda x: (x[1], x[0]),
        default=(-1, INF),
    )
    if best_j < 0:
        return INF, []
    compiled, j = [], best_j
    while j >= 0:
        compiled.append(shapes[j])
        j = parent[j]
    compiled.reverse()
    return best, _assign(work, compiled)


def _split_points(
    work: list[Work], shapes: list[int], n_gpus: int, threshold: int,
    ref: TimingRef = STANDARD,
) -> list[int]:
    n = len(work)
    if n <= threshold:
        return list(range(n + 1))
    pts = {0, n}
    idx = 0
    while idx < n:
        b = _shape_for(work[idx][0], shapes, ref)
        j = idx
        while j < n and _shape_for(work[j][0], shapes, ref) == b:
            j += 1
        pts.add(idx)
        span = j - idx
        if span > 1:
            for t in range(1, n_gpus):
                pts.add(idx + (span * t) // n_gpus)
        idx = j
    return sorted(pts)


def job_floor_us(
    work: list[Work], gpus: list[Gpu], shapes: list[int],
    ref: TimingRef = STANDARD,
) -> tuple[int, str]:
    if not work or not gpus:
        return 0, ''
    best_id, best = '', -1
    for w in work:
        s = _shape_for(w[0], shapes, ref)
        c = min(
            compilation_us(g[0], s, ref) + w[1] * inference_us(g[1], s, ref)
            for g in gpus)
        if c > best:
            best, best_id = c, w[2]
    return best, best_id


def _plan_slice(
    work: list[Work], gpu: Gpu, shapes: list[int], ref: TimingRef = STANDARD,
) -> tuple[int, list[Group]]:
    if not work:
        return 0, []
    return plan_gpu(work, gpu, shapes, ref)


def _best_split(
    work: list[Work], gpu_l: Gpu, gpu_r: Gpu, shapes: list[int],
    left: int, right: int, ref: TimingRef = STANDARD,
) -> int:
    """Index in ``[left, right]`` minimising max(left GPU cost, right GPU cost)."""
    if left >= right:
        return left

    def score(i: int) -> tuple[int, int]:
        c_l, _ = _plan_slice(work[left:i], gpu_l, shapes, ref)
        c_r, _ = _plan_slice(work[i:right], gpu_r, shapes, ref)
        return max(c_l, c_r), i

    lo, hi = left, right
    while lo < hi:
        mid = (lo + hi) // 2
        c_l, _ = _plan_slice(work[left:mid], gpu_l, shapes, ref)
        c_r, _ = _plan_slice(work[mid:right], gpu_r, shapes, ref)
        if c_l < c_r:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    improved = True
    while improved:
        improved = False
        for j in (i - 1, i + 1):
            if left <= j <= right and score(j) < score(i):
                i = j
                improved = True
                break
    return i


def rebalance_contiguous(
    work: list[Work],
    gpus: list[Gpu],
    shapes: list[int],
    assignment: list[list[Group]],
    ref: TimingRef = STANDARD,
) -> tuple[int, list[list[Group]]] | None:
    """Move contiguous split points so predicted GPU times match as closely as possible."""
    G = len(gpus)
    if G < 2 or not work:
        return None
    bounds = [0]
    for gr in assignment:
        bounds.append(bounds[-1] + sum(len(g.job_ids) for g in gr if not g.shared))
    if bounds[-1] != len(work):
        return None
    for _ in range(G):
        for b in range(1, G):
            bounds[b] = _best_split(
                work, gpus[b - 1], gpus[b], shapes, bounds[b - 1], bounds[b + 1],
                ref)
    out: list[list[Group]] = []
    loads: list[int] = []
    for g in range(G):
        cost, groups = _plan_slice(
            work[bounds[g]:bounds[g + 1]], gpus[g], shapes, ref)
        out.append(groups)
        loads.append(cost)
    return max(loads) if loads else 0, out


def plan_multi_gpu(
    work: list[Work],
    gpus: list[Gpu],
    shapes: list[int],
    exact_split_threshold: int = 512,
    ref: TimingRef = STANDARD,
) -> tuple[int, int, str, list[list[Group]]]:
    """Returns (makespan_us, floor_us, floor_job_id, per_gpu_groups)."""
    G = len(gpus)
    floor, floor_id = job_floor_us(work, gpus, shapes, ref)
    if not work:
        return 0, 0, '', [[] for _ in gpus]
    if G == 0:
        return INF, floor, floor_id, []
    if G == 1:
        cost, groups = plan_gpu(work, gpus[0], shapes, ref)
        return cost, floor, floor_id, [groups]
    points = _split_points(work, shapes, G, exact_split_threshold, ref)
    P = len(points)
    costs: list[list[list[int]]] = []
    groups_tbl: list[list[list[list[Group] | None]]] = []
    for g in range(G):
        row_c = [[INF] * P for _ in range(P)]
        row_g: list[list[list[Group] | None]] = [[None] * P for _ in range(P)]
        for i in range(P):
            for j in range(i + 1, P):
                c, gr = plan_gpu(work[points[i]:points[j]], gpus[g], shapes, ref)
                row_c[i][j], row_g[i][j] = c, gr
        costs.append(row_c)
        groups_tbl.append(row_g)
    dp = [[INF] * P for _ in range(G + 1)]
    split = [[0] * P for _ in range(G + 1)]
    dp[0][0] = 0
    for k in range(1, G + 1):
        for j in range(P):
            for i in range(j + 1):
                gpu_cost = 0 if i == j else costs[k - 1][i][j]
                val = max(dp[k - 1][i], gpu_cost)
                if val < dp[k][j]:
                    dp[k][j] = val
                    split[k][j] = i
    assignment: list[list[Group]] = [[] for _ in range(G)]
    j = P - 1
    for k in range(G, 0, -1):
        i = split[k][j]
        if i < j:
            assignment[k - 1] = groups_tbl[k - 1][i][j] or []
        j = i
    makespan = dp[G][P - 1]
    refined = rebalance_contiguous(work, gpus, shapes, assignment, ref)
    if refined is not None:
        makespan, assignment = refined
    return makespan, floor, floor_id, assignment


def groups_cost_us(
    groups: list[Group], work: list[Work], gpu: Gpu, ref: TimingRef = STANDARD,
) -> int:
    """Compile + inference microseconds for an already-assigned GPU plan."""
    compile_us_1024, infer_us_1024 = gpu
    if infer_us_1024 <= 0 or compile_us_1024 < 0:
        return INF
    by_id = {w[2]: w for w in work}
    total = 0
    for g in groups:
        if g.shared:
            continue
        k = sum(by_id[jid][1] for jid in g.job_ids if jid in by_id)
        total += compilation_us(compile_us_1024, g.shape, ref) + k * inference_us(
            infer_us_1024, g.shape, ref)
    return total


def _primary_job_ids(groups: list[Group]) -> list[str]:
    ids: list[str] = []
    for g in groups:
        if not g.shared:
            ids.extend(g.job_ids)
    return ids


def _steal_ids(primary: list[list[str]], gpu_i: int) -> list[str]:
    others = [list(reversed(primary[j])) for j in range(len(primary)) if j != gpu_i]
    if not others:
        return []
    steal: list[str] = []
    for k in range(max(len(o) for o in others)):
        for o in others:
            if k < len(o):
                steal.append(o[k])
    return steal


def _prefix_finish_us(
    groups: list[Group], work: list[Work], gpu: Gpu, ref: TimingRef = STANDARD,
) -> dict[str, int]:
    compile_us_1024, infer_us_1024 = gpu
    by_id = {w[2]: w for w in work}
    t = 0
    out: dict[str, int] = {}
    for g in groups:
        if g.shared:
            continue
        t += compilation_us(compile_us_1024, g.shape, ref)
        for jid in g.job_ids:
            w = by_id.get(jid)
            k = w[1] if w else 1
            t += k * inference_us(infer_us_1024, g.shape, ref)
            out[jid] = t
    return out


def _append_shared(
    groups: list[Group],
    steal: list[str],
    by_id: dict[str, Work],
    shapes: list[int],
    gpu: Gpu,
    leftover_us: int,
    orig_finish: dict[str, int],
    ref: TimingRef = STANDARD,
) -> None:
    compiled = {g.shape for g in groups}
    thief_elapsed = leftover_us
    compile_us_1024, infer_us_1024 = gpu
    cur_s: int | None = None
    cur: list[str] = []
    for jid in steal:
        w = by_id.get(jid)
        if w is None:
            continue
        orig = orig_finish.get(jid)
        if orig is None:
            continue
        tok, n_seeds, _ = w
        s = next((x for x in sorted(compiled) if x >= tok), None)
        if s is None:
            s = _shape_for(tok, shapes, ref)
        extra_compile = 0 if s in compiled else compilation_us(
            compile_us_1024, s, ref)
        job_inf = n_seeds * inference_us(infer_us_1024, s, ref)
        thief_finish = thief_elapsed + extra_compile + job_inf
        if thief_finish >= orig:
            continue
        if cur and s != cur_s:
            groups.append(Group(shape=cur_s, job_ids=tuple(cur), shared=True))
            cur = []
        cur_s = s
        cur.append(jid)
        thief_elapsed = thief_finish
        compiled.add(s)
    if cur and cur_s is not None:
        groups.append(Group(shape=cur_s, job_ids=tuple(cur), shared=True))


def share_tails(
    assignment: list[list[Group]],
    work: list[Work],
    shapes: list[int],
    gpus: list[Gpu],
    ref: TimingRef = STANDARD,
) -> list[list[Group]]:
    """Append other GPUs' jobs, last job first, so an idle GPU can keep working.

    A steal is kept only if the thief would finish that job sooner than the
    original GPU (leftover primary work + compile if the bucket is new).
    Atomic claims skip a job another GPU already took. Predicted makespan uses
    primary groups only.
    """
    G = len(assignment)
    if G < 2:
        return assignment
    by_id = {w[2]: w for w in work}
    primary = [_primary_job_ids(gr) for gr in assignment]
    if not any(primary):
        return assignment
    orig_finish: dict[str, int] = {}
    leftover: list[int] = []
    for i, gr in enumerate(assignment):
        leftover.append(groups_cost_us(gr, work, gpus[i], ref))
        orig_finish.update(_prefix_finish_us(gr, work, gpus[i], ref))
    out: list[list[Group]] = []
    for i in range(G):
        groups = [g for g in assignment[i] if not g.shared]
        _append_shared(
            groups, _steal_ids(primary, i), by_id, shapes, gpus[i],
            leftover[i], orig_finish, ref)
        out.append(groups)
    return out


def plan_round_robin(
    work: list[Work], n_gpus: int, shapes: list[int], ref: TimingRef = STANDARD,
) -> list[list[Group]]:
    if n_gpus <= 0:
        return []
    buckets: list[list[Work]] = [[] for _ in range(n_gpus)]
    for i, w in enumerate(work):
        buckets[i % n_gpus].append(w)
    out = []
    for b in buckets:
        compiled = sorted({_shape_for(w[0], shapes, ref) for w in b})
        out.append(_assign(b, compiled) if b else [])
    return out
