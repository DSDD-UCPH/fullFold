"""Bucket-aware planner: exact DP for one GPU, exact-DP-scored local search for several.

Stdlib only, like ``scheduling.py``: it uses that module's cost model (``compilation_us`` /
``inference_us`` per shape) and returns the same ``Group`` structures, so manifests, the
work-stealing overlay and the worker are unchanged.

One GPU (``plan_single_gpu``)
    Jobs are aggregated into inference-unit counts per rounded shape and the production
    recurrence is run on those counts, so the cost is O(K^2) in the number of occupied
    shapes, independent of the job count. Between two occupied shapes only unoccupied shapes
    that could replace the occupied one below (cheaper compile or cheaper inference, and
    Pareto-minimal) are kept as candidates; every other table bucket is provably useless, so
    the optimum equals the one over every bucket.

Several GPUs (``plan_multi_gpu_search``), after the fixed-charge model of
``multi-gpu-bucket-scheduling``
    The state is ``x[gpu][type]`` (type = rounded shape and n_seeds), and a GPU's cost is the
    exact single-GPU DP of its types, so each GPU opens its own buckets and every move is
    scored with the true compile + inference cost. The search needs no GPU ordering:

    1. seeds: optimal contiguous partitions (bisection on the makespan with bracketed cut
       searches) for a few GPU orders, plus a speed-proportional split;
    2. descent on the sorted load vector with moves of type batches, whole bucket groups and
       swaps; candidates are ranked by a fixed-bucket estimate, the best few are scored exactly;
    3. iterated local search whose budget is counted in DP cells, not seconds, so plans are
       identical across runs and machines;
    4. up to ``EXACT_MAX_JOBS`` jobs: exact subset DP (the absolute optimum).

The result is never worse than the best contiguous seed. A valid lower bound on any plan is
reported through ``info['lower_bound_us']``.
"""

from __future__ import annotations

import bisect
import random

from fullFold.scheduling import (
    INF, STANDARD, Group, Gpu, TimingRef, Work, compilation_modifier, compilation_us,
    inference_modifier, inference_us,
)

EXACT_MAX_JOBS = 12        # subset DP up to here: O(G 3^n), 0.7 s for 12 jobs on 8 GPUs
EXHAUSTIVE_K = 45          # occupied shapes per GPU below which a DP is 'small'
EST_COST = 25              # work units (DP cells) charged per fixed-bucket estimate
RANK_R = (40, 30, 30)      # exact evaluations per candidate stage (best by estimate)
BUDGET_CELLS = {0: 3_000_000, 1: 20_000_000, 2: 80_000_000}
PATIENCE = {1: 60, 2: 300}
LB_REF = 1_000_000_000     # virtual-GPU inference time at 1024 tokens for the lower bound
_BIG = 1 << 62


class _Problem:
    """Aggregated instance: jobs -> classes -> types, plus per-GPU-kind cost tables and DP cache."""

    def __init__(
        self, work: list[Work], gpus: list[Gpu], shapes: list[int], ref: TimingRef,
    ):
        self.ref = ref
        n = len(work)
        order = sorted(range(n), key=lambda i: (work[i][0], work[i][2]))
        self.ids = [work[i][2] for i in order]
        toks = [work[i][0] for i in order]
        self.seeds = [work[i][1] for i in order]
        sh = sorted(set(shapes))
        if not sh or sh[-1] < toks[-1]:  # same rule as scheduling.plan_gpu
            i = bisect.bisect_left(ref.buckets, toks[-1])
            sh.append(ref.buckets[i] if i < len(ref.buckets) else ref.csv_max)
            if sh[-1] < toks[-1]:
                sh[-1] = toks[-1]
            sh = sorted(set(sh))
        self.shapes = sh
        self.m = len(sh)
        self.n = n
        self.cls = [bisect.bisect_left(sh, t) for t in toks]
        by_type: dict[tuple[int, int], list[int]] = {}
        for pos, (k, s) in enumerate(zip(self.cls, self.seeds)):
            by_type.setdefault((k, s), []).append(pos)
        keys = sorted(by_type)
        self.type_k = [k for k, _ in keys]
        self.type_s = [s for _, s in keys]
        self.type_jobs = [by_type[key] for key in keys]
        self.type_n = [len(v) for v in self.type_jobs]
        self.T = len(keys)
        kinds: dict[Gpu, int] = {}
        self.kind_of = [kinds.setdefault(g, len(kinds)) for g in gpus]
        self.tables = [
            ([compilation_us(g[0], s, ref) for s in sh],
             [inference_us(g[1], s, ref) for s in sh])
            for g in sorted(kinds, key=kinds.get)
        ]
        self.cmod = [compilation_modifier(s, ref) for s in sh]
        self.imod = [inference_modifier(s, ref) for s in sh]
        self.G = len(gpus)
        self.gpus = list(gpus)
        self.cache: dict = {}
        self._extras: dict = {}
        self.cells = 0
        self.cell_limit = 1 << 60

    def units_by_class(self) -> dict[int, int]:
        units: dict[int, int] = {}
        for k, s in zip(self.cls, self.seeds):
            units[k] = units.get(k, 0) + s
        return units

    def extras(self, k: int, nxt: int) -> tuple[int, ...]:
        """Unoccupied shapes in (k, nxt) that could replace shape k.

        Such a shape can only serve classes up to k, so it matters only if it is cheaper
        than k to compile or to run (the modifiers are GPU independent), and among those
        only the Pareto-minimal ones.
        """
        r = self._extras.get((k, nxt))
        if r is None:
            cm, im = self.cmod, self.imod
            out: list[int] = []
            ck, ik = cm[k], im[k]
            for b in range(k + 1, nxt):
                cb, ib = cm[b], im[b]
                if (cb < ck or ib < ik) and not any(
                        cm[o] <= cb and im[o] <= ib for o in out):
                    out.append(b)
            r = self._extras[(k, nxt)] = tuple(out)
        return r

    def gcost(self, kind: int, items: tuple) -> tuple[int, tuple[int, ...]]:
        """Optimal cost and compiled shapes of one GPU kind for ``items`` = ((class, units), ...)."""
        if not items:
            return 0, ()
        key = (kind, items)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        comp, inf = self.tables[kind]
        cand: list[int] = []
        hist: list[int] = []
        run = 0
        last = len(items) - 1
        last_pos = 0
        for idx, (k, u) in enumerate(items):
            run += u
            if idx == last:
                last_pos = len(cand)
            cand.append(k)
            hist.append(run)
            nxt = items[idx + 1][0] if idx < last else self.m
            for b in self.extras(k, nxt):
                cand.append(b)
                hist.append(run)
        L = len(cand)
        dp = [0] * L
        par = [-1] * L
        for j in range(L):
            sj = cand[j]
            cj = comp[sj]
            ij = inf[sj]
            hj = hist[j]
            best = cj + hj * ij
            bi = -1
            for i in range(j):
                ni = hj - hist[i]
                if ni:
                    v = dp[i] + cj + ni * ij
                    if v < best:
                        best = v
                        bi = i
            dp[j] = best
            par[j] = bi
        self.cells += L * (L - 1) // 2 + L
        jb = last_pos
        bv = dp[jb]
        for j in range(last_pos + 1, L):
            if dp[j] < bv:
                bv = dp[j]
                jb = j
        chosen: list[int] = []
        j = jb
        while j >= 0:
            chosen.append(cand[j])
            j = par[j]
        chosen.reverse()
        res = (bv, tuple(chosen))
        self.cache[key] = res
        return res

    def groups_for(self, kind: int, positions: list[int]) -> list[Group]:
        """Groups of one GPU: every job at the smallest compiled shape that fits it."""
        if not positions:
            return []
        units: dict[int, int] = {}
        for p in positions:
            units[self.cls[p]] = units.get(self.cls[p], 0) + self.seeds[p]
        _, comp = self.gcost(kind, tuple(sorted(units.items())))
        out: dict[int, list[str]] = {}
        for p in positions:
            c = comp[bisect.bisect_left(comp, self.cls[p])]
            out.setdefault(c, []).append(self.ids[p])
        return [Group(shape=self.shapes[c], job_ids=tuple(ids))
                for c, ids in sorted(out.items())]

    def _hardest(self) -> dict[int, tuple[int, str]]:
        """Per class: (seeds, id) of its first job with the most seeds."""
        first: dict[int, tuple[int, str]] = {}
        for pos in range(self.n):
            k, s = self.cls[pos], self.seeds[pos]
            if k not in first or s > first[k][0]:
                first[k] = (s, self.ids[pos])
        return first

    def floor(self) -> tuple[int, str]:
        """Hardest single job at its own shape on its best GPU (same value and job as ``job_floor_us``)."""
        best, best_id = -1, ''
        for k, (s, jid) in sorted(self._hardest().items()):
            c = min(t[0][k] + s * t[1][k] for t in self.tables)
            if c > best:
                best, best_id = c, jid
        return max(best, 0), best_id

    def job_bound(self) -> int:
        """Valid lower bound from one job: its cheapest run over every GPU and every shape that fits.

        ``floor`` can exceed the optimum when the timing table is not monotone (a job may be cheaper
        at a larger shape that compiles faster), so bounds use this minimum over shapes >= its own.
        """
        best = 0
        for k, (s, _) in self._hardest().items():
            c = min(min(t[0][b] + s * t[1][b] for b in range(k, self.m)) for t in self.tables)
            best = max(best, c)
        return best


# ------------------------------------------------------------------ single GPU
def plan_single_gpu(
    work: list[Work], gpu: Gpu, shapes: list[int], ref: TimingRef = STANDARD,
) -> tuple[int, list[Group]]:
    """Optimal plan for one GPU; same optimum as ``scheduling.plan_gpu``, independent of job count."""
    if not work:
        return 0, []
    if gpu[1] <= 0 or gpu[0] < 0:
        return INF, []
    pb = _Problem(work, [gpu], shapes, ref)
    cost, _ = pb.gcost(0, tuple(sorted(pb.units_by_class().items())))
    return cost, pb.groups_for(0, list(range(pb.n)))


# ------------------------------------------------------------------ search state
class _State:
    __slots__ = ('x', 'units', 'loads')

    def __init__(self, g: int):
        self.x: list[dict[int, int]] = [{} for _ in range(g)]
        self.units: list[dict[int, int]] = [{} for _ in range(g)]
        self.loads: list[int] = [0] * g

    def copy(self) -> '_State':
        s = _State.__new__(_State)
        s.x = [dict(d) for d in self.x]
        s.units = [dict(d) for d in self.units]
        s.loads = list(self.loads)
        return s


def _items(units: dict[int, int]) -> tuple:
    return tuple(sorted((k, u) for k, u in units.items() if u > 0))


def _key(loads: list[int]) -> list[int]:
    return sorted(loads, reverse=True)


def _state_from_counts(pb: _Problem, counts: list[dict[int, int]]) -> _State:
    st = _State(pb.G)
    for g in range(pb.G):
        for t, c in counts[g].items():
            if c:
                st.x[g][t] = c
                k = pb.type_k[t]
                st.units[g][k] = st.units[g].get(k, 0) + c * pb.type_s[t]
        st.loads[g] = pb.gcost(pb.kind_of[g], _items(st.units[g]))[0]
    return st


def _pair_eval(pb, st, a, b, out, inn):
    """Loads after moving ``out`` [(type, q)] from GPU a to b and ``inn`` from b to a."""
    ua = dict(st.units[a])
    ub = dict(st.units[b])
    tk, ts = pb.type_k, pb.type_s
    for t, q in out:
        k, d = tk[t], q * ts[t]
        ua[k] = ua.get(k, 0) - d
        ub[k] = ub.get(k, 0) + d
    for t, q in inn:
        k, d = tk[t], q * ts[t]
        ub[k] = ub.get(k, 0) - d
        ua[k] = ua.get(k, 0) + d
    for u in (ua, ub):
        for v in u.values():
            if v < 0:
                return None
    la = pb.gcost(pb.kind_of[a], _items(ua))[0]
    lb = pb.gcost(pb.kind_of[b], _items(ub))[0]
    return ua, ub, la, lb


def _better(old_loads, a, b, la, lb):
    new = list(old_loads)
    new[a], new[b] = la, lb
    return _key(new) < _key(old_loads), new


def _apply_pair(st, a, b, out, inn, ua, ub, la, lb):
    for t, q in out:
        st.x[a][t] -= q
        if st.x[a][t] == 0:
            del st.x[a][t]
        st.x[b][t] = st.x[b].get(t, 0) + q
    for t, q in inn:
        st.x[b][t] -= q
        if st.x[b][t] == 0:
            del st.x[b][t]
        st.x[a][t] = st.x[a].get(t, 0) + q
    st.units[a] = {k: u for k, u in ua.items() if u > 0}
    st.units[b] = {k: u for k, u in ub.items() if u > 0}
    st.loads[a], st.loads[b] = la, lb


def _batch_sizes(cnt: int) -> list[int]:
    s = {1, cnt, max(1, cnt // 2)}
    if cnt > 8:
        s.update((max(1, cnt // 4), max(1, (3 * cnt) // 4)))
    return sorted(s)


def _bucket_groups(pb, st, g):
    """Types of GPU g grouped by the compiled shape that serves them, ascending."""
    items = _items(st.units[g])
    if not items:
        return []
    comp = pb.gcost(pb.kind_of[g], items)[1]
    by_class: dict[int, list] = {}
    for t, c in st.x[g].items():
        by_class.setdefault(pb.type_k[t], []).append((t, c))
    groups: dict[int, list] = {}
    for k in sorted(by_class):
        groups.setdefault(comp[bisect.bisect_left(comp, k)], []).extend(by_class[k])
    return [groups[c] for c in sorted(groups)]


def _compiled_info(pb, st, g):
    items = _items(st.units[g])
    comp = pb.gcost(pb.kind_of[g], items)[1]
    per: dict[int, int] = {}
    for k, u in items:
        c = comp[bisect.bisect_left(comp, k)]
        per[c] = per.get(c, 0) + u
    return comp, per


def _marginal(pb, g, cinfo, k):
    comp = cinfo[0]
    inf = pb.tables[pb.kind_of[g]][1]
    j = bisect.bisect_left(comp, k)
    return inf[comp[j]] if j < len(comp) else inf[k]


def _est_load(pb, g, cinfo, base, plus, minus):
    """Fixed-bucket estimate of GPU g's load after adding ``plus`` and removing ``minus``."""
    comp, per = cinfo
    kc, ki = pb.tables[pb.kind_of[g]]
    load = base
    delta: dict[int, int] = {}
    for k, u in minus.items():
        c = comp[bisect.bisect_left(comp, k)]
        load -= u * ki[c]
        delta[c] = delta.get(c, 0) - u
    opened: set[int] = set()
    for k, u in plus.items():
        j = bisect.bisect_left(comp, k)
        if j < len(comp):
            c = comp[j]
            load += u * ki[c]
            delta[c] = delta.get(c, 0) + u
        else:
            load += u * ki[k]
            if k not in opened:
                opened.add(k)
                load += kc[k]
    for c, d in delta.items():
        if d < 0 and per.get(c, 0) + d <= 0:
            load -= kc[c]
    return load


def _unit_dicts(pb, out, inn):
    o: dict[int, int] = {}
    for t, q in out:
        o[pb.type_k[t]] = o.get(pb.type_k[t], 0) + q * pb.type_s[t]
    i: dict[int, int] = {}
    for t, q in inn:
        i[pb.type_k[t]] = i.get(pb.type_k[t], 0) + q * pb.type_s[t]
    return o, i


def _candidates(pb, st, a, b, stage, top_types, ga, gb, cia, cib, la0, lb0):
    """Moves between critical GPU a and GPU b as (out, inn) lists of (type, count).

    Stage 1: batches of single types (incl. the batch that equalises the two loads) and whole
    bucket groups or runs of adjacent groups. Stage 2: swaps of bucket groups. Stage 3: swaps
    of single types.
    """
    xa, xb = st.x[a], st.x[b]
    ts = pb.type_s
    if stage == 1:
        gap = la0 - lb0
        by_size = sorted(xa, key=lambda t: -xa[t] * ts[t])[:top_types]
        by_ratio = sorted(
            xa, key=lambda t: -_marginal(pb, a, cia, pb.type_k[t])
            / (_marginal(pb, b, cib, pb.type_k[t]) + 1e-9))[:max(6, top_types // 2)]
        for t in dict.fromkeys(by_size + by_ratio):
            qs = set(_batch_sizes(xa[t]))
            if gap > 0:
                k = pb.type_k[t]
                ma = _marginal(pb, a, cia, k)
                mb = _marginal(pb, b, cib, k)
                qb = int(round(gap / (ts[t] * (ma + mb) + 1e-9)))
                if 1 <= qb <= xa[t]:
                    qs.add(qb)
            for q in sorted(qs):
                yield [(t, q)], []
        for i in range(len(ga)):
            run: list = []
            for j in range(i, len(ga)):
                run = run + ga[j]
                yield list(run), []
    elif stage == 2:
        for i in range(len(ga)):
            for j in range(len(gb)):
                yield list(ga[i]), list(gb[j])
    else:
        ta = sorted(xa, key=lambda t: -xa[t] * ts[t])[:top_types]
        tb = sorted(xb, key=lambda t: -xb[t] * ts[t])[:top_types]
        for t in ta:
            for t2 in tb:
                if pb.type_k[t] == pb.type_k[t2] and ts[t] == ts[t2]:
                    continue
                yield [(t, 1)], [(t2, 1)]
                both = min(xa[t], xb[t2])
                if both > 1:
                    yield [(t, both)], [(t2, both)]


def _descend(pb: _Problem, st: _State, max_iters: int | None = None) -> int:
    """Best-improvement descent on the sorted load vector; returns the accepted moves."""
    moves = 0
    ngpu = pb.G
    big0 = max(len(u) for u in st.units) > EXHAUSTIVE_K
    if max_iters is None:
        max_iters = 400 if big0 else 10_000
    for _ in range(max_iters):
        if pb.cells > pb.cell_limit:
            break
        loads = st.loads
        top = max(loads)
        critical = [g for g in range(ngpu) if loads[g] >= top][:3]
        big = max(len(u) for u in st.units) > EXHAUSTIVE_K
        top_types = 10 if big else 14
        targets_all = sorted(range(ngpu), key=lambda g: loads[g])
        cinfo: dict[int, tuple] = {}
        groups_cache: dict[int, list] = {}

        def groups_of(g):
            r = groups_cache.get(g)
            if r is None:
                r = _bucket_groups(pb, st, g)
                if len(r) > 10:  # keep the 10 heaviest groups
                    r = sorted(r, key=lambda grp: -sum(c * pb.type_s[t] for t, c in grp))[:10]
                groups_cache[g] = r
            return r

        best = None
        for stage in (1, 2, 3):
            cands = []
            for a in critical:
                for b in [x for x in targets_all if x != a][:8]:
                    for g in (a, b):
                        if g not in cinfo:
                            cinfo[g] = _compiled_info(pb, st, g)
                    for out, inn in _candidates(
                            pb, st, a, b, stage, top_types, groups_of(a), groups_of(b),
                            cinfo[a], cinfo[b], loads[a], loads[b]):
                        o, i = _unit_dicts(pb, out, inn)
                        la = _est_load(pb, a, cinfo[a], loads[a], i, o)
                        lb = _est_load(pb, b, cinfo[b], loads[b], o, i)
                        pb.cells += EST_COST
                        cands.append(((max(la, lb), min(la, lb)), a, b, out, inn))
            cands.sort(key=lambda c: c[0])
            for _, a, b, out, inn in cands[:RANK_R[stage - 1]]:
                if pb.cells > pb.cell_limit:
                    break
                r = _pair_eval(pb, st, a, b, out, inn)
                if r is None:
                    continue
                ua, ub, la, lb = r
                ok, new = _better(loads, a, b, la, lb)
                if ok:
                    nk = _key(new)
                    if best is None or nk < best[0]:
                        best = (nk, a, b, out, inn, ua, ub, la, lb)
            if best is not None:
                break
        if best is None:
            break
        _, a, b, out, inn, ua, ub, la, lb = best
        _apply_pair(st, a, b, out, inn, ua, ub, la, lb)
        moves += 1
    return moves


def _iterated_search(
    pb: _Problem, st: _State, rng: random.Random, patience: int,
) -> _State:
    """Random batch kicks followed by descent; keeps the best state seen."""
    best = st.copy()
    cur = st.copy()
    stale = 0
    while pb.cells <= pb.cell_limit and stale < patience:
        y = cur.copy()
        for _ in range(rng.randint(1, 3)):
            a = rng.randrange(pb.G)
            if not y.x[a]:
                continue
            t = rng.choice(list(y.x[a]))
            b = rng.randrange(pb.G - 1)
            b += b >= a
            q = rng.randint(1, y.x[a][t])
            r = _pair_eval(pb, y, a, b, [(t, q)], [])
            if r is not None:
                _apply_pair(y, a, b, [(t, q)], [], *r)
        _descend(pb, y)
        stale += 1
        if _key(y.loads) <= _key(cur.loads):
            cur = y
            if _key(y.loads) < _key(best.loads):
                best = y.copy()
                stale = 0
    return best


# ------------------------------------------------------------------ seeds
def _proportional_state(pb: _Problem) -> _State:
    w = [1.0 / pb.gpus[g][1] for g in range(pb.G)]
    tot = sum(w)
    counts: list[dict[int, int]] = [{} for _ in range(pb.G)]
    for t in range(pb.T):
        n = pb.type_n[t]
        ideal = [n * wi / tot for wi in w]
        fl = [int(v) for v in ideal]
        for g in sorted(range(pb.G), key=lambda g: -(ideal[g] - fl[g]))[:n - sum(fl)]:
            fl[g] += 1
        for g in range(pb.G):
            if fl[g]:
                counts[g][t] = fl[g]
    return _state_from_counts(pb, counts)


class _Slicer:
    """Class unit counts of any contiguous job range [i, j) in O(K) from prefix sums."""

    def __init__(self, pb: _Problem):
        cs = [0] * (pb.n + 1)
        for i, s in enumerate(pb.seeds):
            cs[i + 1] = cs[i] + s
        self.cs = cs
        self.ks = sorted(set(pb.cls))
        self.start = {k: bisect.bisect_left(pb.cls, k) for k in self.ks}
        self.end = {k: bisect.bisect_right(pb.cls, k) for k in self.ks}

    def items(self, i: int, j: int) -> tuple:
        out = []
        for k in self.ks:
            lo = max(self.start[k], i)
            hi = min(self.end[k], j)
            if hi > lo:
                out.append((k, self.cs[hi] - self.cs[lo]))
        return tuple(out)


def _lower_bound_us(pb: _Problem) -> int:
    """Fluid bound: one virtual GPU with the cheapest compile/inference ratio, over the total speed.

    Summing T/S_g >= rho_g * (compile modifiers) + (work * inference modifier) over the GPUs and
    relaxing atomicity and duplicated compiles gives T * sum(1/S_g) >= DP of that virtual GPU.
    """
    rho = min(g[0] / g[1] for g in pb.gpus)
    pb.tables.append((
        [compilation_us(int(round(rho * LB_REF)), s, pb.ref) for s in pb.shapes],
        [inference_us(LB_REF, s, pb.ref) for s in pb.shapes],
    ))
    cost = pb.gcost(len(pb.tables) - 1, tuple(sorted(pb.units_by_class().items())))[0]
    inv = sum(1.0 / g[1] for g in pb.gpus)
    return int(cost / (LB_REF * inv) * (1 - 1e-5))


def _contiguous_seed(
    pb: _Problem, order: list[int], sl: _Slicer, lb: int, rel_tol: float = 2e-5,
) -> _State:
    """Contiguous partition for the GPU order ``order``, optimal to ``rel_tol``.

    The single-GPU cost of a job range never decreases when the range grows, so for a trial
    makespan T each GPU greedily takes the longest range that fits (binary search on its cut).
    The cut is monotone in the start and in T, so earlier cuts bracket later searches; T is
    bracketed upward from the lower bound and then bisected.
    """
    n = pb.n
    ngpu = len(order)
    kinds = [pb.kind_of[g] for g in order]
    memo: list[list[tuple[int, int, int]]] = [[] for _ in range(ngpu)]

    def cost(gi, i, j):
        return pb.gcost(kinds[gi], sl.items(i, j))[0]

    def max_cut(gi, i, T):
        if i >= n:
            return n
        lo, hi = i, n
        for i2, T2, c2 in memo[gi]:
            if i2 <= i and T2 <= T and c2 > lo:
                lo = c2
            if i2 >= i and T2 >= T and max(c2, i2) < hi:
                hi = max(c2, i2)
        if hi >= n and cost(gi, i, n) <= T:
            r = n
        else:
            top = hi if hi < n else n - 1
            a, b = lo, top + 1
            while b - a > 1:
                mid = (a + b) // 2
                if cost(gi, i, mid) <= T:
                    a = mid
                else:
                    b = mid
            r = a
        memo[gi].append((i, T, r))
        return r

    def partition(T):
        i = 0
        cuts = [0]
        for gi in range(ngpu):
            i = max_cut(gi, i, T)
            cuts.append(i)
            if i >= n:
                return cuts + [n] * (ngpu + 1 - len(cuts))
        return None

    hi = min(cost(gi, 0, n) for gi in range(ngpu))
    lo = max(lb - 1, 0)
    best = None
    T = max(int(lb * 1.002), lo + 1)
    step = 0.004
    while T < hi:  # gallop up from the lower bound to a feasible makespan
        c = partition(T)
        if c is not None:
            hi, best = T, c
            break
        lo = T
        T = int(lb * (1 + step))
        step *= 3
    if best is None:
        best = partition(hi)
    while hi - lo > max(1, int(hi * rel_tol)):
        mid = (lo + hi) // 2
        c = partition(mid)
        if c is None:
            lo = mid
        else:
            hi, best = mid, c
    counts: list[dict[int, int]] = [{} for _ in range(pb.G)]
    for gi, g in enumerate(order):
        a, b = best[gi], best[gi + 1]
        if b > a:
            for t in range(pb.T):
                c = (bisect.bisect_left(pb.type_jobs[t], b)
                     - bisect.bisect_left(pb.type_jobs[t], a))
                if c:
                    counts[g][t] = c
    return _state_from_counts(pb, counts)


def _gpu_orders(pb: _Problem, how_many: int) -> list[list[int]]:
    """Listed order, slowest first, fastest first, compile-heaviest first (distinct by GPU kind)."""
    idx = list(range(pb.G))
    cands = [
        idx,
        sorted(idx, key=lambda g: -pb.gpus[g][1]),
        sorted(idx, key=lambda g: pb.gpus[g][1]),
        sorted(idx, key=lambda g: -pb.gpus[g][0] / pb.gpus[g][1]),
    ]
    out: list[list[int]] = []
    seen: set = set()
    for o in cands:
        key = tuple(pb.gpus[g] for g in o)
        if key not in seen:
            seen.add(key)
            out.append(o)
    return out[:how_many]


# ------------------------------------------------------------------ exact for tiny instances
def _exact_state(pb: _Problem) -> _State:
    """Absolute optimum by subset DP: best_g(M) = min over S in M of max(best_{g-1}(M - S), cost_g(S))."""
    n, ngpu = pb.n, pb.G
    full = (1 << n) - 1
    unit_maps: list[dict[int, int]] = [{}] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        j = low.bit_length() - 1
        d = dict(unit_maps[mask ^ low])
        d[pb.cls[j]] = d.get(pb.cls[j], 0) + pb.seeds[j]
        unit_maps[mask] = d
    by_kind: dict[int, list[int]] = {}
    cost: list[list[int]] = []
    for g in range(ngpu):
        kind = pb.kind_of[g]
        if kind not in by_kind:
            by_kind[kind] = [0] + [
                pb.gcost(kind, tuple(sorted(unit_maps[mk].items())))[0]
                for mk in range(1, 1 << n)]
        cost.append(by_kind[kind])
    best = list(cost[0])
    choice: list[list[int] | None] = [None]
    for g in range(1, ngpu):
        cg = cost[g]
        new = [_BIG] * (1 << n)
        ch = [0] * (1 << n)
        for mask in range(1 << n):
            b, bs, s = _BIG, 0, mask
            while True:
                v = best[mask ^ s]
                c = cg[s]
                if c > v:
                    v = c
                if v < b:
                    b, bs = v, s
                if s == 0:
                    break
                s = (s - 1) & mask
            new[mask] = b
            ch[mask] = bs
        best = new
        choice.append(ch)
    masks = [0] * ngpu
    rem = full
    for g in range(ngpu - 1, 0, -1):
        masks[g] = choice[g][rem]
        rem ^= masks[g]
    masks[0] = rem
    type_of = {p: t for t in range(pb.T) for p in pb.type_jobs[t]}
    counts: list[dict[int, int]] = [{} for _ in range(ngpu)]
    for g in range(ngpu):
        for p in range(n):
            if masks[g] >> p & 1:
                t = type_of[p]
                counts[g][t] = counts[g].get(t, 0) + 1
    return _state_from_counts(pb, counts)


# ------------------------------------------------------------------ driver
def _realise(pb: _Problem, st: _State) -> tuple[int, list[list[Group]]]:
    pools = [list(v) for v in pb.type_jobs]
    cursor = [0] * pb.T
    taken: list[list[int]] = [[] for _ in range(pb.G)]
    for g in range(pb.G):
        for t, c in st.x[g].items():
            taken[g].extend(pools[t][cursor[t]:cursor[t] + c])
            cursor[t] += c
    groups = [pb.groups_for(pb.kind_of[g], sorted(taken[g])) for g in range(pb.G)]
    makespan = max(pb.gcost(pb.kind_of[g], _items(st.units[g]))[0] for g in range(pb.G))
    return makespan, groups


def lower_bound_us(
    work: list[Work], gpus: list[Gpu], shapes: list[int], ref: TimingRef = STANDARD,
) -> int:
    """Valid lower bound on the makespan of any plan: the larger of the hardest single job (cheapest
    GPU and shape) and the fluid bound of ``_lower_bound_us``."""
    if not work or not gpus:
        return 0
    pb = _Problem(work, gpus, shapes, ref)
    return max(pb.job_bound(), _lower_bound_us(pb))


def plan_multi_gpu_search(
    work: list[Work], gpus: list[Gpu], shapes: list[int], ref: TimingRef = STANDARD,
    effort: int = 1, seed: int = 0, info: dict | None = None,
) -> tuple[int, int, str, list[list[Group]]]:
    """Plan jobs over GPUs; returns ``(makespan_us, floor_us, floor_job, groups per GPU)``.

    ``effort`` scales the search budget: 0 seeds + descent only (very large runs), 1 default,
    2 for up to a few hundred jobs. Up to ``EXACT_MAX_JOBS`` jobs the optimum is exact. The
    budget is counted in DP cells, so the plan is a deterministic function of the inputs.
    ``info`` (optional dict) receives ``stage``, ``seed_us``, ``final_us`` and ``lower_bound_us``.
    A GPU whose probe is unusable (non-positive inference time) is given no jobs.
    """
    if effort not in BUDGET_CELLS:
        raise ValueError(f'effort must be 0, 1 or 2, got {effort!r}')
    if not work:
        return 0, 0, '', [[] for _ in gpus]
    usable = [i for i, g in enumerate(gpus) if g[1] > 0 and g[0] >= 0]
    if not usable:
        return INF, 0, '', [[] for _ in gpus]
    if len(usable) < len(gpus):  # an unusable probe gets no jobs; plan on the others
        makespan, floor, floor_id, sub = plan_multi_gpu_search(
            work, [gpus[i] for i in usable], shapes, ref, effort, seed, info)
        groups: list[list[Group]] = [[] for _ in gpus]
        for i, grs in zip(usable, sub):
            groups[i] = grs
        return makespan, floor, floor_id, groups
    pb = _Problem(work, gpus, shapes, ref)
    floor, floor_id = pb.floor()
    lb = max(pb.job_bound(), _lower_bound_us(pb))
    if info is not None:
        info['lower_bound_us'] = lb
    if pb.G == 1:
        cost = pb.gcost(0, tuple(sorted(pb.units_by_class().items())))[0]
        if info is not None:
            info.update(stage='single', seed_us=cost, final_us=cost)
        return cost, floor, floor_id, [pb.groups_for(0, list(range(pb.n)))]
    if pb.n <= EXACT_MAX_JOBS and effort >= 1:
        st = _exact_state(pb)
        if info is not None:
            info.update(stage='exact', seed_us=max(st.loads), final_us=max(st.loads))
        makespan, groups = _realise(pb, st)
        return makespan, floor, floor_id, groups
    n_orders = 4 if pb.n <= 2000 else (2 if pb.n <= 50_000 else 1)
    sl = _Slicer(pb)
    starts = [_proportional_state(pb)]
    starts += [_contiguous_seed(pb, o, sl, lb) for o in _gpu_orders(pb, n_orders)]
    starts.sort(key=lambda s: _key(s.loads))
    pb.cells = 0
    budget = BUDGET_CELLS[effort]
    best = None
    for st in starts[:2]:
        pb.cell_limit = pb.cells + budget // 3
        _descend(pb, st)
        if best is None or _key(st.loads) < _key(best.loads):
            best = st
    large = pb.T > 150 or pb.n > 5000  # iterated search buys <= 0.03 % there
    if effort >= 1 and not (large and effort == 1):
        rng = random.Random(seed * 1_000_003 + pb.n * 97 + pb.G)
        pb.cell_limit = pb.cells + budget
        best = _iterated_search(pb, best, rng, 15 if large else PATIENCE[effort])
    if info is not None:
        info.update(stage='search', seed_us=max(starts[0].loads), final_us=max(best.loads))
    makespan, groups = _realise(pb, best)
    return makespan, floor, floor_id, groups
