"""Scheduler DP and planner vs a local brute-force reference. Nothing imported from scheduler/."""

from __future__ import annotations

import itertools
import random

from fullFold.planner import plan_multi_gpu_search as plan_multi_gpu
from fullFold.scheduling import (
    CSV_BUCKETS, CSV_MAX, FAST, INF, STANDARD, Group, candidate_shapes,
    compilation_modifier, compilation_us, groups_cost_us, inference_modifier,
    inference_us, plan_gpu, plan_round_robin, reference_for,
    share_tails, shape_factor,
)


def brute_gpu(work, gpu, shapes, ref=STANDARD):
    compile_us, inf1024 = gpu
    if not work:
        return 0
    if inf1024 <= 0:
        return INF
    mx = max(w[0] for w in work)
    if not any(s >= mx for s in shapes):
        extra = next((s for s in CSV_BUCKETS if s >= mx), mx)
        shapes = sorted(set(shapes) | {extra})
    else:
        shapes = sorted(set(shapes))
    best = INF
    m = len(shapes)
    for mask in range(1, 1 << m):
        compiled = [shapes[i] for i in range(m) if mask & (1 << i)]
        if max(compiled) < mx:
            continue
        cost = sum(compilation_us(compile_us, s, ref) for s in compiled)
        ok = True
        for w in work:
            assigned = next((s for s in compiled if s >= w[0]), None)
            if assigned is None:
                ok = False
                break
            cost += w[1] * inference_us(inf1024, assigned, ref)
        if ok:
            best = min(best, cost)
    return best


def brute_multi(work, gpus, shapes):
    """Absolute optimum: every assignment of jobs to GPUs, each GPU planned by brute force."""
    n, G = len(work), len(gpus)
    if n == 0:
        return 0
    best = INF
    for assign in itertools.product(range(G), repeat=n):
        loads = [
            brute_gpu([w for w, a in zip(work, assign) if a == g], gpus[g], shapes)
            for g in range(G)]
        best = min(best, max(loads))
    return best


def _w(tokens, seeds=1, name=None):
    return (tokens, seeds, name or f'j{tokens}')


def test_plan_gpu_examples_match_brute_force():
    """Hand-checked cases; each is also compared with the exhaustive optimum."""
    oversized = [_w(50, 1, 'big')]
    two = [_w(128, 1, 'a'), _w(256, 1, 'b')]
    cases = [  # name, work, gpu, shapes, ref, expected compiled shapes
        ('one sequence', [_w(128, 1, 'a')], (10, 1_000_000), [128, 256], STANDARD, [128]),
        ('free compile splits', two, (0, 1_000_000), [128, 256], STANDARD, [128, 256]),
        ('huge compile collapses', two, (10**12, 1_000_000), [128, 256], STANDARD, [256]),
        ('oversized job gets its own shape', oversized, (1, 1_000_000),
         candidate_shapes(oversized, (8, 16, 32), 'ladder'), STANDARD, [50]),
        ('standard compiles twice', two, (55_000, 1_000_000), [128, 256], STANDARD, [128, 256]),
        ('fast curve merges', two, (55_000, 1_000_000), [128, 256], FAST, [256]),
    ]
    for name, work, gpu, shapes, ref, want in cases:
        cost, groups = plan_gpu(work, gpu, shapes, ref)
        assert [g.shape for g in groups] == want, name
        assert cost == brute_gpu(work, gpu, shapes, ref), name
    # explicit arithmetic for the simplest two
    one = plan_gpu([_w(128, 1, 'a')], (10, 1_000_000), [128, 256])[0]
    assert one == compilation_us(10, 128) + inference_us(1_000_000, 128)
    split = plan_gpu(two, (0, 1_000_000), [128, 256])[0]
    assert split == inference_us(1_000_000, 128) + inference_us(1_000_000, 256)
    # the production planner agrees on the fast curve
    gpu = (55_000, 1_000_000)
    assert plan_multi_gpu(two, [gpu], [128, 256], FAST)[0] == (
        compilation_us(55_000, 256, FAST) + 2 * inference_us(1_000_000, 256, FAST))
    # degenerate inputs
    assert plan_gpu([], (1, 1_000_000), [128]) == (0, [])
    assert plan_gpu([_w(128)], (1, 0), [128]) == (INF, [])


def test_plan_matches_brute_force_on_random_instances():
    """Seeds and several GPUs included: the DP and the planner reach the true optimum."""
    rng = random.Random(0)
    for _ in range(15):
        n = rng.randint(1, 6)
        work = sorted(
            (_w(rng.choice([128, 256, 384, 512]), rng.randint(1, 3), f'j{i}')
             for i in range(n)),
            key=lambda w: (w[0], w[2]),
        )
        shapes = [128, 256, 384, 512]
        gpu = (rng.randint(0, 20) * 1000, rng.randint(1, 4) * 1_000_000)
        cost, _ = plan_gpu(work, gpu, shapes)
        assert cost == brute_gpu(work, gpu, shapes)
        gpus = [gpu, (gpu[0] + 3000, gpu[1])]
        ms, _, _, _ = plan_multi_gpu(work, gpus, shapes)
        assert ms == brute_multi(work, gpus, shapes)


def test_reference_tables_lookup_snapping_and_extrapolation():
    assert reference_for('standard') is STANDARD and reference_for('fast') is FAST
    rows = {  # (bucket, inference modifier, compile modifier) from the CSVs
        'standard': [(256, 0.12933754, 0.99578898), (2048, 4.52811282, 1.00479588)],
        'fast': [(256, 0.133333333, 0.96285448), (2048, 4.942764228, 1.005325523)],
    }
    last_compile = {'standard': 1.40297111, 'fast': 1.242444415}
    for name, table in rows.items():
        ref = reference_for(name)
        assert inference_modifier(1024, ref) == compilation_modifier(1024, ref) == 1.0, name
        for bucket, inf, comp in table:
            assert inference_us(1_000_000, bucket, ref) == round(1_000_000 * inf), name
            assert compilation_us(1_000_000, bucket, ref) == round(1_000_000 * comp), name
        # a size missing from the table snaps up to the next listed bucket
        below = ref.buckets[0] + 1
        assert inference_us(1_000_000, below, ref) == inference_us(
            1_000_000, next(b for b in ref.buckets if b >= below), ref), name
        # above the table: compile stays at the last row, inference follows the cubic
        assert ref.csv_max == ref.buckets[-1]
        assert compilation_modifier(6000, ref) == compilation_modifier(ref.csv_max, ref) \
            == last_compile[name], name
        assert inference_modifier(6000, ref) == shape_factor(6000), name
        assert inference_us(1_000_000, 6000, ref) > 2 * inference_us(1_000_000, 1024, ref)
    assert inference_us(1_000_000, 256, FAST) != inference_us(1_000_000, 256)
    assert (CSV_BUCKETS, CSV_MAX) == (STANDARD.buckets, STANDARD.csv_max)


def test_candidate_shapes_free_ladder_fast_and_oversized():
    tiny = [_w(1, 1, 'a'), _w(9, 1, 'b'), _w(16, 1, 'c'), _w(17, 1, 'd')]
    free = candidate_shapes(tiny, (128, 256), 'free')
    assert free == [s for s in CSV_BUCKETS if 5 <= s <= 19]  # table sizes, not multiples of 8
    assert 19 in free and 17 not in free and 24 not in free
    fast = candidate_shapes(tiny, (128, 256), 'free', FAST)
    assert fast == [s for s in FAST.buckets if 8 <= s <= 24]
    assert 10 not in fast and 17 not in fast
    # up to the table's end only table sizes; above it the next multiple of 64
    work = [_w(17, 1, 'a'), _w(50, 1, 'b'), _w(6000, 1, 'c')]
    ladder = candidate_shapes(work, (8, 16, 32, 128), 'ladder')
    free = candidate_shapes(work, (128, 256), 'free')
    for shapes in (free, ladder):
        assert all(s in CSV_BUCKETS or s > CSV_MAX for s in shapes)
        assert 17 not in shapes and 6000 not in shapes and 6016 in shapes
        assert all(s % 64 == 0 for s in shapes if s > CSV_MAX)
    assert 19 in free and 50 in ladder


def test_heterogeneous_gpus_split_work_by_speed():
    work = [_w(128, 1, f'j{i}') for i in range(100)]
    _, _, _, groups = plan_multi_gpu(work, [(0, 100_000_000), (0, 165_000_000)], [128])
    n = [sum(len(g.job_ids) for g in gr) for gr in groups]
    assert sum(n) == 100 and n[0] >= 55  # the 65 % faster GPU takes the larger share
    # two shapes, 800 jobs: loads end up within one inference of each other
    work = [_w(128, 1, f's{i}') for i in range(400)] + [_w(256, 1, f'l{i}') for i in range(400)]
    gpus = [(0, 750_000), (0, 1_320_000)]
    ms, _, _, groups = plan_multi_gpu(work, gpus, [128, 256])
    c0, c1 = (groups_cost_us(g, work, gpu) for g, gpu in zip(groups, gpus))
    assert ms == max(c0, c1) and abs(c0 - c1) <= inference_us(1_320_000, 256)
    assert sum(len(g.job_ids) for gr in groups for g in gr) == 800


def test_share_tails_steals_only_when_the_thief_finishes_sooner():
    def stolen(work, gpus, assignment):
        out = share_tails(assignment, work, [256], gpus)
        return out, [[j for g in gr if g.shared for j in g.job_ids] for gr in out]

    a, b = _w(256, 1, 'a'), _w(256, 1, 'b')
    one_gpu_has_both = [[Group(256, ('a', 'b'))], []]
    # equal GPUs, one job each: nothing to gain
    work = [_w(128, 1, 'a'), _w(128, 1, 'b')]
    out = share_tails([[Group(128, ('a',))], [Group(128, ('b',))]], work, [128],
                      [(0, 1_000_000)] * 2)
    assert [[j for g in gr if g.shared for j in g.job_ids] for gr in out] == [[], []]
    # a 4x faster owner would take longest: the slow idle GPU steals the last job first
    gpus = [(0, 4_000_000), (0, 1_000_000)]
    out, steal = stolen([a, b], gpus, one_gpu_has_both)
    assert steal[1][0] == 'b'
    assert [j for g in out[0] if not g.shared for j in g.job_ids] == ['a', 'b']
    assert groups_cost_us(out[0], [a, b], gpus[0]) == groups_cost_us(
        one_gpu_has_both[0], [a, b], gpus[0])
    # a new bucket is charged its compile once, then both jobs are cheaper to steal
    gpus = [(10_000_000, 4_000_000), (10_000_000, 1_000_000)]
    _, steal = stolen([a, b], gpus, one_gpu_has_both)
    assert steal[1] == ['b', 'a']
    compile_256, inf = compilation_us(10_000_000, 256), inference_us(1_000_000, 256)
    orig_a = compile_256 + inference_us(4_000_000, 256)
    assert compile_256 + 2 * inf < orig_a <= 2 * compile_256 + 2 * inf
    # ...and a compile that is too slow means no steal at all
    _, steal = stolen([a], [(0, 1_000_000), (10**12, 1_000_000)], [[Group(256, ('a',))], []])
    assert steal == [[], []]


def test_round_robin_groups_by_shape():
    work = [_w(128, 1, 'a'), _w(256, 1, 'b'), _w(128, 1, 'c'), _w(256, 1, 'd')]
    for gpu_groups in plan_round_robin(work, 2, [128, 256]):
        shapes = [g.shape for g in gpu_groups]
        assert shapes == sorted(shapes)
