"""Scheduler DP vs a local brute-force reference. Nothing imported from scheduler/."""

from __future__ import annotations

import random

from fullFold.scheduling import (
    CSV_BUCKETS, CSV_MAX, INF, Group, candidate_shapes, compilation_modifier,
    compilation_us, groups_cost_us, inference_modifier, inference_us, plan_gpu,
    plan_multi_gpu, plan_round_robin, share_tails, shape_factor,
)


def brute_gpu(work, gpu, shapes):
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
        cost = sum(compilation_us(compile_us, s) for s in compiled)
        ok = True
        for w in work:
            assigned = next((s for s in compiled if s >= w[0]), None)
            if assigned is None:
                ok = False
                break
            cost += w[1] * inference_us(inf1024, assigned)
        if ok:
            best = min(best, cost)
    return best


def compositions(n, g):
    if g == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, g - 1):
            yield (i,) + rest


def brute_multi(work, gpus, shapes):
    n, G = len(work), len(gpus)
    if n == 0:
        return 0
    best = INF
    for parts in compositions(n, G):
        idx, loads = 0, []
        for g, p in enumerate(parts):
            block = work[idx:idx + p]
            idx += p
            loads.append(0 if not block else brute_gpu(block, gpus[g], shapes))
        best = min(best, max(loads))
    return best


def _w(tokens, seeds=1, name=None):
    return (tokens, seeds, name or f'j{tokens}')


def test_single_sequence():
    work = [_w(128, 1, 'a')]
    gpu, shapes = (10, 1_000_000), [128, 256]
    cost, groups = plan_gpu(work, gpu, shapes)
    assert cost == brute_gpu(work, gpu, shapes)
    assert cost == compilation_us(10, 128) + inference_us(1_000_000, 128)
    assert [g.shape for g in groups] == [128]


def test_r0_splits():
    work = [_w(128, 1, 'a'), _w(256, 1, 'b')]
    gpu, shapes = (0, 1_000_000), [128, 256]
    cost, groups = plan_gpu(work, gpu, shapes)
    assert cost == brute_gpu(work, gpu, shapes)
    assert cost == inference_us(1_000_000, 128) + inference_us(1_000_000, 256)
    assert len(groups) == 2


def test_large_r_collapses():
    work = [_w(128, 1, 'a'), _w(256, 1, 'b')]
    gpu, shapes = (10**12, 1_000_000), [128, 256]
    cost, groups = plan_gpu(work, gpu, shapes)
    assert cost == brute_gpu(work, gpu, shapes)
    assert len(groups) == 1
    assert groups[0].shape == 256


def test_seed_multiplicity():
    a = [_w(128, 5, 'a')]
    b = [_w(128, 1, 'a')]
    gpu, shapes = (100, 1_000_000), [128]
    ca, _ = plan_gpu(a, gpu, shapes)
    cb, _ = plan_gpu(b, gpu, shapes)
    r = compilation_us(100, 128)
    assert ca - r == 5 * (cb - r)


def test_one_compile_per_shape():
    work = [_w(128, 1, 'a'), _w(128, 1, 'b'), _w(128, 1, 'c')]
    _, groups = plan_gpu(work, (5, 1_000_000), [128, 256])
    shapes = [g.shape for g in groups]
    assert len(shapes) == len(set(shapes))


def test_oversized_gets_own_shape():
    work = [_w(50, 1, 'big')]
    shapes = candidate_shapes(work, (8, 16, 32), 'ladder')
    assert 50 in shapes
    cost, groups = plan_gpu(work, (1, 1_000_000), shapes)
    assert groups[0].shape == 50
    assert cost == brute_gpu(work, (1, 1_000_000), shapes)


def test_csv_1024_modifiers_are_one():
    assert inference_modifier(1024) == 1.0
    assert compilation_modifier(1024) == 1.0


def test_csv_lookup_matches_reference_rows():
    assert inference_us(1_000_000, 256) == round(1_000_000 * 0.12933754)
    assert inference_us(1_000_000, 2048) == round(1_000_000 * 4.52811282)
    assert compilation_us(1_000_000, 256) == round(1_000_000 * 0.99578898)
    assert compilation_us(1_000_000, 2048) == round(1_000_000 * 1.00479588)
    assert inference_us(1_000_000, 9) == inference_us(1_000_000, 10)
    assert compilation_modifier(6000) == compilation_modifier(CSV_MAX)
    assert compilation_modifier(CSV_MAX) == 1.40297111


def test_free_mode_uses_csv_buckets_not_ceil_8():
    work = [_w(1, 1, 'a'), _w(9, 1, 'b'), _w(16, 1, 'c'), _w(17, 1, 'd')]
    shapes = candidate_shapes(work, (128, 256), 'free')
    assert shapes == [s for s in CSV_BUCKETS if 5 <= s <= 19]
    assert 17 not in shapes
    assert 24 not in shapes
    assert 19 in shapes


def test_candidates_at_or_below_csv_max_are_csv_only():
    work = [_w(17, 1, 'a'), _w(50, 1, 'b'), _w(6000, 1, 'c')]
    free = candidate_shapes(work, (128, 256), 'free')
    ladder = candidate_shapes(work, (8, 16, 32, 128), 'ladder')
    for shapes in (free, ladder):
        assert all(s in CSV_BUCKETS or s > CSV_MAX for s in shapes)
        assert any(s > CSV_MAX for s in shapes)
    assert 17 not in free and 17 not in ladder
    assert 19 in free
    assert 50 in ladder
    assert 6000 in free
    assert 6000 in ladder


def test_plan_gpu_compile_follows_csv_modifiers():
    gpu = (1_000_000, 1_000_000)
    c128, _ = plan_gpu([_w(128, 1, 'a')], gpu, [128, 256])
    c256, _ = plan_gpu([_w(256, 1, 'a')], gpu, [128, 256])
    assert compilation_us(1_000_000, 128) != compilation_us(1_000_000, 256)
    assert c128 == compilation_us(1_000_000, 128) + inference_us(1_000_000, 128)
    assert c256 == compilation_us(1_000_000, 256) + inference_us(1_000_000, 256)


def test_empty_work():
    assert plan_gpu([], (1, 1_000_000), [128]) == (0, [])
    ms, fl, fid, g = plan_multi_gpu([], [(1, 1_000_000), (1, 1_000_000)], [128])
    assert ms == 0 and g == [[], []]


def test_single_gpu_multi():
    work = [_w(128, 1, 'a'), _w(256, 1, 'b')]
    ms, _, _, groups = plan_multi_gpu(work, [(4, 1_000_000)], [128, 256])
    c, g = plan_gpu(work, (4, 1_000_000), [128, 256])
    assert ms == c and groups == [g]


def test_more_gpus_than_jobs():
    work = [_w(128, 1, 'a')]
    ms, _, _, groups = plan_multi_gpu(
        work, [(1, 1_000_000), (1, 1_000_000), (1, 1_000_000)], [128])
    filled = sum(1 for g in groups if g)
    assert filled == 1
    assert ms < INF


def test_all_one_bucket():
    work = [_w(128, 1, n) for n in 'abc']
    gpus = [(10, 1_000_000), (10, 1_000_000)]
    ms, _, _, groups = plan_multi_gpu(work, gpus, [128, 256])
    assert ms == brute_multi(work, gpus, [128, 256])
    assert sum(len(x) for x in groups) >= 1


def test_nonpositive_throughput():
    cost, groups = plan_gpu([_w(128)], (1, 0), [128])
    assert cost == INF and groups == []


def test_makespan_is_max():
    work = [_w(128, 1, 'a'), _w(128, 1, 'b')]
    gpus = [(0, 1_000_000), (0, 1_000_000)]
    ms, _, _, groups = plan_multi_gpu(work, gpus, [128])
    assert ms == brute_multi(work, gpus, [128])
    assert ms == inference_us(1_000_000, 128)


def test_heterogeneous_fast_gpu_gets_more():
    work = [_w(128, 1, n) for n in 'abcd']
    gpus = [(0, 4_000_000), (0, 1_000_000)]  # gpu1 is 4x faster at 1024
    ms, _, _, groups = plan_multi_gpu(work, gpus, [128])
    n0 = sum(len(g.job_ids) for g in groups[0])
    n1 = sum(len(g.job_ids) for g in groups[1])
    assert n1 >= n0


def test_heterogeneous_65_percent_faster_gets_more_than_half():
    work = [_w(128, 1, f'j{i}') for i in range(100)]
    gpus = [(0, 100_000_000), (0, 165_000_000)]  # gpu0 is 65% faster
    _, _, _, groups = plan_multi_gpu(work, gpus, [128])
    n0 = sum(len(g.job_ids) for g in groups[0])
    n1 = sum(len(g.job_ids) for g in groups[1])
    assert n0 + n1 == 100
    assert n0 >= 55


def test_contiguous_shape_ranges_mostly_disjoint():
    work = [_w(128 * (i + 1), 1, f'j{i}') for i in range(20)]
    shapes = [128 * (i + 1) for i in range(20)]
    _, _, _, groups = plan_multi_gpu(work, [(0, 1_000_000), (0, 1_000_000)], shapes)
    s0 = {g.shape for g in groups[0]}
    s1 = {g.shape for g in groups[1]}
    assert s0 and s1
    assert len(s0 & s1) <= 1


def test_brute_small_random():
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


def test_coarsened_agrees_near_threshold():
    work = [_w(128, 1, f'j{i}') for i in range(6)]
    gpus = [(2, 1_000_000), (2, 1_000_000)]
    shapes = [128, 256]
    a = plan_multi_gpu(work, gpus, shapes, exact_split_threshold=1000)[0]
    b = plan_multi_gpu(work, gpus, shapes, exact_split_threshold=3)[0]
    assert a == b


def test_determinism_shuffle():
    rng = random.Random(1)
    base = [_w(t, 1, f'j{t}-{i}') for i, t in enumerate([128, 128, 256, 256, 512])]
    gpus = [(5, 2_000_000), (7, 1_000_000)]
    shapes = [128, 256, 512]
    results = []
    for _ in range(20):
        w = list(base)
        rng.shuffle(w)
        w.sort(key=lambda x: (x[0], x[2]))
        results.append(plan_multi_gpu(w, gpus, shapes))
    assert len(set(r[0] for r in results)) == 1
    ids = [tuple(tuple(g.job_ids for g in gpu) for gpu in r[3]) for r in results]
    assert len(set(ids)) == 1


def test_round_robin_groups_by_shape():
    work = [_w(128, 1, 'a'), _w(256, 1, 'b'), _w(128, 1, 'c'), _w(256, 1, 'd')]
    groups = plan_round_robin(work, 2, [128, 256])
    for gpu_groups in groups:
        shapes = [g.shape for g in gpu_groups]
        assert shapes == sorted(shapes)


def test_cubic_only_above_csv_max():
    assert inference_us(1_000_000, 2048) == round(1_000_000 * 4.52811282)
    assert inference_us(1_000_000, 6000) == max(
        1, int(round(1_000_000 * shape_factor(6000))))
    assert inference_us(1_000_000, 6000) > 2 * inference_us(1_000_000, 1024)
    assert compilation_us(1_000_000, 6000) == compilation_us(1_000_000, CSV_MAX)


def test_coarsened_heterogeneous_near_equal_makespan():
    work = [_w(128, 1, f's{i}') for i in range(400)]
    work += [_w(256, 1, f'l{i}') for i in range(400)]
    gpus = [(0, 750_000), (0, 1_320_000)]
    shapes = [128, 256]
    ms, _, _, groups = plan_multi_gpu(
        work, gpus, shapes, exact_split_threshold=50)
    c0 = groups_cost_us(groups[0], work, gpus[0])
    c1 = groups_cost_us(groups[1], work, gpus[1])
    assert ms == max(c0, c1)
    assert abs(c0 - c1) <= inference_us(1_320_000, 256)
    n0 = sum(len(g.job_ids) for g in groups[0])
    n1 = sum(len(g.job_ids) for g in groups[1])
    assert n0 + n1 == 800


def test_share_tails_omits_when_thief_cannot_beat_original():
    work = [_w(128, 1, 'a'), _w(128, 1, 'b')]
    gpus = [(0, 1_000_000), (0, 1_000_000)]
    shapes = [128]
    assignment = [
        [Group(shape=128, job_ids=('a',))],
        [Group(shape=128, job_ids=('b',))],
    ]
    overlay = share_tails(assignment, work, shapes, gpus)
    steal0 = [jid for g in overlay[0] if g.shared for jid in g.job_ids]
    steal1 = [jid for g in overlay[1] if g.shared for jid in g.job_ids]
    assert steal0 == []
    assert steal1 == []


def test_share_tails_last_job_first_when_thief_finishes_sooner():
    work = [_w(256, 1, 'a'), _w(256, 1, 'b')]
    gpus = [(0, 4_000_000), (0, 1_000_000)]
    shapes = [256]
    assignment = [
        [Group(shape=256, job_ids=('a', 'b'))],
        [],
    ]
    overlay = share_tails(assignment, work, shapes, gpus)
    steal1 = [jid for g in overlay[1] if g.shared for jid in g.job_ids]
    assert steal1[0] == 'b'
    assert [jid for g in overlay[0] if not g.shared for jid in g.job_ids] == [
        'a', 'b']
    assert groups_cost_us(overlay[0], work, gpus[0]) == groups_cost_us(
        assignment[0], work, gpus[0])


def test_share_tails_new_bucket_pays_compile_once():
    work = [_w(256, 1, 'a'), _w(256, 1, 'b')]
    gpus = [(10_000_000, 4_000_000), (10_000_000, 1_000_000)]
    shapes = [256]
    assignment = [
        [Group(shape=256, job_ids=('a', 'b'))],
        [],
    ]
    overlay = share_tails(assignment, work, shapes, gpus)
    steal = [g for g in overlay[1] if g.shared]
    stolen = [jid for g in steal for jid in g.job_ids]
    assert stolen == ['b', 'a']
    inf = inference_us(1_000_000, 256)
    compile_256 = compilation_us(10_000_000, 256)
    orig_a = compilation_us(10_000_000, 256) + inference_us(4_000_000, 256)
    charged_once = compile_256 + 2 * inf
    charged_twice = 2 * compile_256 + 2 * inf
    assert charged_once < orig_a <= charged_twice


def test_share_tails_omits_new_bucket_when_compile_too_slow():
    work = [_w(256, 1, 'a')]
    gpus = [(0, 1_000_000), (10**12, 1_000_000)]
    shapes = [256]
    assignment = [
        [Group(shape=256, job_ids=('a',))],
        [],
    ]
    overlay = share_tails(assignment, work, shapes, gpus)
    steal1 = [jid for g in overlay[1] if g.shared for jid in g.job_ids]
    assert steal1 == []
