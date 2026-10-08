"""Bucket-aware planner against the production DP, brute force, its own seeds and a lower bound."""

from __future__ import annotations

import itertools
import random

import pytest

from fullFold.config import DEFAULT_BUCKETS
from fullFold.planner import (
    EXACT_MAX_JOBS, lower_bound_us, plan_multi_gpu_search, plan_single_gpu,
)
from fullFold.scheduling import (
    FAST, INF, STANDARD, candidate_shapes, groups_cost_us, job_floor_us,
    kernel_tile_shapes, plan_gpu,
)

GPU_KINDS = [
    (80_000_000, 59_423_000), (65_000_000, 32_683_000), (95_000_000, 89_135_000),
    (20_000_000, 90_000_000), (200_000_000, 30_000_000), (1_000_000, 1_000_000),
    (0, 1_000_000),
]


def _work(rng: random.Random, n: int, big: bool = False):
    hi = 7000 if big else 1400
    toks = sorted(int(5 + rng.random() ** 2 * hi) for _ in range(n))
    return [(t, rng.choice((1, 1, 5, 25)), f'j{i:04d}') for i, t in enumerate(toks)]


def _setup(rng: random.Random, n: int, big: bool = False):
    work = _work(rng, n, big)
    kind = rng.choice(('free', 'ladder', 'fast'))
    if kind == 'fast':
        return work, kernel_tile_shapes(work), FAST
    return work, candidate_shapes(work, DEFAULT_BUCKETS, kind, STANDARD), STANDARD


def _check(groups, work, gpus, ref) -> int:
    """Every job once, shape >= tokens, ascending unique shapes; returns the makespan."""
    by_id = {w[2]: w for w in work}
    seen: list[str] = []
    loads = []
    assert len(groups) == len(gpus)
    for grs, gpu in zip(groups, gpus):
        shapes = [g.shape for g in grs]
        assert shapes == sorted(set(shapes))
        for g in grs:
            assert not g.shared
            for jid in g.job_ids:
                assert by_id[jid][0] <= g.shape
                seen.append(jid)
        loads.append(groups_cost_us(grs, work, gpu, ref))
    assert sorted(seen) == sorted(by_id)
    return max(loads)


def test_single_gpu_matches_production_dp():
    rng = random.Random(1)
    for trial in range(80):
        work, shapes, ref = _setup(rng, rng.randint(1, 150), big=trial % 5 == 0)
        gpu = rng.choice(GPU_KINDS)
        want, _ = plan_gpu(work, gpu, shapes, ref)
        cost, groups = plan_single_gpu(work, gpu, shapes, ref)
        assert cost == want, trial
        assert _check([groups], work, [gpu], ref) == want


def test_single_gpu_edge_cases():
    assert plan_single_gpu([], (1, 1), [128]) == (0, [])
    assert plan_single_gpu([(100, 1, 'a')], (5, 0), [128])[0] == INF
    cost, groups = plan_single_gpu([(100, 3, 'a')], (7, 1_000_000), [128, 256])
    assert [g.shape for g in groups] == [128]
    assert cost == plan_gpu([(100, 3, 'a')], (7, 1_000_000), [128, 256])[0]


def test_unoccupied_shapes_can_win_when_compile_is_not_monotone():
    # Table bucket 76 compiles 9% faster than 72 at the same inference cost: a job of 72
    # tokens should be compiled at 76 on a compile-dominated GPU, and the DP must find it.
    gpu = (80_000_000, 1_000)
    work = [(72, 1, 'a')]
    cost, groups = plan_single_gpu(work, gpu, [72, 76], STANDARD)
    assert cost == plan_gpu(work, gpu, [72, 76], STANDARD)[0]
    assert groups[0].shape == 76


def _brute(work, gpus, shapes, ref) -> int:
    best = INF
    for assign in itertools.product(range(len(gpus)), repeat=len(work)):
        loads = []
        for g, gpu in enumerate(gpus):
            sub = [w for w, a in zip(work, assign) if a == g]
            loads.append(plan_gpu(sub, gpu, shapes, ref)[0] if sub else 0)
        best = min(best, max(loads))
    return best


def test_exact_stage_equals_brute_force_and_bound_is_valid():
    rng = random.Random(2)
    for trial in range(25):
        n = rng.randint(1, 6)
        gpus = [rng.choice(GPU_KINDS[:6]) for _ in range(rng.randint(1, 3))]
        work, shapes, ref = _setup(rng, n)
        info: dict = {}
        makespan, floor, floor_id, groups = plan_multi_gpu_search(
            work, gpus, shapes, ref, info=info)
        assert _check(groups, work, gpus, ref) == makespan
        assert makespan == _brute(work, gpus, shapes, ref), trial
        assert info['stage'] in ('exact', 'single')
        lb = lower_bound_us(work, gpus, shapes, ref)
        assert lb <= makespan
        assert info['lower_bound_us'] == lb


def test_job_bound_stays_valid_when_a_larger_shape_is_cheaper():
    # One 72-token job: at its own shape 72 it costs more than at table bucket 76, so the
    # production floor (own shape) exceeds the optimum while the planner's bound does not.
    gpus = [(80_000_000, 1_000), (80_000_000, 1_000)]
    work = [(72, 1, 'a')]
    makespan, floor, _, _ = plan_multi_gpu_search(work, gpus, [72, 76], STANDARD)
    assert (floor, 'a') == job_floor_us(work, gpus, [72, 76], STANDARD)
    assert makespan < floor
    assert lower_bound_us(work, gpus, [72, 76], STANDARD) <= makespan


def test_floor_matches_production():
    rng = random.Random(3)
    for trial in range(40):
        work, shapes, ref = _setup(rng, rng.randint(1, 120), big=trial % 4 == 0)
        gpus = [rng.choice(GPU_KINDS[:6]) for _ in range(rng.randint(1, 5))]
        makespan, floor, floor_id, _ = plan_multi_gpu_search(
            work, gpus, shapes, ref, effort=0)
        assert (floor, floor_id) == job_floor_us(work, gpus, shapes, ref), trial
        assert floor <= makespan


def test_search_plans_valid_deterministic_and_not_worse_than_its_seeds():
    # the seeds are the optimal contiguous partitions and the speed-proportional split
    rng = random.Random(4)
    for trial in range(8):
        n = rng.randint(13, 24)
        gpus = [rng.choice(GPU_KINDS[:6]) for _ in range(rng.randint(2, 3))]
        work, shapes, ref = _setup(rng, n)
        info: dict = {}
        makespan, _, _, groups = plan_multi_gpu_search(work, gpus, shapes, ref, info=info)
        assert _check(groups, work, gpus, ref) == makespan
        assert info['stage'] == 'search' and makespan <= info['seed_us'], trial
        again = plan_multi_gpu_search(work, gpus, shapes, ref)
        assert again[3] == groups and again[0] == makespan


def test_search_beats_its_contiguous_seeds_where_interleaving_matters():
    rng = random.Random(2)
    toks = sorted(int(40 + rng.random() ** 2 * 1400) for _ in range(24))
    work = [(t, rng.choice((1, 1, 5)), f'j{i:03d}') for i, t in enumerate(toks)]
    gpus = [(80_000_000, 59_000_000), (65_000_000, 33_000_000),
            (80_000_000, 59_000_000), (95_000_000, 89_000_000)]
    shapes = candidate_shapes(work, DEFAULT_BUCKETS, 'free', STANDARD)
    info: dict = {}
    makespan, _, _, groups = plan_multi_gpu_search(work, gpus, shapes, info=info)
    assert _check(groups, work, gpus, STANDARD) == makespan
    assert makespan < 0.97 * info['seed_us']


@pytest.mark.parametrize('effort', [0, 1, 2])
def test_efforts_give_valid_plans_with_a_lower_bound(effort):
    rng = random.Random(5)
    work, shapes, ref = _setup(rng, 300)
    gpus = [GPU_KINDS[0], GPU_KINDS[1], GPU_KINDS[2], GPU_KINDS[0]]
    info: dict = {}
    makespan, _, _, groups = plan_multi_gpu_search(
        work, gpus, shapes, ref, effort=effort, info=info)
    assert _check(groups, work, gpus, ref) == makespan
    assert info['lower_bound_us'] <= makespan <= 1.2 * info['lower_bound_us']


def test_large_instance_and_oversized_jobs():
    rng = random.Random(6)
    work, shapes, ref = _setup(rng, 20_000, big=True)
    gpus = [GPU_KINDS[0], GPU_KINDS[1], GPU_KINDS[2], GPU_KINDS[3], GPU_KINDS[0], GPU_KINDS[1]]
    makespan, _, _, groups = plan_multi_gpu_search(work, gpus, shapes, ref, effort=0)
    assert _check(groups, work, gpus, ref) == makespan


def test_degenerate_inputs_and_effort_validation():
    assert plan_multi_gpu_search([], [(1, 1)], [128]) == (0, 0, '', [[]])
    assert plan_multi_gpu_search([(10, 1, 'a')], [], [128])[0] == INF
    with pytest.raises(ValueError):
        plan_multi_gpu_search([(10, 1, 'a')], [(1, 1)], [128], effort=3)
    # a GPU with an unusable probe gets no jobs; the others are planned as usual
    work = [(100, 1, 'a'), (200, 1, 'b')]
    ms, _, _, groups = plan_multi_gpu_search(work, [(1, 1_000_000), (1, 0)], [128, 256])
    assert groups[1] == [] and ms == plan_gpu(work, (1, 1_000_000), [128, 256])[0]
    ms, _, _, groups = plan_multi_gpu_search(work, [(1, 0), (1, 1_000_000), (1, 1_000_000)], [128, 256])
    assert groups[0] == [] and ms < INF and all(g for g in groups[1:])
    assert plan_multi_gpu_search(work, [(1, 0), (1, 0)], [128, 256])[0] == INF


def test_exact_stage_threshold_is_documented_size():
    assert EXACT_MAX_JOBS == 12
    rng = random.Random(7)
    work, shapes, ref = _setup(rng, EXACT_MAX_JOBS)
    info: dict = {}
    plan_multi_gpu_search(work, [GPU_KINDS[0], GPU_KINDS[1]], shapes, ref, info=info)
    assert info['stage'] == 'exact'
    work, shapes, ref = _setup(rng, EXACT_MAX_JOBS + 1)
    plan_multi_gpu_search(work, [GPU_KINDS[0], GPU_KINDS[1]], shapes, ref, info=info)
    assert info['stage'] == 'search'
