"""CLI flags: aliases, mode selection, rejected flags, model choices."""

from __future__ import annotations

from pathlib import Path

import pytest

from fullFold.config import XLA_TRITON_GEMM_OFF, worker_environ, Config


def _patch_tree(monkeypatch, *, colabfold: bool, faster: bool = False):
    monkeypatch.setattr(
        'fullFold.af3args.colabfold_installed', lambda: colabfold)
    monkeypatch.setattr(
        'fullFold.af3args.af3_faster_installed', lambda: faster)


def _run(monkeypatch, argv):
    seen = {}

    def fake_run(cfg):
        seen['cfg'] = cfg
        return 0

    monkeypatch.setattr('fullFold.engine.cmd_run', fake_run)
    from fullFold.cli import main
    rc = main(argv)
    return rc, seen


def _argv(tmp_path: Path, *extra: str, cmd: str = 'run') -> list[str]:
    return [cmd, '--input-dir', str(tmp_path / 'in'), '--output-dir', str(tmp_path / 'out'),
            *extra]


def test_mode_defaults_and_fast_mode(monkeypatch, tmp_path: Path, capsys):
    # omitted: fast when af3-faster is installed, with a notice, else default
    _patch_tree(monkeypatch, colabfold=False, faster=True)
    rc, seen = _run(monkeypatch, _argv(tmp_path))
    cfg = seen['cfg']
    assert rc == 0 and (cfg.mode, cfg.reference, cfg.model) == ('fast', 'fast', 'alphafold3')
    assert cfg.buckets_explicit is False
    assert 'using --mode fast' in capsys.readouterr().err
    # an explicit --mode always wins
    _, seen = _run(monkeypatch, _argv(tmp_path, '--mode', 'default'))
    assert (seen['cfg'].mode, seen['cfg'].reference) == ('default', 'standard')
    # fast schedules on the fast table, unless --reference says otherwise
    _, seen = _run(monkeypatch, _argv(tmp_path, '--mode', 'fast'))
    assert (seen['cfg'].mode, seen['cfg'].reference) == ('fast', 'fast')
    _, seen = _run(monkeypatch, _argv(tmp_path, '--mode', 'fast', '--reference', 'standard'))
    assert (seen['cfg'].mode, seen['cfg'].reference) == ('fast', 'standard')
    # explicit buckets that are not multiples of 64 draw a warning
    capsys.readouterr()
    rc, seen = _run(monkeypatch, _argv(tmp_path, '--mode', 'fast', '--buckets', '100,200'))
    assert rc == 0 and seen['cfg'].buckets_explicit is True
    assert seen['cfg'].buckets == (100, 200)
    assert 'not multiples of 64' in capsys.readouterr().err
    # without af3-faster: default, and --mode fast is refused
    _patch_tree(monkeypatch, colabfold=False, faster=False)
    _, seen = _run(monkeypatch, _argv(tmp_path))
    assert (seen['cfg'].mode, seen['cfg'].reference) == ('default', 'standard')
    rc, seen = _run(monkeypatch, _argv(tmp_path, '--mode', 'fast'))
    assert rc == 2 and 'cfg' not in seen
    # a missing package, or a missing parent of a dotted one, is absence, not an error
    from fullFold import af3args
    assert af3args._module_exists('no_such_pkg') is False
    assert af3args._module_exists('no_such_pkg.sub') is False


def test_flag_spellings_and_scheduling_flags(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False)
    # run_alphafold.py spellings: underscores, --flag=false, --json_path, --gpu_device
    rc, seen = _run(monkeypatch, [
        'run', '--input_dir', str(tmp_path / 'in'), '--output_dir', str(tmp_path / 'out'),
        '--num_recycles=3', '--save_embeddings=false', '--flash_attention_implementation=xla',
        '--gpu_device=2'])
    cfg = seen['cfg']
    assert rc == 0 and (cfg.input_dir, cfg.output_dir) == (tmp_path / 'in', tmp_path / 'out')
    assert (cfg.num_recycles, cfg.save_embeddings, cfg.flash_attention) == (3, False, 'xla')
    assert cfg.gpus == ('2',)
    src = tmp_path / 'in.json'
    src.write_text('{}')
    _, seen = _run(monkeypatch, ['run', '--json_path', str(src), '--output_dir', str(tmp_path)])
    assert seen['cfg'].json_path == src
    # defaults, then the switches that turn overlap and planning choices around
    _, seen = _run(monkeypatch, _argv(tmp_path))
    cfg = seen['cfg']
    assert (cfg.prefetch, cfg.background_extract, cfg.verbose) == (1, True, False)
    assert (cfg.bucket_mode, cfg.reference) == ('free', 'standard')
    _, seen = _run(monkeypatch, _argv(
        tmp_path, '--no-prefetch', '--no-background-extract', '--bucket-mode', 'ladder',
        '--reference', 'fast', '-v'))
    cfg = seen['cfg']
    assert (cfg.prefetch, cfg.background_extract, cfg.verbose) == (0, False, True)
    assert (cfg.bucket_mode, cfg.reference) == ('ladder', 'fast')
    assert _run(monkeypatch, _argv(tmp_path, '--verbose'))[1]['cfg'].verbose is True


def test_data_pipeline_flags_are_accepted_with_one_warning(monkeypatch, tmp_path: Path, capsys):
    _patch_tree(monkeypatch, colabfold=False)
    rc, _ = _run(monkeypatch, _argv(tmp_path, '--db_dir', '/data', '--run_data_pipeline=true'))
    assert rc == 0 and capsys.readouterr().err.count('not running genetic') == 1
    _run(monkeypatch, _argv(tmp_path, '--run_data_pipeline=false'))
    assert 'not running genetic' not in capsys.readouterr().err
    # ColabFold's own --cache_dir is not our benchmark cache
    _patch_tree(monkeypatch, colabfold=True)
    rc, seen = _run(monkeypatch, _argv(tmp_path, '--cache_dir', '/tokamax'))
    assert rc == 0 and 'tokamax' not in str(seen['cfg'].cache_dir)
    assert 'ignoring --cache_dir' in capsys.readouterr().err


def test_flags_that_would_change_the_model_are_rejected(monkeypatch, tmp_path: Path, capsys):
    _patch_tree(monkeypatch, colabfold=False)
    rc, seen = _run(monkeypatch, _argv(tmp_path, '--run_inference=false'))
    assert rc == 2 and 'cfg' not in seen
    _patch_tree(monkeypatch, colabfold=True)
    rc, seen = _run(monkeypatch, _argv(tmp_path, '--nojit'))
    assert rc == 2 and 'cfg' not in seen
    capsys.readouterr()
    rc, seen = _run(monkeypatch, _argv(tmp_path, '--of3_checkpoint', 'weights.pt'))
    assert rc == 2 and 'cfg' not in seen
    assert 'convert weights offline' in capsys.readouterr().err


def test_model_choices_follow_the_installed_alphafold3(monkeypatch, tmp_path: Path, capsys):
    from fullFold.cli import main
    _patch_tree(monkeypatch, colabfold=False)
    with pytest.raises(SystemExit) as exc:
        main(['run', '--help'])
    out = capsys.readouterr().out
    assert exc.value.code == 0 and 'alphafold3' in out
    assert 'openfold3' not in out and 'openbind0' not in out
    with pytest.raises(SystemExit):
        main(_argv(tmp_path, '--model', 'openfold3'))
    err = capsys.readouterr().err
    assert 'invalid choice' in err and 'openbind0' not in err
    with pytest.raises(SystemExit):
        main(_argv(tmp_path, '--of3_weights'))  # only a ColabFold install knows it
    _patch_tree(monkeypatch, colabfold=True)
    with pytest.raises(SystemExit) as exc:
        main(['run', '--help'])
    out = capsys.readouterr().out
    assert exc.value.code == 0 and 'openfold3' in out and 'openbind0' in out
    rc, seen = _run(monkeypatch, _argv(tmp_path, '--of3_weights'))
    assert rc == 0 and seen['cfg'].model == 'openfold3'


def test_fast_worker_env_sets_levers(monkeypatch, tmp_path: Path):
    for var in ('XLA_FLAGS', 'AF3_JAX_SAMPLER_BF16', 'AF3_JAX_HOIST_LOGITS',
                'AF3_FLASHPAIRFORMER'):
        monkeypatch.delenv(var, raising=False)
    fast = worker_environ(
        Config(mode='fast', sampler_bf16=False, cache_dir=tmp_path), '0', device_kind='A100')
    assert fast['XLA_FLAGS'] == XLA_TRITON_GEMM_OFF
    assert fast['AF3_JAX_SAMPLER_BF16'] == '0' and fast['AF3_JAX_HOIST_LOGITS'] == '1'
    assert fast['AF3_FLASHPAIRFORMER'] == 'both'
    for mode in ('off', 'default'):
        env = worker_environ(Config(mode=mode, cache_dir=tmp_path), '0', device_kind='A100')
        assert 'AF3_FLASHPAIRFORMER' not in env, mode
    # the user's flags are kept and the one fast mode needs is added to them
    monkeypatch.setenv('XLA_FLAGS', '--xla_dump_to=/tmp')
    merged = worker_environ(
        Config(mode='fast', cache_dir=tmp_path), '0', device_kind='A100')
    assert merged['XLA_FLAGS'] == f'--xla_dump_to=/tmp {XLA_TRITON_GEMM_OFF}'
    # an explicit choice for that flag is respected
    monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=true --xla_dump_to=/tmp')
    chosen = worker_environ(
        Config(mode='fast', cache_dir=tmp_path), '0', device_kind='A100')
    assert chosen['XLA_FLAGS'] == '--xla_gpu_enable_triton_gemm=true --xla_dump_to=/tmp'


def test_no_hoist_logits_is_not_recorded_as_active(monkeypatch, tmp_path: Path):
    """The hoist set is part of the probe-cache key, so it must match what runs."""
    from fullFold.hoists.fingerprint import HoistStatus
    _patch_tree(monkeypatch, colabfold=True, faster=True)
    names = ('cond_share', 'atom_cond_hoist', 'diffusion_hoist', 'hoist_logits')
    monkeypatch.setattr(
        'fullFold.hoists.evaluate_hoists',
        lambda **k: {n: HoistStatus(n, False, True, 'native') for n in names})
    base = _argv(tmp_path, '--mode', 'default')
    _, on = _run(monkeypatch, base)
    _, off = _run(monkeypatch, [*base, '--no-hoist-logits'])
    assert 'hoist_logits' in on['cfg'].hoists
    assert 'hoist_logits' not in off['cfg'].hoists and 'cond_share' in off['cfg'].hoists
