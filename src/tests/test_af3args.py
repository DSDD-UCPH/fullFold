"""CLI aliases, fast-mode default, and model choices."""

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


def test_hyphen_flags_and_default_mode_stay_off(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False, faster=True)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out),
    ])
    assert rc == 0
    cfg = seen['cfg']
    assert cfg.mode == 'off'
    assert cfg.reference == 'standard'
    assert cfg.model == 'alphafold3'
    assert cfg.buckets_explicit is False


def test_underscore_aliases_and_absl_false(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input_dir', str(inp), '--output_dir', str(out),
        '--num_recycles=3', '--save_embeddings=false',
        '--flash_attention_implementation=xla',
    ])
    assert rc == 0
    cfg = seen['cfg']
    assert cfg.input_dir == inp
    assert cfg.output_dir == out
    assert cfg.num_recycles == 3
    assert cfg.save_embeddings is False
    assert cfg.flash_attention == 'xla'


def test_json_path_alias(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False)
    src = tmp_path / 'in.json'
    src.write_text('{}')
    out = tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--json_path', str(src), '--output_dir', str(out),
    ])
    assert rc == 0
    assert seen['cfg'].json_path == src


def test_data_pipeline_warns_once(monkeypatch, tmp_path: Path, capsys):
    _patch_tree(monkeypatch, colabfold=False)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, _ = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out),
        '--db_dir', '/data', '--run_data_pipeline=true',
    ])
    assert rc == 0
    err = capsys.readouterr().err
    assert err.count('not running genetic') == 1


def test_data_pipeline_false_is_silent(monkeypatch, tmp_path: Path, capsys):
    _patch_tree(monkeypatch, colabfold=False)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, _ = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out),
        '--run_data_pipeline=false',
    ])
    assert rc == 0
    assert 'not running genetic' not in capsys.readouterr().err


def test_run_inference_false_fails(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out),
        '--run_inference=false',
    ])
    assert rc == 2
    assert 'cfg' not in seen


def test_nojit_rejected_on_colabfold(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=True)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out), '--nojit',
    ])
    assert rc == 2
    assert 'cfg' not in seen


def test_mode_fast_requires_package(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False, faster=False)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out), '--mode', 'fast',
    ])
    assert rc == 2
    assert 'cfg' not in seen


def test_mode_fast_switches_reference_unless_set(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False, faster=True)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    base = ['run', '--input-dir', str(inp), '--output-dir', str(out), '--mode', 'fast']
    rc, seen = _run(monkeypatch, base)
    assert rc == 0
    assert seen['cfg'].mode == 'fast'
    assert seen['cfg'].reference == 'fast'
    rc, seen = _run(monkeypatch, base + ['--reference', 'standard'])
    assert rc == 0
    assert seen['cfg'].reference == 'standard'
    assert seen['cfg'].mode == 'fast'


def test_explicit_buckets_warn_when_not_tile(monkeypatch, tmp_path: Path, capsys):
    _patch_tree(monkeypatch, colabfold=False, faster=True)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out),
        '--mode', 'fast', '--buckets', '100,200',
    ])
    assert rc == 0
    assert seen['cfg'].buckets_explicit is True
    assert seen['cfg'].buckets == (100, 200)
    assert 'not multiples of 64' in capsys.readouterr().err


def test_model_help_hides_openfold_on_orig(monkeypatch, capsys):
    _patch_tree(monkeypatch, colabfold=False)
    from fullFold.cli import main
    with pytest.raises(SystemExit) as exc:
        main(['run', '--help'])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert 'openfold3' not in out
    assert 'openbind0' not in out
    assert 'alphafold3' in out


def test_model_help_shows_openfold_on_colabfold(monkeypatch, capsys):
    _patch_tree(monkeypatch, colabfold=True)
    from fullFold.cli import main
    with pytest.raises(SystemExit) as exc:
        main(['run', '--help'])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert 'openfold3' in out
    assert 'openbind0' in out


def test_openfold_is_invalid_on_orig(monkeypatch, tmp_path: Path, capsys):
    _patch_tree(monkeypatch, colabfold=False)
    from fullFold.cli import main
    inp, out = tmp_path / 'in', tmp_path / 'out'
    with pytest.raises(SystemExit):
        main([
            'run', '--input-dir', str(inp), '--output-dir', str(out),
            '--model', 'openfold3',
        ])
    err = capsys.readouterr().err
    assert 'invalid choice' in err
    assert 'openbind0' not in err


def test_of3_weights_selects_openfold_on_colabfold(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=True)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out), '--of3_weights',
    ])
    assert rc == 0
    assert seen['cfg'].model == 'openfold3'


def test_of3_checkpoint_rejected_on_colabfold(monkeypatch, tmp_path: Path, capsys):
    _patch_tree(monkeypatch, colabfold=True)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out),
        '--of3_checkpoint', 'weights.pt',
    ])
    assert rc == 2
    assert 'cfg' not in seen
    assert 'convert weights offline' in capsys.readouterr().err


def test_of3_weights_unknown_on_orig(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False)
    from fullFold.cli import main
    inp, out = tmp_path / 'in', tmp_path / 'out'
    with pytest.raises(SystemExit):
        main([
            'run', '--input-dir', str(inp), '--output-dir', str(out),
            '--of3_weights',
        ])


def test_gpu_device_fills_gpus(monkeypatch, tmp_path: Path):
    _patch_tree(monkeypatch, colabfold=False)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out), '--gpu_device=2',
    ])
    assert rc == 0
    assert seen['cfg'].gpus == ('2',)


def test_fast_worker_env_sets_levers(monkeypatch, tmp_path: Path):
    monkeypatch.delenv('XLA_FLAGS', raising=False)
    monkeypatch.delenv('AF3_JAX_SAMPLER_BF16', raising=False)
    monkeypatch.delenv('AF3_JAX_HOIST_LOGITS', raising=False)
    monkeypatch.delenv('AF3_FLASHPAIRFORMER', raising=False)
    fast = worker_environ(
        Config(mode='fast', sampler_bf16=False, cache_dir=tmp_path),
        '0', device_kind='A100')
    assert fast['XLA_FLAGS'] == XLA_TRITON_GEMM_OFF
    assert fast['AF3_JAX_SAMPLER_BF16'] == '0'
    assert fast['AF3_JAX_HOIST_LOGITS'] == '1'
    assert fast['AF3_FLASHPAIRFORMER'] == 'both'
    off = worker_environ(
        Config(mode='off', cache_dir=tmp_path), '0', device_kind='A100')
    assert 'AF3_FLASHPAIRFORMER' not in off
    monkeypatch.setenv('XLA_FLAGS', '--xla_dump_to=/tmp')
    kept = worker_environ(
        Config(mode='fast', cache_dir=tmp_path), '0', device_kind='A100')
    assert kept['XLA_FLAGS'] == '--xla_dump_to=/tmp'


def test_cache_dir_underscore_is_not_the_benchmark_cache(monkeypatch, tmp_path, capsys):
    _patch_tree(monkeypatch, colabfold=True)
    inp, out = tmp_path / 'in', tmp_path / 'out'
    rc, seen = _run(monkeypatch, [
        'run', '--input-dir', str(inp), '--output-dir', str(out),
        '--cache_dir', '/tokamax',
    ])
    assert rc == 0
    assert 'tokamax' not in str(seen['cfg'].cache_dir)
    assert 'ignoring --cache_dir' in capsys.readouterr().err
