"""argparse -> Config -> one function. No scheduling logic here."""

from __future__ import annotations

import argparse
from pathlib import Path

from fullFold.banner import print_banner
from fullFold.config import DEFAULT_BUCKETS, Config, config_from_dict


def _buckets(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(',')) if s else DEFAULT_BUCKETS


def _add_io(p: argparse.ArgumentParser) -> None:
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)


def _add_sched(p: argparse.ArgumentParser) -> None:
    p.add_argument('--model-dir', type=Path, default=Config.model_dir)
    p.add_argument('--gpus', default='')
    p.add_argument('--no-prefetch', dest='prefetch', action='store_const',
                   const=0, default=1)
    p.add_argument('--no-background-extract', dest='background_extract',
                   action='store_false')
    p.add_argument('--policy', choices=('contiguous', 'roundrobin'), default='contiguous')
    p.add_argument('--bucket-mode', choices=('ladder', 'free'), default='free')
    p.add_argument('--buckets', type=_buckets, default=DEFAULT_BUCKETS)
    p.add_argument('--bucket-margin', type=float, default=0.05)
    p.add_argument('--stale-lock-seconds', type=int, default=900)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--exact-split-threshold', type=int, default=512)
    p.add_argument('--force-benchmark', action='store_true')
    p.add_argument('--bench-seed', type=int, default=42)
    p.add_argument('--cache-dir', type=Path, default=Config.cache_dir)
    p.add_argument('--exact-tokens', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--jax-compilation-cache-dir', type=Path, default=None)
    p.add_argument('--xla-mem-fraction', type=float, default=0.97)
    p.add_argument('--save-embeddings', action='store_true')
    p.add_argument('--save-distogram', action='store_true')
    p.add_argument('--compress-large-output-files', action='store_true')
    p.add_argument('--num-recycles', type=int, default=10)
    p.add_argument('--num-diffusion-samples', type=int, default=5)
    p.add_argument('--flash-attention', default='triton')


def _cfg(ns: argparse.Namespace) -> Config:
    kw = {n: getattr(ns, n) for n in Config.__dataclass_fields__ if hasattr(ns, n)}
    g = kw.get('gpus', ())
    if isinstance(g, str):
        kw['gpus'] = tuple(x for x in g.split(',') if x)
    return config_from_dict(kw)


def main(argv: list[str] | None = None) -> int:
    print_banner()
    parser = argparse.ArgumentParser(prog='fullFold')
    sub = parser.add_subparsers(dest='cmd', required=True)

    for name in ('scan', 'plan', 'run'):
        p = sub.add_parser(name)
        _add_io(p)
        _add_sched(p)
        if name == 'run':
            p.add_argument('-v', '--verbose', action='store_true',
                           help='Show per-GPU progress bars during the run')

    p = sub.add_parser('benchmark')
    p.add_argument('--output-dir', type=Path, default=Path('.'))
    p.add_argument('--input-dir', type=Path, default=Path('.'))
    _add_sched(p)

    p = sub.add_parser('template')
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--type', dest='record_type', default='protein')

    args = parser.parse_args(argv)
    cfg = _cfg(args)
    if args.cmd == 'scan':
        from fullFold.engine import cmd_scan
        return cmd_scan(cfg)
    if args.cmd == 'benchmark':
        from fullFold.engine import cmd_benchmark
        return cmd_benchmark(cfg)
    if args.cmd == 'plan':
        from fullFold.engine import cmd_plan
        return cmd_plan(cfg)
    if args.cmd == 'run':
        from fullFold.engine import cmd_run
        return cmd_run(cfg)
    if args.cmd == 'template':
        from fullFold.templates import cmd_template
        return cmd_template(cfg)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
