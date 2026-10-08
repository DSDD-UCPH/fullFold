#!/usr/bin/env python3
"""Regenerate fingerprints.json from the installed alphafold3.

Refuses unless the CPU equivalence test passes, unless --hashes-only.
Target is google-deepmind/alphafold3 v3.0.4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fullFold.hoists.fingerprint import (  # noqa: E402
    TARGET_VERSION, alphafold3_root, collect_hashes, fingerprints_path,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--hashes-only', action='store_true')
    p.add_argument('--root', type=Path, default=None)
    args = p.parse_args(argv)
    root = args.root or alphafold3_root()
    if root is None:
        print('fullFold: no alphafold3 package found', file=sys.stderr)
        return 2
    hashes = collect_hashes(root)
    if not args.hashes_only:
        import pytest
        rc = pytest.main([
            '-q', str(ROOT / 'src/tests/test_hoist_equivalence.py'),
        ])
        if rc != 0:
            print(
                'fullFold: equivalence test failed; fingerprints.json not updated',
                file=sys.stderr,
            )
            return rc if isinstance(rc, int) else 1
    payload = {
        'target_version': TARGET_VERSION,
        'hashes': {k: [v] for k, v in hashes.items()},
    }
    dest = fingerprints_path()
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'wrote {dest} ({len(hashes)} functions, target {TARGET_VERSION})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
