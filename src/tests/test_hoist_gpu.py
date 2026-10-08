"""Manual GPU checks for default-mode hoists vs --mode off.

Run on AlphaFold 3 v3.0.4 with real weights:

1. One 1024-token protein and one ligand job, same seeds, ``--mode default``
   vs ``--mode off``. Compare coordinate RMSD, ranking score, pLDDT, and PAE.
2. Memory at bucket 5120 with a ligand-heavy input at the automatic
   ``XLA_CLIENT_MEM_FRACTION`` (total minus 640 MiB), on a large card and a
   card of 24 GB or less. If it OOMs, skip ``atom_cond_hoist`` above an atom
   threshold.
3. Timing probe for default vs off, plus the same on ColabFold.

Not executed in ordinary pytest.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason='manual GPU validation against AF3 v3.0.4')
def test_hoists_match_stock_on_gpu():
    raise AssertionError('run the procedure in this module docstring')
