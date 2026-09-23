"""Startup banner for the fullFold CLI. Printed to stderr."""

from __future__ import annotations

import sys

BANNER = """\
  __       _  _ _____     _     _
 / _|_  _ | || |  ___|___| | __| |
| |_| || || || |_|   \\___/|_|\\__,_|

AlphaFold 3 at full throughput!
Please cite AlphaFold3 and the upcoming fullFold paper (updates in our repo).
"""


def print_banner(file=None) -> None:
    print(BANNER, file=file or sys.stderr, end='')
