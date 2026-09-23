"""Startup banner for the fullFold CLI. Printed to stderr."""

from __future__ import annotations

import sys

BANNER = """\
  __       _ _ _____     _     _ 
 / _|_   _| | |  ___|__ | | __| |
| |_| | | | | | |_ / _ \\| |/ _` |
|  _| |_| | | |  _| (_) | | (_| |
|_|  \\__,_|_|_|_|  \\___/|_|\\__,_|

fullFold: unlocking the full speed of AF3!
Please cite AlphaFold3 and fullFold (see our repository).
"""


def print_banner(file=None) -> None:
    print(BANNER, file=file or sys.stderr, end='')
