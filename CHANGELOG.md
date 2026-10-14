# Changelog

## 0.1.3

* **Featurisation workers**: each GPU worker keeps up to `2N+2` upcoming (job, seed) items featurising in `N` CPU-only processes
  (`--feat-workers N`, default `2`, `0` = featurise in the worker as before). The first two items are checked bitwise against
  in-process featurisation; a mismatch, a failing child or an unknown item falls back to in-process featurisation.
  Log: `_af3sched/featpool.jsonl`.
* **Fast featurisation** for AlphaFold 3 v3.0.4 (`--no-fast-featurisation` / `FULLFOLD_FAST_FEAT=0` to disable): an exact,
  vectorised `extract_msa_features` and threaded RDKit reference conformers with replayed seeds. Gated on the sha256 of the
  installed AF3 functions; any other source (e.g. a modified or newer fork) is left untouched and says so.
* Fix: AF3 v3.0.4's `write_output` has no `keep_license` argument; it is now dropped when unsupported (jobs failed after inference).
* Fix: hoist fingerprints are valid for Python 3.12-3.14 `ast.dump` output as well, so the hoists are enabled on those versions.
* Fix: look-ahead bookkeeping when jobs are skipped (claimed by another worker).
