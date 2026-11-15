# Reproducibility procedure

## Environment

Export the actual experimental environment with `python --version` and `python -m pip freeze > requirements.txt`. Record separate environments if experiments differ. Document hardware, simulator, operating system and Slurm settings.

## Full experiment execution

For every experiment, record the exact tested command, working directory, seed, configuration, dataset path, output directory and expected metrics. Retain the original frozen code and its internal directory structure. Do not replace manuscript-generating implementations with newer variants.

## Numerical regeneration

Archive per-seed round/client/security metrics and final summary CSVs. Update notebook paths to point to the archive without modifying analysis semantics. Run every notebook from a fresh checkout and compare regenerated numerical values with the submitted paper. Record discrepancies, missing seeds and any diagnostic experiments that use fewer seeds. Record the distinction between modeled communication and measured software overhead, and overlapping V2X events.

## Release checks

- Verify all final experiments and seed coverage.
- Remove credentials, confidential peer-review materials and unauthorized datasets.
- Verify all notebook dependencies and paths.
- Confirm author and institutional permission and software licence.
- Tag the verified manuscript-associated commit.

Exact execution commands and archive paths are pending inspection of the experimental files.
