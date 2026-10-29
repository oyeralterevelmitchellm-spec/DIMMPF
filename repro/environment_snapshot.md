# Environment snapshot

- Generated: 2026-10-29 13:34:53 CST (Asia/Shanghai)
- Python: 3.13.9 (D:/anaconda/python.exe)
- Git HEAD: 7cc1f57322a7c4085a9ae46c54759f40c923391a
- git describe: 7cc1f57-dirty
- Branch: codex/research-tlb
- Scope: third-party imports under repro/src and repro/exp (stdlib and local modules excluded)
- Note: this pin task does **not** create a git commit

## Dependency table

| Import name | Pip package | Version |
|-------------|-------------|---------|
| numpy | numpy | 2.4.6 |
| scipy | scipy | 1.17.1 |
| pandas | pandas | 2.3.3 |
| matplotlib | matplotlib | 3.11.0 |
| sklearn | scikit-learn | 1.7.2 |
| torch | torch | 2.12.0.dev20260408+cu128 |
| markdown | Markdown | 3.8 |

## Notes

- torch is a CUDA 12.8 nightly/dev build (local version tag +cu128); reinstall may need the matching PyTorch index URL.
- Source of truth for install pins: requirements.txt in this directory.
