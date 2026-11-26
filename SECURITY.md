# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest (main branch) | yes |
| older releases | no |

VeloxQuant-MLX is a pre-1.0 library under active development. Only the current
`master` branch receives security fixes.

## Scope

This policy covers security issues in VeloxQuant-MLX's own code, including:

- **Control-panel HTTP server.** `veloxquant panel` (`veloxquant_mlx/ui/server.py`)
  always binds to `127.0.0.1`; it has no `--host` option, and its API can start
  processes and change saved configuration. It checks `Host` and `Origin`
  headers, limits request bodies, and confines static files to its asset
  directory. Bypasses of those controls, including cross-site requests and DNS
  rebinding, are in scope.
- **Local artifact and file loading.** Loading of precomputed `.npy`/`.npz`
  artifacts, Q-Filters calibrations, reservoir files, and worker temp files by
  VeloxQuant code (for example `artifacts/npy_store.py`,
  `quantizers/qfilters_calibration.py`, `weight/reservoir.py`, and
  `cli/worker.py`) is in scope. This includes unsafe handling of
  attacker-controlled paths or file contents: code execution, path traversal,
  crashes, and resource exhaustion. Pickle-based NumPy loads are deliberately
  avoided, but accepting a hostile file remains a security boundary.

The following are out of scope:

- **Publicly exposed inference servers.** `veloxquant serve` binds to
  `127.0.0.1` by default, but `--host 0.0.0.0` is a supported operator choice
  and is separate from the panel's fixed loopback bind. Issues that arise only
  because the inference server was deliberately exposed to a network, or that
  live in upstream `mlx_lm.server`, should be reported upstream. Defects in
  VeloxQuant's own integration code remain in scope.
- **Model files and `mlx_lm`.** Model weights are loaded by the third-party
  `mlx_lm` package. Vulnerabilities in model files or weight formats, or in
  `mlx_lm` itself, are upstream issues; report them to that project unless the
  flaw is in VeloxQuant's own integration code.

If you are unsure whether a finding is covered, report it privately and we will
triage it.

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Email **rathodrajveer1311@gmail.com** with:

- a description of the vulnerability and its potential impact,
- steps to reproduce or a minimal proof-of-concept,
- affected versions or commits, and
- any suggested mitigations if you have them.

You will receive an acknowledgement within 72 hours. We aim to resolve confirmed
vulnerabilities within 14 days and will credit reporters in the release notes
unless anonymity is requested.
