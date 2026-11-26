# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest PyPI release (`VeloxQuant-MLX`) | yes |
| `master` branch | yes |
| Older PyPI releases | no |

VeloxQuant-MLX is a pre-1.0 library under active development. Security fixes land
on the default `master` branch and are shipped to users through a PyPI hotfix
release — a manual run of the release workflow rather than the weekly release.
See [Releases and the hotfix process](CONTRIBUTING.md#releases) for how a fix is
published. Older PyPI releases are not patched; upgrade to the latest release to
receive fixes. Each hotfix produces a GitHub Release and a `CHANGELOG.md` entry,
which is how users are told about the fix; reporters are credited in the release
notes unless anonymity is requested.

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
