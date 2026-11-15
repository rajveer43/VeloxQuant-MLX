#!/usr/bin/env bash
# Print the CHANGELOG section for one version, for use as GitHub Release notes.
#
# Usage: scripts/extract_release_notes.sh <version> [changelog-path]
#   <version>        without the leading "v", e.g. 0.95.3
#   [changelog-path] defaults to CHANGELOG.md
#
# semantic-release's auto-generated heading is "## v<version> (<date>)" (no
# brackets) -- distinct from this file's older hand-written
# "## [<version>] — <date>" entries. Match either so a future switch back to
# hand-editing doesn't silently break release notes.
set -euo pipefail

version="${1:?usage: extract_release_notes.sh <version> [changelog-path]}"
changelog="${2:-CHANGELOG.md}"

awk -v v1="## v${version} " -v v2="## [${version}]" '
  (index($0, v1) == 1 || index($0, v2) == 1) { found=1; print; next }
  found && index($0, "## ") == 1 { exit }
  found { print }
' "${changelog}"
