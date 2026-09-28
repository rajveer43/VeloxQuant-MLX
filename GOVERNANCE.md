# Governance

## Maintainer

VeloxQuant-MLX is currently maintained by
[rajveer43](https://github.com/rajveer43). The maintainer makes final decisions
on architecture, releases, and merges.

## Decision Making

For routine contributions (bug fixes, documentation, benchmark additions) the
normal PR review process applies. Significant changes — new quantization
methods, API changes, dependencies — should be discussed in a GitHub issue
before a PR is opened. This avoids wasted effort and aligns expectations early.

The maintainer has final say. For contested decisions the maintainer will
explain the reasoning in the relevant issue or PR thread.

## Becoming a Contributor

Open PRs, review others' PRs, help triage issues, or improve documentation.
There is no formal application process. Consistent, high-quality contributions
are the only criterion. For how a merged contribution is credited — a
[CONTRIBUTORS.md](CONTRIBUTORS.md) listing, a paper acknowledgement, or
co-authorship — see [Authorship & Acknowledgement
Policy](#authorship--acknowledgement-policy) below.

## Becoming a Co-Maintainer

Co-maintainers may be added if the project grows to a scale where a single
maintainer is a bottleneck. This will be announced in the repository. There is
no formal path to apply — the maintainer will reach out directly.

## Releases

Releases are tagged on `master` by the maintainer. The changelog is maintained
in [CHANGELOG.md](CHANGELOG.md). There is no fixed release cadence; releases
are driven by meaningful feature or fix milestones.

## Authorship & Acknowledgement Policy

The MIT license already covers code use and redistribution — it says nothing
about who gets named where, so this section exists to state that separately
and in the open, rather than leaving it as an informal, ask-privately norm.

Every merged PR is credited via normal git/GitHub history and the
[CHANGELOG.md](CHANGELOG.md), regardless of size. The distinctions below are
about the *additional* forms of credit — a `CONTRIBUTORS.md` listing, a
paper acknowledgements section, or paper co-authorship — that a contribution
can also earn, roughly in increasing order of what it took to earn them:

- **Contributor listing** (`CONTRIBUTORS.md`): any merged, non-trivial PR —
  a bug fix, a new benchmark, a documentation improvement, a new
  quantization method. This is the default outcome for participating at all.
- **Paper/writeup acknowledgement**: a contribution that shaped the project's
  direction without being a from-scratch research contribution itself —
  a nontrivial validated bug report that changed a method's correctness
  claims, a benchmark run on hardware the maintainer doesn't own that
  became load-bearing evidence for a documented result, sustained review
  or triage work over multiple releases.
- **Co-authorship on a paper or formal writeup**: a contribution with real
  research or design content — proposing and implementing a new compression
  method (not just porting an existing one), a nontrivial theoretical or
  algorithmic correction to an existing method's adaptation, or benchmark
  infrastructure work that materially changes what the project's results
  claims can be trusted to mean (e.g. the kind of cross-hardware validation
  described in #394).

**How this actually gets decided**: there is no formal point system. The
maintainer proposes the level that fits a given contribution, in the open, in
the relevant issue or PR thread, before or shortly after merge — not
retroactively when a paper is being drafted. If a contributor disagrees with
the proposed level, that's a conversation to have then, on that thread, not
a surprise later. This is deliberately informal because the project is
solo-maintained and the volume of contributions doesn't yet justify a formal
rubric — if that changes, this section will be revised to say so.

This policy applies going forward; it is not a claim about how any past
contribution was or wasn't credited.
