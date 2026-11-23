## Summary

Brief description of what this PR changes and why.

## Related issue

Closes #

## Test plan

**Code changes**

- [ ] `python -m pytest veloxquant_mlx/tests -q` passes (locally on Apple Silicon if you have it; if not, say so here and rely on CI)
- [ ] New or updated unit tests cover the change
- [ ] Docs updated if user-facing behavior changed

**Docs-only changes** (skip the code checks above)

- [ ] No code changed, and the links and anchors I touched resolve

## Checklist

- [ ] The PR title follows [Conventional Commits](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/CONTRIBUTING.md#commit-conventions) (`fix:`, `feat:`, `docs:`, ...)
- [ ] I have the right to contribute this under the project's MIT license
- [ ] I agree to follow the [Code of Conduct](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/CODE_OF_CONDUCT.md)
- [ ] I read and understand every line of this PR, and disclosed any substantial AI assistance in the summary (see [CONTRIBUTING](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/CONTRIBUTING.md#ai-assisted-contributions))

## Number claims (if any)

Compression, speedup, or memory claims **must** link a reproducible script and a committed `results.json`.

- Script path:
- Results path:
- Metric type (check one):
  - [ ] Key-byte **accounting** (`fp16_key_bytes / compressed_key_bytes`)
  - [ ] Full-KV accounting (keys + values + residual windows)
  - [ ] MLX peak memory (`mx.get_peak_memory`)
  - [ ] OS RSS / long-context OOM comparison
  - [ ] Throughput (tok/s)

Do not describe accounting ratios as resident RAM savings unless resident memory was measured.
