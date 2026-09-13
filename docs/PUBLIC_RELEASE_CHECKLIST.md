# Public-release checklist

Complete these items before changing repository visibility:

- [x] Confirm the NCT-TSO Team software citation and release metadata.
- [x] Select an explicit license for original GeoSCOPE/M1/M3 code (MIT).
- [ ] Confirm compatibility with the Endo-4DGS/Gaussian-Splatting research license.
- [x] Confirm the organizer permits redistribution of the two prediction-only GIFs.
- [x] Generate and visually inspect both GIFs; confirm they contain no target ground truth.
- [x] Generate and validate `uv.lock` for deterministic root-environment resolution.
- [x] Document that no pretrained checkpoints are distributed.
- [ ] Pin mutable Docker parent tags by digest where possible.
- [ ] Run the GeoSCOPE static checks, CUDA smoke test, one-sequence output validator, and pixel-parity test.
- [x] Run `bash scripts/validate_release.sh`.
- [ ] Review `git diff`, especially for absolute cluster paths, usernames, secrets, datasets, and large files.
- [ ] Confirm no checkpoints, environments, compiled extensions, or Docker archives are staged.

The root MIT license covers original release code only. Third-party snapshots
and adapted components retain their upstream licenses; review
[`../THIRD_PARTY.md`](../THIRD_PARTY.md) before distribution.
