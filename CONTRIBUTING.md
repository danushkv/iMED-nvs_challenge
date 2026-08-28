# Contributing experiments

This repository is an experiment record first and a software package second.
Changes should preserve comparability and the challenge's source-only inference
boundary.

## Adding an experiment

1. Give the variant a unique name and name its frozen parent.
2. Change one conceptual component at a time where possible.
3. Add an immutable config under the owning method.
4. Run the method's geometry/source gate before target rendering.
5. Use the predeclared sequence set and evaluator protocol.
6. Save per-frame metrics and an aggregate result.
7. Record runtime, peak VRAM, seed, environment, and source revisions.
8. Mark the decision `keep`, `drop`, or `inconclusive`.

## Required experiment-card fields

```text
name:
parent:
research question:
controlled difference:
legal inputs:
forbidden inputs:
configuration:
environment:
sequence set:
metric protocol:
aggregate result:
per-frame wins/losses:
runtime and memory:
qualitative advantage:
qualitative failure:
decision:
```

## Repository hygiene

- Do not commit datasets, target GT, checkpoints, environments, CUDA toolkits,
  compiled extensions, downloaded installers, or complete render streams.
- Keep a few qualitative predictions only after confirming redistribution
  rights, with sequence/frame/protocol provenance.
- Do not overwrite a frozen result directory.
- Keep rejected ablations documented.
- Never combine hidden and local metrics into one ranking.
- Never use Endoscope1 RGB in inference, optimization, initialization,
  calibration, selection, or appearance correction.

Run the static audit after assembling or changing the source snapshot:

```bash
bash scripts/validate_release.sh
```
