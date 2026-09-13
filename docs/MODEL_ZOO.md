# Model availability

This release does not distribute pretrained checkpoints.

## Important checkpoint semantics

GeoSCOPE does not have a universal pretrained checkpoint. M3-B is
training-free, and the Method-2 fallback is optimized separately for each
input sequence using permitted Endoscope2 inputs. A saved checkpoint would
therefore reproduce only its corresponding sequence and configuration; it is
not a general model that can be applied to a new sequence.

If sequence-specific reproduction states are published in the future, their
recommended structure is:

```text
<sequence>/
├── checkpoint/                  # Method-2 model state
├── config.json                  # 300/1000, metric_l1, 5e-5
├── source_manifest.json         # frame ids and source-data fingerprint
├── environment.json             # image digest, CUDA, PyTorch, GPU
├── metrics.json                 # clearly labeled local protocol
└── README.md                    # legal inputs and exact render command
```

Do not upload challenge data, target RGB, target masks, private hidden data,
or organizer-owned weights without redistribution permission.
