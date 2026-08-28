# Method 4A Docker submission

This directory packages only the frozen M4-A 50k configuration. It does not
contain evaluation code, M4-B selection, target RGB access, or pretrained
sequence checkpoints.

Build context must be the Method 4 root:

```bash
docker build \
  -f docker_submission/Dockerfile.m4a \
  -t method4-gsharp:m4a-dev \
  .
```

The default interface is:

```text
ENTRYPOINT python /workspace/method4_gsharp/docker_submission/entrypoint_m4a.py
CMD run-dataset --data-root /input --output-root /output
```

The image trains each sequence independently using only Endoscope2/L RGB,
depth, source tool masks, and legal calibration. It writes only native-size
RGB PNG predictions to `/output/<sequence>/renders/`.

## Method 4B reproducibility profile

M4-B is a separate, unsuccessful surface-initialization ablation. It is never
selected by the M4-A image and is not the recommended challenge submission.
Build it only under its explicit tag:

```bash
docker build \
  --ignorefile docker_submission/Dockerfile.m4b.dockerignore \
  -f docker_submission/Dockerfile.m4b \
  -t method4-gsharp:m4b-dev \
  .
```

Its immutable defaults are 50,000 initial Gaussians, 0.5 mm voxel fusion,
0.2 surface thickness ratio, a 4x candidate budget, seed 42, and the same
500 coarse plus 3,000 fine training schedule as M4-A.
