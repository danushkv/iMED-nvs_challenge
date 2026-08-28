# Data contract and leakage boundary

## Expected sequence layout

The implemented methods use the challenge sequence layout:

```text
<data-root>/<sequence>/
├── K.txt
├── pose.txt
└── endoscope2/
    ├── L/frame_*.png
    ├── depthL/frame_*.npy
    └── toolL/frame_*.png       # required only by methods that use source masks
```

Offline evaluation additionally requires organizer-provided Endoscope1 target
RGB and masks, but inference code must not enumerate or open them.

## Shared camera and depth conventions

- Camera ID 0 is the Endoscope2 source camera.
- Camera ID 1 is the Endoscope1 target camera.
- Pose rows are `camera_id tx ty tz qx qy qz qw`.
- Poses are camera-to-world.
- Depth is optical-axis Z depth in millimetres.
- Source RGB is 1280×1024 in the inspected data; depth is 640×512.
- Intrinsics must be scaled when operating on the lower-resolution grid.
- Raw `toolL` value 255 is tool/excluded and 0 is tissue/included.

Method 4's independent geometry check matched the validated Method 1
backprojection to approximately `1e-14` world-coordinate L2 error and zero
pixel round-trip error.

## Legal inference inputs

| Input | M1/MV1A | M2 | M3 | M4 |
| --- | :---: | :---: | :---: | :---: |
| Endoscope2/L RGB | Yes | Yes | Yes | Yes |
| Endoscope2/L depth | Yes | Yes | Yes | Yes |
| Endoscope2/L tool mask | No in MV1A | Yes | No | Yes |
| Source/target calibration | Yes | Yes | Yes | Yes |
| Endoscope1 RGB | **No** | **No** | **No** | **No** |
| Endoscope1 tool mask | No | No | No | No |

Endoscope1 RGB and target masks may be opened only by a separate offline
evaluator after predictions are frozen. They must not influence training,
initialization, calibration, transform selection, appearance correction,
hyperparameter selection, checkpoint selection, or method selection.

## Output contract

Challenge-style inference writes:

```text
/output/<sequence>/renders/00000.png
/output/<sequence>/renders/00001.png
...
```

Files are five-digit, zero-based, sequential RGB PNGs at native target
resolution. Nothing writes to the dataset or `/input`.
