# Getting started

## Dataset

1. Register at the [iMED Challenge website](https://imed-challenge.github.io/).
2. Join/request access through the [Synapse project](https://www.synapse.org/Synapse%3Asyn74277461/wiki/640044).
3. Accept the organizer's data-use conditions.
4. Keep the downloaded dataset outside this repository.

For GeoSCOPE, point `/input` either at a single sequence or a root containing
multiple challenge-style sequences.

## One-sequence GeoSCOPE test

Build the image first, then use the provided staging helper. It copies only
legal source-side inputs into a temporary test tree:

```bash
cd methods/geoscope/code/docker_submission/method8_f1
bash scripts/stage_one_sequence.sh /path/to/iMED_NVS session_004_scene_2_tool_1
```

Copy the printed `TEST_ROOT=...` value and run:

```bash
TEST_ROOT=/tmp/method8-f1-nvs-test.REPLACE_ME
bash scripts/local_test.sh geoscope:latest "$TEST_ROOT"
```

The test disables networking, mounts input read-only, enforces the challenge
timeout, and validates the output filename/resolution contract. Clean only the
printed temporary root with the supplied cleanup script after inspection.

## Training-free development

To inspect geometry without waiting for Endo-4DGS optimization:

```bash
uv sync
source .venv/bin/activate
cd methods/method1_rgbd_reprojection/code
python identity_test.py --data_root /path/to/iMED_NVS --sequence session_004_scene_2_tool_1 --frame_id 2 --output_json outputs/identity.json
```

Then follow the M1 or M3 README for a one-frame render.

## Evaluation

Evaluation is a separate offline phase. Freeze predictions first, then run the
released evaluator with Endoscope1 RGB available only to that evaluator. The
local protocols and their differences are documented in
[`EVALUATION.md`](EVALUATION.md).
