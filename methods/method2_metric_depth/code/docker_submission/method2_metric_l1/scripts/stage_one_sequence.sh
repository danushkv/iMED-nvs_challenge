#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 DATASET_ROOT SEQUENCE_NAME [TMP_PARENT]" >&2
    exit 2
fi

DATASET_ROOT="$1"
SEQUENCE_NAME="$2"
TMP_PARENT="${3:-/tmp}"
SOURCE_SEQUENCE="$DATASET_ROOT/$SEQUENCE_NAME"

if [[ "$SEQUENCE_NAME" == */* || "$SEQUENCE_NAME" == "." || "$SEQUENCE_NAME" == ".." ]]; then
    echo "Sequence name must be one path component: $SEQUENCE_NAME" >&2
    exit 2
fi

for required in K.txt pose.txt endoscope2/L endoscope2/depthL endoscope2/toolL; do
    if [[ ! -e "$SOURCE_SEQUENCE/$required" ]]; then
        echo "Missing required source input: $SOURCE_SEQUENCE/$required" >&2
        exit 1
    fi
done

TEST_ROOT="$(mktemp -d "$TMP_PARENT/method2-nvs-test.XXXXXX")"
trap 'status=$?; if [[ $status -ne 0 ]]; then echo "Staging failed; retained: $TEST_ROOT" >&2; fi' EXIT

INPUT_SEQUENCE="$TEST_ROOT/input/$SEQUENCE_NAME"
mkdir -p "$INPUT_SEQUENCE/endoscope2" "$TEST_ROOT/output"
printf '%s\n' "method2-nvs-test-v1" > "$TEST_ROOT/.method2-nvs-test-marker"

cp "$SOURCE_SEQUENCE/K.txt" "$INPUT_SEQUENCE/K.txt"
cp "$SOURCE_SEQUENCE/pose.txt" "$INPUT_SEQUENCE/pose.txt"
cp -a "$SOURCE_SEQUENCE/endoscope2/L" "$INPUT_SEQUENCE/endoscope2/L"
cp -a "$SOURCE_SEQUENCE/endoscope2/depthL" "$INPUT_SEQUENCE/endoscope2/depthL"
cp -a "$SOURCE_SEQUENCE/endoscope2/toolL" "$INPUT_SEQUENCE/endoscope2/toolL"
if [[ -f "$SOURCE_SEQUENCE/target_frames.txt" ]]; then
    cp "$SOURCE_SEQUENCE/target_frames.txt" "$INPUT_SEQUENCE/target_frames.txt"
fi

echo "Staged only permitted source inputs; no Endoscope-1 data was copied."
printf 'TEST_ROOT=%q\n' "$TEST_ROOT"
printf 'INPUT_ROOT=%q\n' "$TEST_ROOT/input"
printf 'OUTPUT_ROOT=%q\n' "$TEST_ROOT/output"
echo "Temporary data is retained until cleanup_test.sh is called explicitly."

trap - EXIT
