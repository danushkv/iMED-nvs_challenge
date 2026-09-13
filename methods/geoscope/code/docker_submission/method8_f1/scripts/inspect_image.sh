#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 IMAGE" >&2
    exit 2
fi

docker image inspect "$1" --format $'image_id={{.Id}}\nsize={{.Size}}\ncreated={{.Created}}\nos_arch={{.Os}}/{{.Architecture}}\nworking_dir={{.Config.WorkingDir}}\nentrypoint={{json .Config.Entrypoint}}\ncmd={{json .Config.Cmd}}\nenv={{json .Config.Env}}\nlabels={{json .Config.Labels}}'

docker run --rm --network=none --entrypoint /bin/sh "$1" -c '
    set -eu
    test "$METHOD3_RENDERER" = surface
    test "$TORCH_CUDA_ARCH_LIST" = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"
    test -s /app/METHOD8_SOURCE_MANIFEST.sha256
    sha256sum -c /app/METHOD8_SOURCE_MANIFEST.sha256
    printf "renderer=%s\narchitectures=%s\n" "$METHOD3_RENDERER" "$TORCH_CUDA_ARCH_LIST"
'
