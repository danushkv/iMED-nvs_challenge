#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 IMAGE" >&2
    exit 2
fi

docker image inspect "$1" --format $'image_id={{.Id}}\nworking_dir={{.Config.WorkingDir}}\nentrypoint={{json .Config.Entrypoint}}\ncmd={{json .Config.Cmd}}\nrenderer={{range .Config.Env}}{{println .}}{{end}}\nlabels={{json .Config.Labels}}'
