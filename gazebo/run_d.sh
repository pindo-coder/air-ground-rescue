#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${bundle_dir}${GZ_SIM_SYSTEM_PLUGIN_PATH:+:${GZ_SIM_SYSTEM_PLUGIN_PATH}}"
if [[ -S /mnt/wslg/.X11-unix/X0 ]]; then
    export DISPLAY="${DISPLAY:-:0}"
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
    d3d12_core=/usr/lib/wsl/lib/libd3d12core.so
    if [[ -e /dev/dxg && -r "$d3d12_core" && "${LIBGL_ALWAYS_SOFTWARE:-0}" != 1 ]]; then
        # Keep the WSL graphics runtime loaded while Ogre releases its sensors.
        export LD_PRELOAD="$d3d12_core${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
fi

world="$bundle_dir/search_d.sdf"
if [[ $# -ne 0 || ! -S /mnt/wslg/.X11-unix/X0 ]]; then
    exec gz sim -r "$world" "$@"
fi

# A second D instance can exhaust WSLg's graphics allocations. Hold the lock
# until both processes have exited, including after a renderer crash.
exec 9>"$bundle_dir/.launch.lock"
if ! flock -n 9; then
    exit 0
fi

children=()
cleanup() {
    if (( ${#children[@]} == 0 )); then
        return
    fi
    kill -TERM "${children[@]}" 2>/dev/null || true
    for _ in {1..60}; do
        if [[ -z "$(jobs -pr)" ]]; then
            break
        fi
        sleep 0.1
    done
    kill -KILL "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

gz sim -s -r "$world" &
children+=("$!")
gz sim -g "$world" &
children+=("$!")
wait -n "${children[@]}"
