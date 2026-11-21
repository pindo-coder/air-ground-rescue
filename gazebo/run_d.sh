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
exec gz sim -r "${bundle_dir}/search_d.sdf" "$@"
