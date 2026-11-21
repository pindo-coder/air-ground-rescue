#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_file="${1:-${script_dir}/bridge_config.yaml}"

if [[ -z "${ROS_DISTRO:-}" ]]; then
    if [[ -f /opt/ros/jazzy/setup.bash ]]; then
        # shellcheck disable=SC1091
        source /opt/ros/jazzy/setup.bash
    else
        echo "[ros2_rescue] ROS 2 Jazzy not sourced and /opt/ros/jazzy/setup.bash not found." >&2
        exit 1
    fi
fi

exec ros2 run ros_gz_bridge parameter_bridge \
    --ros-args -p config_file:="${config_file}" -p use_sim_time:=true
