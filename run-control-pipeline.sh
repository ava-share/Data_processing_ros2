#!/usr/bin/env bash
# Run control data extraction with local ROS 2 Jazzy.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f /home/atlab/ros2_jazzy/install/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /home/atlab/ros2_jazzy/install/setup.bash
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/jazzy/setup.bash
else
  echo "ERROR: ROS 2 Jazzy setup.bash not found" >&2
  exit 1
fi

exec python3 control-data-pipeline.py "$@"
