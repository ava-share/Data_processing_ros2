#!/usr/bin/env bash
# Run perception / trajectory extraction with local ROS 2 + yolo_msgs.
set -eo pipefail  # no -u: ROS setup.bash may reference unset vars
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

if [[ -f /home/atlab/yolo_ws/install/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /home/atlab/yolo_ws/install/setup.bash
fi

exec python3 data-pipeline.py "$@"
