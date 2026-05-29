## AVA ROS 2 Data Pipelines

Python pipelines for extracting perception and control data from ROS 2 Jazzy bags.

### Scripts

- `data-pipeline.py`: perception detections, object tracks, odometry, plots, camera frames, MP4 clips, and key metrics.
- `control-data-pipeline.py`: lateral-control performance, command data, odometry, plots, and control metrics.
- `ros2_bag_utils.py`: small `rosbag2_py` wrapper used by both scripts.

### Setup

Source ROS 2 Jazzy (and `yolo_ws` for perception bags if needed):

```bash
source /opt/ros/jazzy/setup.bash   # or ~/ros2_jazzy/install/setup.bash
source /path/to/your/ros2_ws/install/setup.bash   # e.g. ~/yolo_ws/install/setup.bash
python3 -m pip install numpy matplotlib opencv-python

# MCAP bags with custom msgs: project-local .deps (rosbags + opencv for perception frames)
python3 -m pip install --target=.deps 'numpy<2' rosbags opencv-python-headless
```

Control bags under `/home/atlab/Downloads/April23Testing` do not require a separate `raptor_dbw_msgs` install when `.deps` includes rosbags.

For custom message types, topic discovery still uses `rosbag2_py`. Deserialization falls back to **rosbags** and the schema embedded in the MCAP file when the installed ROS package is missing or mismatched (`yolo_msgs`, `raptor_dbw_msgs`).

### Bag Input

Edit `BAG_FILE` or `BAG_FOLDER` near the top of each script:

```python
BAG_FILE = "/path/to/one/ros2_bag_directory"
BAG_FOLDER = None
```

or:

```python
BAG_FILE = None
BAG_FOLDER = "/path/to/folder/of/bags"
SEARCH_RECURSIVELY = True
```

Override without editing the script:

```bash
export AVA_BAG_FOLDER="/home/atlab/Downloads/April23Testing"
export AVA_BAG_FILE="/path/to/one/recording.mcap"   # single bag; takes precedence over AVA_BAG_FOLDER
```

Supported inputs are ROS 2 bag directories with `metadata.yaml`, standalone `.mcap` files, and standalone `.db3` files.

The control pipeline is currently configured for the April 23, 2026 ctrl6 bags under `/home/atlab/Downloads/April23Testing` (five recordings: `ctrl6_20260423_131927` through `ctrl6_20260423_150017`).

### Perception Topics

The perception script now matches the provided YOLO bag metadata:

- `/fused_bbox` (`yolo_msgs/msg/DetectionArray`)
- `/novatel/oem7/odom` (`nav_msgs/msg/Odometry`)
- `/yolo/dbg_image` (`sensor_msgs/msg/Image`)

It also still accepts the older `jsk_recognition_msgs/BoundingBoxArray` shape if `TOPIC_FUSED_BBOX` is changed to a bag that uses `msg.boxes`.

### Control Topics

The control script matches the provided `ctrl6_20260423_131927_0.mcap` topic set.

Core controller topics:

- `/novatel/oem7/odom` (`nav_msgs/msg/Odometry`)
- `/lat_ctrl_perf` (`geometry_msgs/msg/Vector3Stamped`)
- `/ctrl_ref_twist` (`geometry_msgs/msg/TwistStamped`)
- `/lat_ctrl_cmd` (`geometry_msgs/msg/Vector3Stamped`)
- `/ctrl_ref_curv` (`geometry_msgs/msg/PointStamped`)
- `/ctrl_ref_pose` (`geometry_msgs/msg/PoseStamped`)
- `/steer_ctrl_cmd` (`geometry_msgs/msg/Vector3Stamped`)
- `/protection_levels` (`geometry_msgs/msg/Vector3Stamped`)

Raptor DBW topics extracted when present:

- `/raptor_dbw_interface/misc_report`
- `/raptor_dbw_interface/steering_report`
- `/raptor_dbw_interface/steering_cmd`
- `/raptor_dbw_interface/accelerator_pedal_cmd`
- `/raptor_dbw_interface/brake_cmd`
- `/raptor_dbw_interface/global_enable_cmd`

### Run

```bash
./run-perception-pipeline.sh   # sources ROS + yolo_ws, then data-pipeline.py
./run-control-pipeline.sh      # sources ROS 2 Jazzy, then control-data-pipeline.py
python3 data-pipeline.py       # if ROS is already sourced
python3 control-data-pipeline.py
```

Set `BAG_FILE` or `BAG_FOLDER` at the top of each script, or use `AVA_BAG_FILE` / `AVA_BAG_FOLDER`. Use `TRAJECTORY_ONLY = True` in the perception script to skip camera/MP4 steps when OpenCV is not installed.

Perception results go to `Extracted_data_{EXTRACTION_DATE}/` and `Intermediate_data_{EXTRACTION_DATE}/`.
Control results go to `Control_data_{EXTRACTION_DATE}/` (currently `Control_data_05292026/` for the April 23 testing batch).

### Notes

- `ros2_bag_utils.py` infers `mcap` or `sqlite3` storage from `metadata.yaml`.
- `cv_bridge` is optional; image extraction falls back to raw `uint8` decoding.
- Set `DEBUG_FIRST_N_OBJECTS` or `DEBUG_FIRST_N_MESSAGES` for quick sanity checks.
