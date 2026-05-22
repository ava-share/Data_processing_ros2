#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, math, shutil, errno, glob
import fnmatch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt

from ros2_bag_utils import Ros2Bag, stamp_to_sec

# ====== CONFIG =========================================================
EXTRACTION_DATE = "11252025"   # e.g., run date token you want in the root folder name

# BATCH PROCESSING: Set one of these options
# Option 1: Process single ROS 2 bag directory, .db3, or .mcap file
# BAG_FILE = "/media/avresearch/RouteData/perception_output_2025-09-11_15-09-03"

# Option 2: Process all ROS 2 bags in a folder
BAG_FOLDER = "/home/avresearch/Downloads"  # Set to None to use single file
BAG_FILE = None  # Set to None to use folder processing
# Folder search options
SEARCH_RECURSIVELY = True  # Set to True to search all subdirectories, False for top-level only

# Option 3: Process specific rosbags by pattern
# BAG_PATTERN = "*trial*"  # Process only bag paths matching pattern (not currently applied)

# Topics in the detection_yolo bag metadata.yaml
TOPIC_ODOM        = "/novatel/oem7/odom"             # nav_msgs/Odometry
TOPIC_FUSED_BBOX  = "/fused_bbox"                    # yolo_msgs/DetectionArray
TOPIC_CAMERA_IMG  = "/yolo/dbg_image"                # sensor_msgs/Image
TOPIC_METADATA    = None                             # Optional std_msgs/String topic, if a bag has one

# Association / filtering / video params
MAX_SPEED_M_S     = 20.0     # gating speed upper bound
BASE_GATING_M     = 2.0      # extra gating slack (meters)
MIN_TRACK_LIFETIME_S = 1.0   # min duration to keep a track
SMOOTH_WINDOW_K   = 5        # moving-average window (odd recommended)
FRAME_RATE        = 10.0     # output video FPS

# Verbosity and debugging
VERBOSE                = True        # print progress messages during processing
DEBUG_FIRST_N_OBJECTS  = 2           # set to >0 to process only first N objects (by sorted ID)

# Note: Label 9999 (unclassified_radar_detection) objects are automatically excluded from
# tracking and video extraction, but are included in key metrics summary

# Object type mapping: numeric ID -> name
# NOTE: Label 9999 is typically used for unknown/unclassified objects from the detection system
OBJECT_TYPE_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",
    9: "traffic_light", 10: "fire_hydrant", 11: "stop_sign", 12: "parking_meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
    31: "snowboard", 32: "sports_ball", 33: "kite", 34: "baseball_bat", 35: "baseball_glove", 36: "skateboard",
    37: "surfboard", 38: "tennis_racket", 39: "bottle", 40: "wine_glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot_dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted_plant",
    59: "bed", 60: "dining_table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell_phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy_bear", 78: "hair_drier", 79: "toothbrush", 80: "cone",
    81: "speed_limit_70", 82: "speed_limit_75", 83: "speed_limit_30", 84: "speed_limit_35", 85: "speed_limit_40",
    86: "speed_limit_45", 87: "speed_limit_50", 88: "speed_limit_55", 89: "speed_limit_60", 90: "speed_limit_65",
    9999: "unclassified_radar_detection"  # Radar detections without classification (excluded from tracking/video extraction)
}
OBJECT_TYPE_IDS = dict((name, label_id) for label_id, name in OBJECT_TYPE_NAMES.items())

# Static-object handling: For these labels, replace all x,y with their global average
# NOTE: Update this set to match your detector's numeric label IDs for static objects
# Examples (YOU SHOULD CONFIRM IDs): cone, traffic light, stop sign, speed limit sign
STATIC_LABEL_IDS = set([
    #  Example placeholders; replace with your system's static class IDs
    #  e.g., cone=80, fire_hydrant=10, traffic_light=9, stop_sign=11,
    80, 10, 9, 11, 9999
])

# ====== PATHS ==========================================================
def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Initialize output directories (will be set per bag in batch processing)
ROOT_OUT = "Extracted_data_{}".format(EXTRACTION_DATE)
INTERMEDIATE_OUT = "Intermediate_data_{}".format(EXTRACTION_DATE)
ensure_dir(ROOT_OUT)
ensure_dir(INTERMEDIATE_OUT)

# ====== UTILITIES ======================================================
try:
    from cv_bridge import CvBridge
    _BRIDGE = CvBridge()
except Exception as _e:
    _BRIDGE = None

def convert_img_to_cv2(msg):
    """Try cv_bridge, fallback to raw uint8 decode (assumes 8UC3 BGR)."""
    if _BRIDGE is not None:
        try:
            return _BRIDGE.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            print("[WARN] cv_bridge conversion failed: {}".format(e))
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return arr.reshape((msg.height, msg.width, -1))

def yaw_from_quat(qx, qy, qz, qw):
    """Extract yaw (heading) from quaternion in ENU/UTM-ish (z-up)."""
    # yaw from quaternion (Z rotation)
    # ref: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def rot2d(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c*x - s*y, s*x + c*y)

def moving_average(vals, k):
    if k <= 1 or len(vals) == 0:
        return vals[:]
    half = k // 2
    out = []
    for i in range(len(vals)):
        lo = max(0, i - half)
        hi = min(len(vals), i + half + 1)
        out.append(sum(vals[lo:hi]) / float(hi - lo))
    return out

def dist2d(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

def get_nested_attr(obj, path, default=None):
    cur = obj
    for part in path.split('.'):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur

def first_nested_attr(obj, paths, default=None):
    for path in paths:
        value = get_nested_attr(obj, path, None)
        if value is not None:
            return value
    return default

def normalize_label(value, class_name=None):
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            class_name = value
    if class_name:
        key = str(class_name).strip().lower().replace(' ', '_')
        return OBJECT_TYPE_IDS.get(key, -1)
    return -1

def detection_to_csv_row(msg, detection, bag_time):
    """Return common bbox CSV fields from yolo_msgs/Detection or jsk BoundingBox."""
    header = first_nested_attr(detection, ['header'], first_nested_attr(msg, ['header']))
    ts = stamp_to_sec(getattr(header, 'stamp', None), fallback=bag_time)
    frame_id = getattr(header, 'frame_id', '') if header is not None else ''

    position = first_nested_attr(detection, [
        'pose.position',
        'bbox3d.center.position',
        'bbox3d.center',
        'bbox_3d.center.position',
        'bbox_3d.center',
        'bbox.center.position',
        'bbox.center',
        'position',
    ])
    dimensions = first_nested_attr(detection, [
        'dimensions',
        'bbox3d.size',
        'bbox_3d.size',
        'bbox.size',
        'size',
    ])
    if position is None:
        return None

    label_value = first_nested_attr(detection, ['label', 'class_id', 'id'])
    class_name = first_nested_attr(detection, ['class_name', 'class_id_name', 'name'])
    label = normalize_label(label_value, class_name)

    return [
        frame_id,
        ts,
        getattr(position, 'x', 0.0),
        getattr(position, 'y', 0.0),
        getattr(position, 'z', 0.0),
        getattr(dimensions, 'x', 0.0) if dimensions is not None else 0.0,
        getattr(dimensions, 'y', 0.0) if dimensions is not None else 0.0,
        getattr(dimensions, 'z', 0.0) if dimensions is not None else 0.0,
        label,
    ]

def choose_fourcc_h264():
    return cv2.VideoWriter_fourcc(*'avc1')  # try H.264 first

def plot_xy(ts, xs, ys, out_path, title):
    plt.figure()
    plt.plot(xs, ys, '-o', markersize=2)
    plt.xlabel("X"); plt.ylabel("Y"); plt.title(title); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

def plot_yt(ts, ys, out_path, title):
    plt.figure()
    plt.plot(ts, ys, '-o', markersize=2)
    plt.xlabel("Time (s)"); plt.ylabel("Y"); plt.title(title); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

# # ===== (COMMENTED) PCD writer for future bags ==========================
# import struct
# def save_pointcloud2_as_pcd(msg, filename):
#     pc = []
#     step = msg.point_step
#     data = msg.data
#     for i in range(0, len(data), step):
#         x, y, z = struct.unpack_from('fff', data, offset=i)
#         pc.append((x, y, z))
#     with open(filename, 'w') as f:
#         f.write('# .PCD v0.7 - Point Cloud Data file format\n')
#         f.write('VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n')
#         f.write('WIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'.format(len(pc)))
#         f.write('POINTS {}\nDATA ascii\n'.format(len(pc)))
#         for (x,y,z) in pc: f.write('{} {} {}\n'.format(x,y,z))

# ====== STEP 1: READ FUSED BBOX FROM BAG -> CSV ========================
def step1_dump_fused_bbox_csv(bag, bag_basename):
    """
    Read yolo_msgs/DetectionArray or jsk_recognition_msgs/BoundingBoxArray
    from TOPIC_FUSED_BBOX and dump a tidy CSV with timestamp, frame_id,
    position, dims, label.
    Also prints diagnostic information about detected labels.
    """
    # Save to intermediate results folder parallel to ROOT_OUT
    out_csv = os.path.join(INTERMEDIATE_OUT, "{}_fused_bbox_results.csv".format(bag_basename))
    msg_count = 0
    label_counts = {}  # Track label frequencies for diagnostics
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['frame_id','timestamp','x','y','z','dx','dy','dz','label'])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_FUSED_BBOX]):
            msg_count += 1
            detections = getattr(msg, 'detections', None)
            if detections is None:
                detections = getattr(msg, 'boxes', [])
            for detection in detections:
                row = detection_to_csv_row(msg, detection, t)
                if row is None:
                    continue
                lab = row[-1]
                w.writerow(row)
                # Track label counts
                label_counts[lab] = label_counts.get(lab, 0) + 1
            if VERBOSE and (msg_count % 200 == 0):
                print("[INFO] Read {} fused bbox messages...".format(msg_count))
    
    # Print label diagnostics
    if VERBOSE and label_counts:
        print("\n[DIAGNOSTIC] Label distribution in fused bbox data:")
        total_detections = sum(label_counts.values())
        for lab in sorted(label_counts.keys()):
            count = label_counts[lab]
            pct = 100.0 * count / total_detections if total_detections > 0 else 0.0
            label_name = OBJECT_TYPE_NAMES.get(lab, 'unknown_{}'.format(lab))
            print("  Label {} ({}): {} detections ({:.1f}%)".format(lab, label_name, count, pct))
        print("  Total detections: {}\n".format(total_detections))
    
    if not label_counts:
        print("[WARN] No detections extracted from topic: {}".format(TOPIC_FUSED_BBOX))
    print("[OK] Fused bbox CSV -> {}".format(out_csv))
    return out_csv

# ====== STEP 2: READ METADATA FROM BAG =================================
def step2_extract_metadata(bag):
    """
    Read std_msgs/String from TOPIC_METADATA and parse metadata fields.
    Returns dict with parsed metadata or empty dict if not found.
    """
    metadata = {}
    if not TOPIC_METADATA:
        if VERBOSE:
            print("[INFO] No metadata topic configured; continuing with blank metadata fields")
        return metadata
    for topic, msg, t in bag.read_messages(topics=[TOPIC_METADATA]):
        if hasattr(msg, 'data'):
            data_str = msg.data
            # Parse format: "location: RTA-4 Transit, vehicle: blue, passengers: 2, ..."
            try:
                pairs = [pair.strip() for pair in data_str.split(',')]
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        metadata[key] = value
                break  # Take first message
            except Exception as e:
                print("[WARN] Failed to parse metadata: {}".format(e))
                break
    if VERBOSE:
        print("[INFO] Extracted metadata: {}".format(metadata))
    return metadata

# ====== STEP 3: READ ODOM FROM BAG -> CSV =============================
def step3_dump_odom_csv(bag, bag_basename, bag_out_dir):
    """
    Read nav_msgs/Odometry and dump timestamp + (x,y,z) + quaternion.
    """
    out_csv = os.path.join(bag_out_dir, "{}_novatel_odom_data.csv".format(bag_basename))
    msg_count = 0
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(["timestamp","position_x","position_y","position_z",
                    "orientation_x","orientation_y","orientation_z","orientation_w"])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_ODOM]):
            msg_count += 1
            ts = stamp_to_sec(getattr(msg.header, 'stamp', None), fallback=t)
            p = msg.pose.pose.position; q = msg.pose.pose.orientation
            w.writerow([ts, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            if VERBOSE and (msg_count % 500 == 0):
                print("[INFO] Read {} odom messages...".format(msg_count))
    print("[OK] Odom CSV -> {}".format(out_csv))
    return out_csv

# ====== STEP 4: CALCULATE KEY METRICS ===================================
def step4_calculate_key_metrics(odom_csv, metadata, trajs, fused_csv=None):
    """
    Calculate key metrics from odometry and object counts.
    Also counts unclassified radar detections (label 9999) from fused CSV if provided.
    Returns dict with all metrics.
    """
    # Load odometry data
    ego = np.genfromtxt(odom_csv, delimiter=',', names=True)
    ts = ego['timestamp']
    xs = ego['position_x']
    ys = ego['position_y']
    
    # Calculate duration
    duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    
    # Calculate distance (cumulative path length)
    distances = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        distances.append(math.sqrt(dx*dx + dy*dy))
    total_distance = sum(distances)
    
    # Calculate average velocity directly from distance/duration
    avg_velocity = total_distance / duration if duration > 0 else 0.0
    
    # Downsample for velocity calculation - use every Nth point to reduce noise
    downsample_factor = max(1, len(ts) // 100)  # Aim for ~100 velocity samples
    velocities = []
    for i in range(downsample_factor, len(ts), downsample_factor):
        prev_i = i - downsample_factor
        dt = ts[i] - ts[prev_i]
        if dt > 0.1 and dt < 10.0:  # Use 0.1s minimum time interval
            dx = xs[i] - xs[prev_i]
            dy = ys[i] - ys[prev_i]
            vel = math.sqrt(dx*dx + dy*dy) / dt
            velocities.append(vel)
    
    # Find max velocity < 25 m/s
    valid_velocities = [v for v in velocities if v < 25.0]
    max_velocity = max(valid_velocities) if valid_velocities else 0.0
    
    # Calculate accelerations from downsampled velocities
    accelerations = []
    for i in range(1, len(velocities)):
        # Use the time interval between the two velocity measurements
        vel_idx = i * downsample_factor
        prev_vel_idx = (i-1) * downsample_factor
        if vel_idx < len(ts) and prev_vel_idx < len(ts):
            dt = ts[vel_idx] - ts[prev_vel_idx]
            if dt > 0.1 and dt < 5.0:  # Use 0.1s minimum time interval
                accel = (velocities[i] - velocities[i-1]) / dt
                accelerations.append(accel)
    
    # Find max acceleration < 5 m/s²
    valid_accelerations = [a for a in accelerations if a < 5.0]
    max_acceleration = max(valid_accelerations) if valid_accelerations else 0.0
    
    # Find max deceleration > -5 m/s² (most negative but not too extreme)
    valid_decelerations = [a for a in accelerations if a > -5.0]
    max_deceleration = min(valid_decelerations) if valid_decelerations else 0.0  # Most negative
    
    if VERBOSE:
        print("[INFO] Velocity stats: avg={:.2f} m/s (from distance/duration), max={:.2f} m/s (<25 m/s filter)".format(
            avg_velocity, max_velocity))
        print("[INFO] Acceleration stats: max_accel={:.2f} m/s² (<5 m/s²), max_decel={:.2f} m/s² (>-5 m/s²)".format(
            max_acceleration, max_deceleration))
    
    # Count objects by type from trajectories (excludes label 9999)
    object_counts = {}
    for tid, rows in trajs.items():
        if not rows:
            continue
        # Use most common label for this track
        labels = [r['label'] for r in rows]
        try:
            from collections import Counter
            dom_label = Counter(labels).most_common(1)[0][0]
        except Exception:
            dom_label = labels[0] if labels else -1
        
        if dom_label not in object_counts:
            object_counts[dom_label] = 0
        object_counts[dom_label] += 1
    
    # Count unclassified radar detections (label 9999) from fused CSV
    # These are excluded from tracking but should be included in metrics
    if fused_csv is not None:
        try:
            fused = np.genfromtxt(fused_csv, delimiter=',', names=True)
            if 'label' in fused.dtype.names:
                f_lab = fused['label']
                mask_9999 = (f_lab == 9999)
                if np.any(mask_9999):
                    # Count unique unclassified radar detections
                    # Group by timestamp and rounded position (0.5m precision) to count distinct detections
                    # This avoids overcounting the same object detected multiple times in the same frame
                    radar_detections = fused[mask_9999]
                    if len(radar_detections) > 0:
                        # Round positions to 0.5m to group nearby detections
                        rounded_x = np.round(radar_detections['x'] / 0.5) * 0.5
                        rounded_y = np.round(radar_detections['y'] / 0.5) * 0.5
                        # Create unique keys from (timestamp, rounded_x, rounded_y)
                        # Use timestamp rounded to 0.1s to group detections in the same frame
                        rounded_ts = np.round(radar_detections['timestamp'] / 0.1) * 0.1
                        unique_keys = set(zip(rounded_ts, rounded_x, rounded_y))
                        unique_objects_9999 = len(unique_keys)
                        
                        if 9999 not in object_counts:
                            object_counts[9999] = 0
                        object_counts[9999] = unique_objects_9999
                        if VERBOSE:
                            total_detections = np.sum(mask_9999)
                            print("[INFO] Found {} unclassified radar detections (label 9999) in fused bbox data".format(
                                total_detections))
                            print("[INFO] Counted {} unique unclassified radar detection objects (grouped by time+position)".format(
                                unique_objects_9999))
        except Exception as e:
            if VERBOSE:
                print("[WARN] Could not count unclassified radar detections from fused CSV: {}".format(e))
    
    # Calculate total objects (tracked objects + unclassified radar detections)
    total_tracked_objects = len(trajs)
    total_unclassified_radar = object_counts.get(9999, 0)
    total_objects = total_tracked_objects + total_unclassified_radar
    
    # Build metrics dict
    metrics = {
        'duration_s': duration,
        'distance_m': total_distance,
        'max_velocity_ms': max_velocity,
        'avg_velocity_ms': avg_velocity,
        'max_acceleration_ms2': max_acceleration,
        'max_deceleration_ms2': max_deceleration,
        'total_objects': total_objects,
        'total_tracked_objects': total_tracked_objects,
        'total_unclassified_radar_detections': total_unclassified_radar,
        'object_counts_by_type': object_counts
    }
    
    # Add metadata fields
    for field in ['location', 'vehicle', 'passengers', 'road_type', 'road_condition', 'comments', 'maneuver']:
        metrics[field] = metadata.get(field, '')
    
    if VERBOSE:
        print("[INFO] Calculated key metrics: duration={:.1f}s, distance={:.1f}m, max_vel={:.1f}m/s, objects={}".format(
            duration, total_distance, max_velocity, len(trajs)))
    
    return metrics

# ====== STEP 5: TRAJECTORY EXTRACTION (Notebook -> .py) ===============
def step5_build_trajectories(fused_csv, odom_csv, bag_basename):
    """
    Implements your notebook's logic:
      - filter fused by frame_id if needed,
      - align with ego trajectory,
      - rotate/translate detections into ego global UTM,
      - associate across time -> IDs,
      - filter short tracks, smooth,
      - write *_trajectories_raw.csv
    Returns: trajs (dict: id -> list of rows) and the output CSV path.
    """
    # --- load ego odom
    ego = np.genfromtxt(odom_csv, delimiter=',', names=True)
    ego_ts = ego['timestamp']
    ego_xy = np.vstack((ego['position_x'], ego['position_y'])).T
    ego_qx, ego_qy, ego_qz, ego_qw = ego['orientation_x'], ego['orientation_y'], ego['orientation_z'], ego['orientation_w']
    ego_yaw = np.array([yaw_from_quat(eqx, eqy, eqz, eqw) for (eqx,eqy,eqz,eqw) in zip(ego_qx, ego_qy, ego_qz, ego_qw)])

    # --- load fused
    fused = np.genfromtxt(fused_csv, delimiter=',', names=True)
    # optional: keep only lidar-aligned frame_id
    # mask = (fused['frame_id'] == 'lidar_tc')  # may fail if dtype isn't string array in np
    # Keep all if dtype mixing; otherwise implement pandas. We'll proceed without frame_id filter.

    f_ts = fused['timestamp']
    f_xyz = np.vstack((fused['x'], fused['y'], fused['z'])).T
    f_lab = fused['label'] if 'label' in fused.dtype.names else np.full(len(f_ts), -1, dtype=np.int32)
    
    # Filter out unclassified radar detections (label 9999) - these are excluded from tracking
    mask_not_9999 = (f_lab != 9999)
    if not np.all(mask_not_9999):
        num_filtered = np.sum(~mask_not_9999)
        f_ts = f_ts[mask_not_9999]
        f_xyz = f_xyz[mask_not_9999, :]
        f_lab = f_lab[mask_not_9999]
        if VERBOSE:
            print("[INFO] Filtered out {} unclassified radar detections (label 9999) from trajectory building".format(num_filtered))

    # --- interpolate ego pose at detection times
    def interp_ego(ts_query):
        # linear interp of x, y, yaw
        x = np.interp(ts_query, ego_ts, ego_xy[:,0])
        y = np.interp(ts_query, ego_ts, ego_xy[:,1])
        yaw = np.interp(ts_query, ego_ts, ego_yaw)
        return x, y, yaw

    # --- transform detections into global using ego pose:
    # Here we assume fused positions are already in ego's local/lidar frame (x,y relative to ego),
    # so global = ego_xy + R(yaw) * local
    det_global = []
    if VERBOSE:
        print("[INFO] Transforming {} detections to global frame...".format(len(f_ts)))
    for i in range(len(f_ts)):
        gx, gy, gyaw = interp_ego(f_ts[i])
        lx, ly = f_xyz[i,0], f_xyz[i,1]
        rx, ry = rot2d(lx, ly, gyaw)
        det_global.append((f_ts[i], gx + rx, gy + ry, f_xyz[i,2], int(f_lab[i])))
        if VERBOSE and (i % 5000 == 0) and i > 0:
            print("[INFO] Transformed {} / {} detections...".format(i, len(f_ts)))
    # sort by time
    det_global.sort(key=lambda r: r[0])

    # --- multi-target data association (greedy NN with speed gating)
    tracks = {}     # id -> {'last_xy':(x,y), 't':t, 'path': [(t,x,y,z,label)]}
    next_id = 1

    # group by timestamp (frame-like)
    # (We can scan sequentially and associate per message timestamp)
    from collections import defaultdict
    frame = defaultdict(list)
    for (ts, gx, gy, gz, lab) in det_global:
        frame[ts].append((gx, gy, gz, lab))
    times = sorted(frame.keys())

    if VERBOSE:
        print("[INFO] Associating detections across {} timestamps...".format(len(times)))
    for idx_ts, ts in enumerate(times):
        dets = frame[ts]  # list of (x,y,z,lab)
        unmatched = set(range(len(dets)))
        candidates = []  # (dist, tid, j)
        for tid, rec in tracks.items():
            dt = max(1e-6, ts - rec['t'])
            gate = MAX_SPEED_M_S * dt + BASE_GATING_M
            for j, d in enumerate(dets):
                dxy = (d[0], d[1])
                if dist2d(rec['last_xy'], dxy) <= gate:
                    candidates.append((dist2d(rec['last_xy'], dxy), tid, j))
        candidates.sort(key=lambda x: x[0])

        matched_tids = set(); matched_dets = set()
        for (distv, tid, j) in candidates:
            if tid in matched_tids or j in matched_dets:
                continue
            # match
            gx, gy, gz, lab = dets[j]
            tracks[tid]['path'].append((ts, gx, gy, gz, lab))
            tracks[tid]['last_xy'] = (gx, gy)
            tracks[tid]['t'] = ts
            matched_tids.add(tid); matched_dets.add(j)

        # new tracks for leftovers
        for j in sorted(list(unmatched - matched_dets)):
            gx, gy, gz, lab = dets[j]
            tracks[next_id] = {'last_xy': (gx, gy),
                               't': ts,
                               'path': [(ts, gx, gy, gz, lab)]}
            next_id += 1
        if VERBOSE and (idx_ts % 200 == 0) and idx_ts > 0:
            print("[INFO] Associated up to timestamp index {} / {} (tracks so far: {})".format(idx_ts, len(times), len(tracks)))

    # --- filter short tracks & smooth
    trajs = {}  # id -> list of dict rows
    if VERBOSE:
        print("[INFO] Built {} tentative tracks. Filtering and smoothing...".format(len(tracks)))
    for tid, rec in tracks.items():
        path = rec['path']
        if len(path) < 2:
            continue
        t0, t1 = path[0][0], path[-1][0]
        if (t1 - t0) < MIN_TRACK_LIFETIME_S:
            continue
        ts = [p[0] for p in path]
        xs = [p[1] for p in path]
        ys = [p[2] for p in path]
        zs = [p[3] for p in path]
        labs = [p[4] for p in path]
        # Decide smoothing strategy: if static object label, use global average for x,y
        # Determine dominant label for this track
        try:
            from collections import Counter
            dom_label = Counter(labs).most_common(1)[0][0]
        except Exception:
            dom_label = labs[0]

        # Note: Label 9999 (unclassified radar detections) are already filtered out
        # at the data loading stage, so they won't reach this point

        # Heuristic: treat as static either by label or by small spatial spread
        is_label_static = (dom_label in STATIC_LABEL_IDS)
        span_dist = 0.0
        if len(xs) > 1:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_dist = math.hypot(max_x - min_x, max_y - min_y)
        is_heuristic_static = (span_dist < 1.0)  # within 1 m total span

        if (is_label_static or (len(STATIC_LABEL_IDS) == 0 and is_heuristic_static)) and len(xs) > 0:
            avg_x = sum(xs) / float(len(xs))
            avg_y = sum(ys) / float(len(ys))
            xs_s = [avg_x] * len(xs)
            ys_s = [avg_y] * len(ys)
            if VERBOSE:
                reason = "label" if is_label_static else "heuristic"
                print("[INFO] Track {} flagged static ({}). Using avg position ({:.2f},{:.2f}).".format(tid, reason, avg_x, avg_y))
        else:
            xs_s = moving_average(xs, SMOOTH_WINDOW_K)
            ys_s = moving_average(ys, SMOOTH_WINDOW_K)
        zs_s = moving_average(zs, SMOOTH_WINDOW_K)
        rows = []
        for i in range(len(ts)):
            rows.append({'ID': tid,
                         'time': ts[i],
                         'rosbagtime_int': int(ts[i]),  # used for slicing ranges
                         'x': xs_s[i], 'y': ys_s[i], 'z': zs_s[i],
                         'label': labs[i]})
        trajs[tid] = rows

    # --- write trajectories_raw.csv
    # Save to intermediate results folder parallel to ROOT_OUT
    out_csv = os.path.join(INTERMEDIATE_OUT, "{}_trajectories_raw.csv".format(bag_basename))
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['ID','time','rosbagtime_int','x','y','z','label'])
        for tid in sorted(trajs.keys()):
            for r in trajs[tid]:
                w.writerow([r['ID'], r['time'], r['rosbagtime_int'], r['x'], r['y'], r['z'], r['label']])
    print("[OK] Trajectories CSV -> {}".format(out_csv))
    return trajs, out_csv

# ====== STEP 6: WRITE KEY METRICS CSV ===================================
def step6_write_key_metrics_csv(metrics, bag_basename, bag_out_dir):
    """
    Write key metrics to CSV file in BAG_OUT_DIR.
    """
    out_csv = os.path.join(bag_out_dir, "{}_key_metrics.csv".format(bag_basename))
    
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        
        # Write header
        header = ['metric', 'value']
        w.writerow(header)
        
        # Write odometry-based metrics
        w.writerow(['duration_s', metrics['duration_s']])
        w.writerow(['distance_m', metrics['distance_m']])
        w.writerow(['max_velocity_ms', metrics['max_velocity_ms']])
        w.writerow(['avg_velocity_ms', metrics['avg_velocity_ms']])
        w.writerow(['max_acceleration_ms2', metrics['max_acceleration_ms2']])
        w.writerow(['max_deceleration_ms2', metrics['max_deceleration_ms2']])
        w.writerow(['total_objects', metrics['total_objects']])
        # Write object breakdown
        if 'total_tracked_objects' in metrics:
            w.writerow(['total_tracked_objects', metrics['total_tracked_objects']])
        if 'total_unclassified_radar_detections' in metrics:
            w.writerow(['total_unclassified_radar_detections', metrics['total_unclassified_radar_detections']])
        
        # Write object counts by type (using names instead of numeric labels)
        for label, count in metrics['object_counts_by_type'].items():
            object_name = OBJECT_TYPE_NAMES.get(label, 'unknown_{}'.format(label))
            w.writerow(['objects_type_{}'.format(object_name), count])
        
        # Write metadata fields
        for field in ['location', 'vehicle', 'passengers', 'road_type', 'road_condition', 'comments', 'maneuver']:
            w.writerow([field, metrics[field]])
    
    print("[OK] Key metrics CSV -> {}".format(out_csv))
    return out_csv

# ====== STEP 7: EXTRACT FRAMES PER OBJECT (±2s buffer) ================
def step7_extract_frames(bag, trajs, bag_basename, bag_out_dir):
    # Build per-object time windows
    ranges = {}
    for tid, rows in trajs.items():
        if not rows:
            continue
        tmins = [r['rosbagtime_int'] for r in rows]
        t0, t1 = min(tmins) - 2, max(tmins) + 2
        ranges[tid] = (t0, t1)

    # Walk the camera stream once and write frames for tids in range
    if VERBOSE:
        print("[INFO] Extracting frames for {} objects...".format(len(ranges)))
    for topic, msg, t in bag.read_messages(topics=[TOPIC_CAMERA_IMG]):
        ts = t.to_sec(); ts_i = int(ts); ms = int(ts * 1000.0)
        for tid, (lo, hi) in ranges.items():
            if lo <= ts_i <= hi:
                obj_dir = os.path.join(bag_out_dir, "{}_{}".format(bag_basename, tid))
                cam_dir = os.path.join(obj_dir, "camera")
                ensure_dir(cam_dir)
                img = convert_img_to_cv2(msg)
                out_path = os.path.join(cam_dir, "frame_{}.png".format(ms))
                if not cv2.imwrite(out_path, img):
                    print("[WARN] Failed to write {}".format(out_path))
    print("[OK] Frame extraction complete.")

# ====== STEP 7.5: MAKE PER-OBJECT FOLDERS, CSV, PLOTS =================
def step7p_finalize_objects(trajs, bag_basename, bag_out_dir):
    if VERBOSE:
        print("[INFO] Finalizing {} objects (CSV + plots)...".format(len(trajs)))
    for tid in sorted(trajs.keys()):
        obj_dir = os.path.join(bag_out_dir, "{}_{}".format(bag_basename, tid))
        cam_dir = os.path.join(obj_dir, "camera")
        ensure_dir(obj_dir); ensure_dir(cam_dir)
        # per-object CSV
        csv_path = os.path.join(obj_dir, "smoothed_trajectory_{}.csv".format(tid))
        with open(csv_path, 'w') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','x','y','z','label'])
            for r in trajs[tid]:
                w.writerow([r['time'], r['x'], r['y'], r['z'], r['label']])
        
        # Determine object type for filename
        labels = [r['label'] for r in trajs[tid]]
        try:
            from collections import Counter
            dom_label = Counter(labels).most_common(1)[0][0]
        except Exception:
            dom_label = labels[0] if labels else -1
        
        # Get object type name
        object_type_name = OBJECT_TYPE_NAMES.get(dom_label, 'unknown_{}'.format(dom_label))
        
        # plots
        ts = [r['time'] for r in trajs[tid]]
        xs = [r['x'] for r in trajs[tid]]
        ys = [r['y'] for r in trajs[tid]]
        plot_xy(ts, xs, ys, os.path.join(obj_dir, "x-y-{}-{}.png".format(tid, object_type_name)), "Object {} (x-y)".format(tid))
        plot_yt(ts, ys, os.path.join(obj_dir, "y-t-{}-{}.png".format(tid, object_type_name)), "Object {} (y-t)".format(tid))

# ====== STEP 8: MAKE WEB-PLAYABLE MP4s ================================
def step8_make_videos_and_copy_odom(bag_basename, bag_out_dir):
    # Videos
    for name in sorted(os.listdir(bag_out_dir)):
        obj_dir = os.path.join(bag_out_dir, name)
        if not os.path.isdir(obj_dir): continue
        cam_dir = os.path.join(obj_dir, "camera")
        if not os.path.isdir(cam_dir): continue
        images = sorted([f for f in os.listdir(cam_dir) if f.endswith(".png")])
        if not images: continue

        first = cv2.imread(os.path.join(cam_dir, images[0]))
        if first is None: continue
        h, w = first.shape[:2]
        out_mp4 = os.path.join(obj_dir, "{}.mp4".format(os.path.basename(obj_dir)))
        fourcc = choose_fourcc_h264()
        writer = cv2.VideoWriter(out_mp4, fourcc, FRAME_RATE, (w, h))
        if not writer.isOpened():
            # fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_mp4, fourcc, FRAME_RATE, (w, h))

        for fn in images:
            frame = cv2.imread(os.path.join(cam_dir, fn))
            if frame is None: continue
            writer.write(frame)
        writer.release()
        print("[OK] Video -> {}".format(out_mp4))

    # Copy odom CSV to the bag folder (already there, but ensure presence)
    src = os.path.join(bag_out_dir, "{}_novatel_odom_data.csv".format(bag_basename))
    dst = os.path.join(bag_out_dir, os.path.basename(src))
    if os.path.exists(src):
        try:
            shutil.copy(src, dst)
        except shutil.Error:
            pass

# ====== BATCH PROCESSING FUNCTIONS ====================================
def find_files_recursive(directory, pattern):
    """Find files matching pattern recursively in directory."""
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

def get_bag_files():
    """Get list of ROS 2 bag directories/files to process based on configuration."""
    if BAG_FILE is not None:
        # Single file mode
        if os.path.exists(BAG_FILE):
            return [BAG_FILE]
        else:
            print("[ERROR] ROS 2 bag path not found: {}".format(BAG_FILE))
            return []
    elif BAG_FOLDER is not None:
        # Folder mode
        if not os.path.exists(BAG_FOLDER):
            print("[ERROR] ROS 2 bag folder not found: {}".format(BAG_FOLDER))
            return []
        
        # Find ROS 2 bag directories by metadata.yaml, plus standalone .db3/.mcap files.
        if SEARCH_RECURSIVELY:
            metadata_files = find_files_recursive(BAG_FOLDER, "metadata.yaml")
            bag_files = sorted(set(os.path.dirname(path) for path in metadata_files))
            for path in find_files_recursive(BAG_FOLDER, "*.db3") + find_files_recursive(BAG_FOLDER, "*.mcap"):
                if not os.path.exists(os.path.join(os.path.dirname(path), "metadata.yaml")):
                    bag_files.append(path)
        else:
            metadata_path = os.path.join(BAG_FOLDER, "metadata.yaml")
            bag_files = [BAG_FOLDER] if os.path.exists(metadata_path) else []
            if not os.path.exists(metadata_path):
                bag_files.extend(glob.glob(os.path.join(BAG_FOLDER, "*.db3")))
                bag_files.extend(glob.glob(os.path.join(BAG_FOLDER, "*.mcap")))
        
        if not bag_files:
            print("[WARN] No ROS 2 bag metadata, .db3, or .mcap files found in folder: {}".format(BAG_FOLDER))
            return []
        
        print("[INFO] Found {} ROS 2 bag paths in folder: {}".format(len(bag_files), BAG_FOLDER))
        return sorted(set(bag_files))
    else:
        print("[ERROR] Neither BAG_FILE nor BAG_FOLDER is configured")
        return []

def process_single_bag(bag_path):
    """Process a single ROS 2 bag and return success status."""
    try:
        bag_basename = os.path.splitext(os.path.basename(os.path.normpath(bag_path)))[0]
        bag_out_dir = os.path.join(ROOT_OUT, "{}_extracted_data".format(bag_basename))
        ensure_dir(bag_out_dir)
        
        print("\n=== Processing: {} ===".format(os.path.basename(os.path.normpath(bag_path))))
        print("Output: {}".format(bag_out_dir))
        
        with Ros2Bag(bag_path) as bag:
            fused_csv = step1_dump_fused_bbox_csv(bag, bag_basename)
            metadata = step2_extract_metadata(bag)
            odom_csv = step3_dump_odom_csv(bag, bag_basename, bag_out_dir)
            trajs, traj_csv = step5_build_trajectories(fused_csv, odom_csv, bag_basename)
            
            # If debugging, keep only first N objects by sorted ID
            if DEBUG_FIRST_N_OBJECTS and DEBUG_FIRST_N_OBJECTS > 0:
                keep_ids = sorted(trajs.keys())[:DEBUG_FIRST_N_OBJECTS]
                trajs = {tid: trajs[tid] for tid in keep_ids}
                if VERBOSE:
                    print("[INFO] DEBUG: Restricting to first {} objects: {}".format(DEBUG_FIRST_N_OBJECTS, keep_ids))
            
            metrics = step4_calculate_key_metrics(odom_csv, metadata, trajs, fused_csv)
            step6_write_key_metrics_csv(metrics, bag_basename, bag_out_dir)
            step7p_finalize_objects(trajs, bag_basename, bag_out_dir)
            step7_extract_frames(bag, trajs, bag_basename, bag_out_dir)
        
        step8_make_videos_and_copy_odom(bag_basename, bag_out_dir)
        print("=== Completed: {} ===".format(os.path.basename(os.path.normpath(bag_path))))
        return True
        
    except Exception as e:
        print("[ERROR] Failed to process {}: {}".format(bag_path, str(e)))
        if VERBOSE:
            import traceback
            traceback.print_exc()
        return False

# ====== MAIN ==========================================================
def main():
    print("=== ROS 2 Jazzy Perception Data Pipeline start ===")
    
    # Get list of ROS 2 bag paths to process
    bag_files = get_bag_files()
    if not bag_files:
        print("[ERROR] No ROS 2 bag paths to process")
        return
    
    print("Processing {} ROS 2 bag paths...".format(len(bag_files)))
    
    # Process each bag
    successful = 0
    failed = 0
    
    for i, bag_file in enumerate(bag_files, 1):
        print("\n[{}/{}] Processing: {}".format(i, len(bag_files), os.path.basename(bag_file)))
        
        if process_single_bag(bag_file):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n=== Batch Processing Complete ===")
    print("Successful: {}".format(successful))
    print("Failed: {}".format(failed))
    print("Results under: {}".format(ROOT_OUT))

if __name__ == "__main__":
    main()
