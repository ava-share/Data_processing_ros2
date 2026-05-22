#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, math, shutil, errno, glob
import fnmatch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt

from ros2_bag_utils import Ros2Bag, normalize_topic, stamp_to_sec

# ====== CONFIG =========================================================
EXTRACTION_DATE = "12172025"   # e.g., run date token you want in the root folder name

# BATCH PROCESSING: Set one of these options
# Option 1: Process single ROS 2 bag directory, .db3, or .mcap file
BAG_FILE = "/home/avresearch/Downloads/ctrl6_20260423_131927_0.mcap"
BAG_FOLDER = None
SEARCH_RECURSIVELY = True

# # Option 2: Process all ROS 2 bags in a folder
# BAG_FOLDER = "/home/avresearch/Downloads/perception_output_planning_2025-11-20_11-20-12"  # Set to None to use single file
# BAG_FILE = None  # Set to None to use folder processing
# Folder search options
# SEARCH_RECURSIVELY = True searches all subdirectories; False searches only the top-level folder.


# Control topics in the ctrl6 MCAP bag
TOPIC_ODOM = "/novatel/oem7/odom"                              # nav_msgs/Odometry
TOPIC_LAT_CTRL_PERF = "/lat_ctrl_perf"                         # geometry_msgs/Vector3Stamped (y=cross-track error, z=yaw error)
TOPIC_CTRL_REF_TWIST = "/ctrl_ref_twist"                       # geometry_msgs/TwistStamped (velocity reference)
TOPIC_LAT_CTRL_CMD = "/lat_ctrl_cmd"                           # geometry_msgs/Vector3Stamped
TOPIC_CTRL_REF_CURV = "/ctrl_ref_curv"                         # geometry_msgs/PointStamped
TOPIC_CTRL_REF_POSE = "/ctrl_ref_pose"                         # geometry_msgs/PoseStamped
TOPIC_STEER_CTRL_CMD = "/steer_ctrl_cmd"                       # geometry_msgs/Vector3Stamped
TOPIC_PROTECTION_LEVELS = "/protection_levels"                 # geometry_msgs/Vector3Stamped

# Raptor DBW topics present in the example control bag
TOPIC_MISC_REPORT = "/raptor_dbw_interface/misc_report"        # raptor_dbw_msgs/MiscReport
TOPIC_STEERING_REPORT = "/raptor_dbw_interface/steering_report" # raptor_dbw_msgs/SteeringReport
TOPIC_STEERING_CMD = "/raptor_dbw_interface/steering_cmd"      # raptor_dbw_msgs/SteeringCmd
TOPIC_ACCELERATOR_CMD = "/raptor_dbw_interface/accelerator_pedal_cmd" # raptor_dbw_msgs/AcceleratorPedalCmd
TOPIC_BRAKE_CMD = "/raptor_dbw_interface/brake_cmd"            # raptor_dbw_msgs/BrakeCmd
TOPIC_GEAR_REPORT = "/raptor_dbw_interface/gear_report"        # raptor_dbw_msgs/GearReport
TOPIC_GLOBAL_ENABLE_CMD = "/raptor_dbw_interface/global_enable_cmd" # raptor_dbw_msgs/GlobalEnableCmd

# Verbosity and debugging
VERBOSE = True        # print progress messages during processing
DEBUG_FIRST_N_MESSAGES = 0  # set to >0 to process only first N messages (for debugging)

# ====== PATHS ==========================================================
def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Initialize output directories (will be set per bag in batch processing)
ROOT_OUT = "Control_data_{}".format(EXTRACTION_DATE)
ensure_dir(ROOT_OUT)

# ====== UTILITIES ======================================================
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

def debug_bag_topics(bag_path):
    """Debug function to print all available topics in the ROS 2 bag."""
    print("\n=== DEBUG: Available topics in {} ===".format(os.path.basename(os.path.normpath(bag_path))))
    try:
        with Ros2Bag(bag_path) as bag:
            info = bag.get_type_and_topic_info()
            print("Topics found:")
            for topic, topic_info in info.topics.items():
                print("  {}: {} ({} messages)".format(topic, topic_info.msg_type, topic_info.message_count))
            
            print("\nLooking for control topics:")
            control_topics = [
                TOPIC_ODOM, TOPIC_LAT_CTRL_PERF, TOPIC_CTRL_REF_TWIST, TOPIC_LAT_CTRL_CMD,
                TOPIC_CTRL_REF_CURV, TOPIC_CTRL_REF_POSE, TOPIC_STEER_CTRL_CMD,
                TOPIC_PROTECTION_LEVELS, TOPIC_MISC_REPORT, TOPIC_STEERING_REPORT,
                TOPIC_STEERING_CMD, TOPIC_ACCELERATOR_CMD, TOPIC_BRAKE_CMD,
                TOPIC_GEAR_REPORT, TOPIC_GLOBAL_ENABLE_CMD,
            ]
            topics_by_normalized = dict((normalize_topic(topic), topic) for topic in info.topics)
            for topic in control_topics:
                actual_topic = topics_by_normalized.get(normalize_topic(topic))
                if actual_topic is not None:
                    print("  Found: {} ({} messages)".format(actual_topic, info.topics[actual_topic].message_count))
                else:
                    print("  Missing: {}".format(topic))
    except Exception as e:
        print("Error reading bag info: {}".format(e))
    print("=" * 60)

def plot_time_series(ts, values, out_path, title, ylabel):
    plt.figure()
    plt.plot(ts, values, '-o', markersize=2)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def msg_stamp_to_sec(msg, fallback):
    return stamp_to_sec(getattr(getattr(msg, 'header', None), 'stamp', None), fallback=fallback)

def read_scalar_attr(obj, names, default=0.0):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
    return default

# ====== STEP 1: READ CONTROL PERFORMANCE DATA =========================
def step1_dump_lat_ctrl_perf_csv(bag, bag_basename, bag_out_dir):
    """
    Read geometry_msgs/Vector3Stamped from TOPIC_LAT_CTRL_PERF
    and dump CSV with timestamp, cross_track_error, yaw_error.
    """
    out_csv = os.path.join(bag_out_dir, "{}_lat_ctrl_perf.csv".format(bag_basename))
    msg_count = 0
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'cross_track_error', 'yaw_error'])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_LAT_CTRL_PERF]):
            msg_count += 1
            ts = msg_stamp_to_sec(msg, t)
            cross_track_error = msg.vector.y
            yaw_error = msg.vector.z
            w.writerow([ts, cross_track_error, yaw_error])
            if VERBOSE and (msg_count % 2000 == 0):
                print("[INFO] Read {} lat_ctrl_perf messages...".format(msg_count))
    
    if msg_count == 0:
        print("[WARN] No messages found for topic: {}".format(TOPIC_LAT_CTRL_PERF))
    else:
        print("[OK] Lateral control performance CSV -> {} ({} messages)".format(out_csv, msg_count))
    return out_csv

# ====== STEP 2: READ VELOCITY COMMANDED DATA ==========================
def step2_dump_velocity_cmd_csv(bag, bag_basename, bag_out_dir):
    """
    Read geometry_msgs/TwistStamped from TOPIC_CTRL_REF_TWIST
    and dump CSV with timestamp, velocity_commanded.
    """
    out_csv = os.path.join(bag_out_dir, "{}_velocity_cmd.csv".format(bag_basename))
    msg_count = 0
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'velocity_commanded'])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_CTRL_REF_TWIST]):
            msg_count += 1
            ts = msg_stamp_to_sec(msg, t)
            # Calculate velocity magnitude from linear components
            vx = msg.twist.linear.x
            vy = msg.twist.linear.y
            velocity_commanded = math.sqrt(vx*vx + vy*vy)
            w.writerow([ts, velocity_commanded])
            if VERBOSE and (msg_count % 2000 == 0):
                print("[INFO] Read {} velocity command messages...".format(msg_count))
    
    if msg_count == 0:
        print("[WARN] No messages found for topic: {}".format(TOPIC_CTRL_REF_TWIST))
    else:
        print("[OK] Velocity commanded CSV -> {} ({} messages)".format(out_csv, msg_count))
    return out_csv

# ====== STEP 3: READ ACCELERATION COMMANDED DATA =====================
def step3_dump_acceleration_cmd_csv(bag, bag_basename, bag_out_dir):
    """
    Read geometry_msgs/Vector3Stamped from TOPIC_LAT_CTRL_CMD
    and dump CSV with timestamp, acceleration_commanded.
    """
    out_csv = os.path.join(bag_out_dir, "{}_acceleration_cmd.csv".format(bag_basename))
    msg_count = 0
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'acceleration_commanded'])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_LAT_CTRL_CMD]):
            msg_count += 1
            ts = msg_stamp_to_sec(msg, t)
            # Preserve the controller's signed longitudinal value when available.
            acceleration_commanded = msg.vector.x
            w.writerow([ts, acceleration_commanded])
            if VERBOSE and (msg_count % 2000 == 0):
                print("[INFO] Read {} acceleration command messages...".format(msg_count))
    
    if msg_count == 0:
        print("[WARN] No messages found for topic: {}".format(TOPIC_LAT_CTRL_CMD))
    else:
        print("[OK] Acceleration commanded CSV -> {} ({} messages)".format(out_csv, msg_count))
    return out_csv

# ====== STEP 4: READ CURVATURE REFERENCE DATA ========================
def step4_dump_curvature_ref_csv(bag, bag_basename, bag_out_dir):
    """
    Read geometry_msgs/PointStamped from TOPIC_CTRL_REF_CURV
    and dump CSV with timestamp, curvature_reference.
    """
    out_csv = os.path.join(bag_out_dir, "{}_curvature_ref.csv".format(bag_basename))
    msg_count = 0
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'curvature_reference'])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_CTRL_REF_CURV]):
            msg_count += 1
            ts = msg_stamp_to_sec(msg, t)
            curvature_reference = msg.point.x
            w.writerow([ts, curvature_reference])
            if VERBOSE and (msg_count % 2000 == 0):
                print("[INFO] Read {} curvature reference messages...".format(msg_count))
    
    if msg_count == 0:
        print("[WARN] No messages found for topic: {}".format(TOPIC_CTRL_REF_CURV))
    else:
        print("[OK] Curvature reference CSV -> {} ({} messages)".format(out_csv, msg_count))
    return out_csv

# ====== STEP 5: READ AUTONOMOUS MODE DATA ============================
def step5_dump_autonomous_mode_csv(bag, bag_basename, bag_out_dir):
    """
    Read geometry_msgs/Vector3Stamped from TOPIC_STEER_CTRL_CMD
    and dump CSV with timestamp, command vector, and autonomous mode indicator.
    """
    out_csv = os.path.join(bag_out_dir, "{}_autonomous_mode.csv".format(bag_basename))
    msg_count = 0
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'autonomous_mode', 'cmd_x', 'cmd_y', 'cmd_z'])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_STEER_CTRL_CMD]):
            msg_count += 1
            ts = msg_stamp_to_sec(msg, t)
            # If steer_ctrl_cmd exists, it's autonomous mode
            autonomous_mode = 1
            w.writerow([ts, autonomous_mode, msg.vector.x, msg.vector.y, msg.vector.z])
            if VERBOSE and (msg_count % 2000 == 0):
                print("[INFO] Read {} autonomous mode messages...".format(msg_count))
    
    if msg_count == 0:
        print("[WARN] No messages found for topic: {}".format(TOPIC_STEER_CTRL_CMD))
    else:
        print("[OK] Autonomous mode CSV -> {} ({} messages)".format(out_csv, msg_count))
    return out_csv

# ====== STEP 6: READ ODOMETRY DATA ====================================
def step6_dump_odom_csv(bag, bag_basename, bag_out_dir):
    """
    Read nav_msgs/Odometry from TOPIC_ODOM
    and dump CSV with timestamp, position, orientation, linear velocity, angular velocity.
    Saves directly to bag_out_dir (not intermediate).
    """
    out_csv = os.path.join(bag_out_dir, "{}_odom.csv".format(bag_basename))
    msg_count = 0
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'position_x', 'position_y', 'position_z', 
                   'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                   'linear_velocity_x', 'linear_velocity_y', 'linear_velocity_z',
                   'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_ODOM]):
            msg_count += 1
            ts = msg_stamp_to_sec(msg, t)
            
            # Extract position
            pos_x = msg.pose.pose.position.x
            pos_y = msg.pose.pose.position.y
            pos_z = msg.pose.pose.position.z
            
            # Extract orientation (quaternion)
            ori_x = msg.pose.pose.orientation.x
            ori_y = msg.pose.pose.orientation.y
            ori_z = msg.pose.pose.orientation.z
            ori_w = msg.pose.pose.orientation.w
            
            # Extract linear velocity
            lin_vel_x = msg.twist.twist.linear.x
            lin_vel_y = msg.twist.twist.linear.y
            lin_vel_z = msg.twist.twist.linear.z
            
            # Extract angular velocity
            ang_vel_x = msg.twist.twist.angular.x
            ang_vel_y = msg.twist.twist.angular.y
            ang_vel_z = msg.twist.twist.angular.z
            
            w.writerow([ts, pos_x, pos_y, pos_z, 
                       ori_x, ori_y, ori_z, ori_w,
                       lin_vel_x, lin_vel_y, lin_vel_z,
                       ang_vel_x, ang_vel_y, ang_vel_z])
            if VERBOSE and (msg_count % 2000 == 0):
                print("[INFO] Read {} odometry messages...".format(msg_count))
    
    if msg_count == 0:
        print("[WARN] No messages found for topic: {}".format(TOPIC_ODOM))
    else:
        print("[OK] Odometry CSV -> {} ({} messages)".format(out_csv, msg_count))
    return out_csv

# ====== STEP 7: READ RAPTOR DBW DATA ==================================
def step7_dump_raptor_dbw_csvs(bag, bag_basename, bag_out_dir):
    """
    Extract additional controller and DBW topics present in the ctrl6 MCAP bag.
    Returns a dict of output CSV paths.
    """
    outputs = {}

    def write_topic_csv(topic, suffix, header, row_fn):
        out_csv = os.path.join(bag_out_dir, "{}_{}.csv".format(bag_basename, suffix))
        msg_count = 0
        with open(out_csv, 'w') as f:
            w = csv.writer(f)
            w.writerow(header)
            for _, msg, t in bag.read_messages(topics=[topic]):
                msg_count += 1
                w.writerow(row_fn(msg, t))
                if VERBOSE and (msg_count % 2000 == 0):
                    print("[INFO] Read {} {} messages...".format(msg_count, suffix))
        if msg_count == 0:
            print("[WARN] No messages found for topic: {}".format(topic))
        else:
            print("[OK] {} CSV -> {} ({} messages)".format(suffix, out_csv, msg_count))
        outputs[suffix] = out_csv

    write_topic_csv(
        TOPIC_CTRL_REF_POSE,
        "ctrl_ref_pose",
        ['timestamp', 'position_x', 'position_y', 'position_z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'],
        lambda msg, t: [
            msg_stamp_to_sec(msg, t),
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
    )
    write_topic_csv(
        TOPIC_PROTECTION_LEVELS,
        "protection_levels",
        ['timestamp', 'x', 'y', 'z'],
        lambda msg, t: [msg_stamp_to_sec(msg, t), msg.vector.x, msg.vector.y, msg.vector.z],
    )
    write_topic_csv(
        TOPIC_MISC_REPORT,
        "misc_report",
        ['timestamp', 'vehicle_speed_kmh', 'vehicle_speed_ms'],
        lambda msg, t: [
            msg_stamp_to_sec(msg, t),
            read_scalar_attr(msg, ['vehicle_speed']),
            read_scalar_attr(msg, ['vehicle_speed']) / 3.6,
        ],
    )
    write_topic_csv(
        TOPIC_STEERING_REPORT,
        "steering_report",
        ['timestamp', 'steering_wheel_angle_deg', 'steering_wheel_angle_cmd_deg', 'steering_wheel_torque', 'fault_steering_system', 'steering_overheat_warning'],
        lambda msg, t: [
            msg_stamp_to_sec(msg, t),
            read_scalar_attr(msg, ['steering_wheel_angle']),
            read_scalar_attr(msg, ['steering_wheel_angle_cmd']),
            read_scalar_attr(msg, ['steering_wheel_torque']),
            int(bool(getattr(msg, 'fault_steering_system', False))),
            int(bool(getattr(msg, 'steering_overheat_warning', False))),
        ],
    )
    write_topic_csv(
        TOPIC_STEERING_CMD,
        "steering_cmd",
        ['timestamp', 'angle_cmd_deg', 'angle_velocity_deg_s', 'vehicle_curvature_cmd'],
        lambda msg, t: [
            msg_stamp_to_sec(msg, t),
            read_scalar_attr(msg, ['angle_cmd', 'steering_wheel_angle_cmd']),
            read_scalar_attr(msg, ['angle_velocity']),
            read_scalar_attr(msg, ['vehicle_curvature_cmd']),
        ],
    )
    write_topic_csv(
        TOPIC_ACCELERATOR_CMD,
        "accelerator_pedal_cmd",
        ['timestamp', 'speed_cmd_ms', 'accel_limit_ms2', 'accel_positive_jerk_limit_ms3'],
        lambda msg, t: [
            msg_stamp_to_sec(msg, t),
            read_scalar_attr(msg, ['speed_cmd']),
            read_scalar_attr(msg, ['accel_limit']),
            read_scalar_attr(msg, ['accel_positive_jerk_limit']),
        ],
    )
    write_topic_csv(
        TOPIC_BRAKE_CMD,
        "brake_cmd",
        ['timestamp', 'decel_limit_ms2', 'decel_negative_jerk_limit_ms3'],
        lambda msg, t: [
            msg_stamp_to_sec(msg, t),
            read_scalar_attr(msg, ['decel_limit']),
            read_scalar_attr(msg, ['decel_negative_jerk_limit']),
        ],
    )
    write_topic_csv(
        TOPIC_GLOBAL_ENABLE_CMD,
        "global_enable_cmd",
        ['timestamp', 'command_present'],
        lambda msg, t: [stamp_to_sec(None, fallback=t), 1],
    )
    return outputs

# ====== STEP 8: CALCULATE CONTROL KEY METRICS ========================
def step8_calculate_control_metrics(lat_ctrl_perf_csv, velocity_cmd_csv, acceleration_cmd_csv, autonomous_mode_csv, dbw_csvs=None):
    """
    Calculate control key metrics from the control data.
    Returns dict with all metrics.
    """
    metrics = {}
    dbw_csvs = dbw_csvs or {}
    
    # Load lateral control performance data
    if os.path.exists(lat_ctrl_perf_csv):
        lat_perf = np.genfromtxt(lat_ctrl_perf_csv, delimiter=',', names=True)
        if len(lat_perf) > 0:
            cross_track_errors = lat_perf['cross_track_error']
            yaw_errors = lat_perf['yaw_error']
            
            # Calculate max errors (absolute values)
            max_cross_track_error = np.max(np.abs(cross_track_errors))
            max_yaw_error = np.max(np.abs(yaw_errors))
            
            metrics['max_cross_track_error'] = max_cross_track_error
            metrics['max_yaw_error'] = max_yaw_error
        else:
            metrics['max_cross_track_error'] = 0.0
            metrics['max_yaw_error'] = 0.0
    else:
        metrics['max_cross_track_error'] = 0.0
        metrics['max_yaw_error'] = 0.0
    
    # Load velocity commanded data
    if os.path.exists(velocity_cmd_csv):
        vel_cmd = np.genfromtxt(velocity_cmd_csv, delimiter=',', names=True)
        if len(vel_cmd) > 0:
            velocities = vel_cmd['velocity_commanded']
            max_velocity_commanded = np.max(velocities)
            metrics['max_velocity_commanded'] = max_velocity_commanded
        else:
            metrics['max_velocity_commanded'] = 0.0
    else:
        metrics['max_velocity_commanded'] = 0.0

    misc_report_csv = dbw_csvs.get('misc_report')
    if misc_report_csv and os.path.exists(misc_report_csv):
        misc = np.genfromtxt(misc_report_csv, delimiter=',', names=True)
        if len(misc) > 0:
            metrics['max_vehicle_speed_ms'] = float(np.max(misc['vehicle_speed_ms']))
            metrics['avg_vehicle_speed_ms'] = float(np.mean(misc['vehicle_speed_ms']))
        else:
            metrics['max_vehicle_speed_ms'] = 0.0
            metrics['avg_vehicle_speed_ms'] = 0.0
    else:
        metrics['max_vehicle_speed_ms'] = 0.0
        metrics['avg_vehicle_speed_ms'] = 0.0
    
    # Load acceleration commanded data
    if os.path.exists(acceleration_cmd_csv):
        acc_cmd = np.genfromtxt(acceleration_cmd_csv, delimiter=',', names=True)
        if len(acc_cmd) > 0:
            accelerations = acc_cmd['acceleration_commanded']
            max_acceleration_commanded = np.max(accelerations)
            # Calculate max deceleration (most negative acceleration)
            max_deceleration_commanded = np.min(accelerations)  # Most negative value
            metrics['max_acceleration_commanded'] = max_acceleration_commanded
            metrics['max_deceleration_commanded'] = max_deceleration_commanded
        else:
            metrics['max_acceleration_commanded'] = 0.0
            metrics['max_deceleration_commanded'] = 0.0
    else:
        metrics['max_acceleration_commanded'] = 0.0
        metrics['max_deceleration_commanded'] = 0.0
    
    # Calculate autonomous mode duration and distance (estimated)
    autonomous_duration = 0.0
    autonomous_distance = 0.0
    
    if os.path.exists(autonomous_mode_csv):
        auto_mode = np.genfromtxt(autonomous_mode_csv, delimiter=',', names=True)
        if len(auto_mode) > 0:
            # Autonomous window from first to last steer_ctrl_cmd
            auto_timestamps = auto_mode['timestamp']
            auto_start = np.min(auto_timestamps)
            auto_end = np.max(auto_timestamps)
            autonomous_duration = max(0.0, float(auto_end - auto_start))
            
            # Estimate distance from measured DBW speed when available, otherwise command speed.
            avg_speed_auto = 0.0
            if misc_report_csv and os.path.exists(misc_report_csv):
                misc = np.genfromtxt(misc_report_csv, delimiter=',', names=True)
                if len(misc) > 0:
                    mask = (misc['timestamp'] >= auto_start) & (misc['timestamp'] <= auto_end)
                    speed_in_auto = misc['vehicle_speed_ms'][mask]
                    if len(speed_in_auto) > 0:
                        avg_speed_auto = float(np.mean(speed_in_auto))
            if avg_speed_auto == 0.0 and os.path.exists(velocity_cmd_csv):
                vel_cmd = np.genfromtxt(velocity_cmd_csv, delimiter=',', names=True)
                if len(vel_cmd) > 0:
                    mask = (vel_cmd['timestamp'] >= auto_start) & (vel_cmd['timestamp'] <= auto_end)
                    vel_in_auto = vel_cmd['velocity_commanded'][mask]
                    if len(vel_in_auto) > 0:
                        avg_speed_auto = float(np.mean(vel_in_auto))
            autonomous_distance = avg_speed_auto * autonomous_duration
    
    metrics['autonomous_duration_s'] = autonomous_duration
    metrics['autonomous_distance_m'] = autonomous_distance
    
    if VERBOSE:
        print("[INFO] Control metrics: max_cross_track_error={:.3f}m, max_cmd_velocity={:.2f}m/s, max_vehicle_speed={:.2f}m/s".format(
            metrics['max_cross_track_error'], metrics['max_velocity_commanded'], metrics['max_vehicle_speed_ms']))
        print("[INFO] Autonomous mode: duration={:.1f}s, distance={:.1f}m".format(
            autonomous_duration, autonomous_distance))
    
    return metrics

# ====== STEP 9: WRITE CONTROL KEY METRICS CSV ========================
def step9_write_control_metrics_csv(metrics, bag_basename, bag_out_dir):
    """
    Write control key metrics to CSV file in BAG_OUT_DIR.
    """
    out_csv = os.path.join(bag_out_dir, "{}_control_key_metrics.csv".format(bag_basename))
    
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        
        # Write header
        header = ['metric', 'value']
        w.writerow(header)
        
        # Write control metrics
        w.writerow(['autonomous_duration_s', metrics['autonomous_duration_s']])
        w.writerow(['autonomous_distance_m', metrics['autonomous_distance_m']])
        w.writerow(['max_cross_track_error', metrics['max_cross_track_error']])
        w.writerow(['max_yaw_error', metrics['max_yaw_error']])
        w.writerow(['max_velocity_commanded', metrics['max_velocity_commanded']])
        w.writerow(['max_vehicle_speed_ms', metrics['max_vehicle_speed_ms']])
        w.writerow(['avg_vehicle_speed_ms', metrics['avg_vehicle_speed_ms']])
        w.writerow(['max_acceleration_commanded', metrics['max_acceleration_commanded']])
        w.writerow(['max_deceleration_commanded', metrics['max_deceleration_commanded']])
    
    print("[OK] Control key metrics CSV -> {}".format(out_csv))
    return out_csv

# ====== STEP 10: CREATE CONTROL PLOTS =================================
def step10_create_control_plots(lat_ctrl_perf_csv, velocity_cmd_csv, acceleration_cmd_csv, bag_basename, bag_out_dir, dbw_csvs=None):
    """
    Create plots for control data visualization.
    """
    plots_dir = os.path.join(bag_out_dir, "control_plots")
    ensure_dir(plots_dir)
    
    # Plot lateral error (cross-track error) over time
    if os.path.exists(lat_ctrl_perf_csv):
        lat_perf = np.genfromtxt(lat_ctrl_perf_csv, delimiter=',', names=True)
        if len(lat_perf) > 0:
            ts = lat_perf['timestamp']
            cross_track_errors = lat_perf['cross_track_error']
            yaw_errors = lat_perf['yaw_error']
            
            plot_time_series(ts, cross_track_errors, 
                           os.path.join(plots_dir, "lateral_error_vs_time.png"),
                           "Lateral Error vs Time", "Lateral Error (m)")
            
            plot_time_series(ts, yaw_errors,
                           os.path.join(plots_dir, "yaw_error_vs_time.png"),
                           "Yaw Error vs Time", "Yaw Error (rad)")
    
    # Plot velocity commanded over time
    if os.path.exists(velocity_cmd_csv):
        vel_cmd = np.genfromtxt(velocity_cmd_csv, delimiter=',', names=True)
        if len(vel_cmd) > 0:
            ts = vel_cmd['timestamp']
            velocities = vel_cmd['velocity_commanded']
            
            plot_time_series(ts, velocities,
                           os.path.join(plots_dir, "velocity_commanded_vs_time.png"),
                           "Velocity Commanded vs Time", "Velocity (m/s)")
    
    # Plot acceleration commanded over time
    if os.path.exists(acceleration_cmd_csv):
        acc_cmd = np.genfromtxt(acceleration_cmd_csv, delimiter=',', names=True)
        if len(acc_cmd) > 0:
            ts = acc_cmd['timestamp']
            accelerations = acc_cmd['acceleration_commanded']
            
            plot_time_series(ts, accelerations,
                           os.path.join(plots_dir, "acceleration_commanded_vs_time.png"),
                           "Acceleration Commanded vs Time", "Acceleration (m/s^2)")

    dbw_csvs = dbw_csvs or {}
    misc_report_csv = dbw_csvs.get('misc_report')
    if misc_report_csv and os.path.exists(misc_report_csv):
        misc = np.genfromtxt(misc_report_csv, delimiter=',', names=True)
        if len(misc) > 0:
            plot_time_series(misc['timestamp'], misc['vehicle_speed_ms'],
                           os.path.join(plots_dir, "vehicle_speed_vs_time.png"),
                           "Vehicle Speed vs Time", "Speed (m/s)")

    steering_report_csv = dbw_csvs.get('steering_report')
    if steering_report_csv and os.path.exists(steering_report_csv):
        steering = np.genfromtxt(steering_report_csv, delimiter=',', names=True)
        if len(steering) > 0:
            plot_time_series(steering['timestamp'], steering['steering_wheel_angle_deg'],
                           os.path.join(plots_dir, "steering_wheel_angle_vs_time.png"),
                           "Steering Wheel Angle vs Time", "Angle (deg)")
    
    print("[OK] Control plots created in -> {}".format(plots_dir))

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
        bag_out_dir = os.path.join(ROOT_OUT, "{}_control_data".format(bag_basename))
        ensure_dir(bag_out_dir)
        
        print("\n=== Processing: {} ===".format(os.path.basename(os.path.normpath(bag_path))))
        print("Output: {}".format(bag_out_dir))
        
        # Debug: Print available topics
        debug_bag_topics(bag_path)
        
        with Ros2Bag(bag_path) as bag:
            # Extract control data
            lat_ctrl_perf_csv = step1_dump_lat_ctrl_perf_csv(bag, bag_basename, bag_out_dir)
            velocity_cmd_csv = step2_dump_velocity_cmd_csv(bag, bag_basename, bag_out_dir)
            acceleration_cmd_csv = step3_dump_acceleration_cmd_csv(bag, bag_basename, bag_out_dir)
            curvature_ref_csv = step4_dump_curvature_ref_csv(bag, bag_basename, bag_out_dir)
            autonomous_mode_csv = step5_dump_autonomous_mode_csv(bag, bag_basename, bag_out_dir)
            
            # Extract odometry data (saves directly to bag_out_dir)
            step6_dump_odom_csv(bag, bag_basename, bag_out_dir)

            # Extract Raptor DBW report/command data when present
            dbw_csvs = step7_dump_raptor_dbw_csvs(bag, bag_basename, bag_out_dir)
            
            # Calculate control metrics
            metrics = step8_calculate_control_metrics(
                lat_ctrl_perf_csv,
                velocity_cmd_csv,
                acceleration_cmd_csv,
                autonomous_mode_csv,
                dbw_csvs,
            )
            
            # Write metrics and create plots
            step9_write_control_metrics_csv(metrics, bag_basename, bag_out_dir)
            step10_create_control_plots(
                lat_ctrl_perf_csv,
                velocity_cmd_csv,
                acceleration_cmd_csv,
                bag_basename,
                bag_out_dir,
                dbw_csvs,
            )
        
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
    print("=== ROS 2 Jazzy Control Data Pipeline start ===")
    
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
    print("\n=== Control Data Processing Complete ===")
    print("Successful: {}".format(successful))
    print("Failed: {}".format(failed))
    print("Results under: {}".format(ROOT_OUT))

if __name__ == "__main__":
    main()
