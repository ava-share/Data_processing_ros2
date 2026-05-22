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
BAG_FILE = "/home/avresearch/Downloads/Route3AutonomousTesting11_20_2025_ctrl6_2025-11-20-13-15-07"
BAG_FOLDER = None
SEARCH_RECURSIVELY = True

# # Option 2: Process all ROS 2 bags in a folder
# BAG_FOLDER = "/home/avresearch/Downloads/perception_output_planning_2025-11-20_11-20-12"  # Set to None to use single file
# BAG_FILE = None  # Set to None to use folder processing
# Folder search options
# SEARCH_RECURSIVELY = True searches all subdirectories; False searches only the top-level folder.


# Control topics in your bag
TOPIC_ODOM = "novatel/oem7/odom"                    # nav_msgs/Odometry
TOPIC_LAT_CTRL_PERF = "lat_ctrl_perf"               # geometry_msgs/Vector3Stamped (y=cross-track error, z=yaw error)
TOPIC_CTRL_REF_TWIST = "ctrl_ref_twist"             # geometry_msgs/TwistStamped (velocity commanded)
TOPIC_LAT_CTRL_CMD = "lat_ctrl_cmd"                 # geometry_msgs/Vector3Stamped (acceleration commanded)
TOPIC_CTRL_REF_CURV = "ctrl_ref_curv"               # geometry_msgs/PointStamped (curvature reference)
TOPIC_STEER_CTRL_CMD = "steer_ctrl_cmd"             # geometry_msgs/Vector3Stamped (autonomous mode indicator)

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
            control_topics = [TOPIC_ODOM, TOPIC_LAT_CTRL_PERF, TOPIC_CTRL_REF_TWIST, TOPIC_LAT_CTRL_CMD, TOPIC_CTRL_REF_CURV, TOPIC_STEER_CTRL_CMD]
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
            ts = stamp_to_sec(getattr(msg.header, 'stamp', None), fallback=t)
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
            ts = stamp_to_sec(getattr(msg.header, 'stamp', None), fallback=t)
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
            ts = stamp_to_sec(getattr(msg.header, 'stamp', None), fallback=t)
            # Calculate acceleration magnitude from vector components
            ax = msg.vector.x
            ay = msg.vector.y
            acceleration_commanded = math.sqrt(ax*ax + ay*ay)
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
            ts = stamp_to_sec(getattr(msg.header, 'stamp', None), fallback=t)
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
    and dump CSV with timestamp and autonomous mode indicator.
    """
    out_csv = os.path.join(bag_out_dir, "{}_autonomous_mode.csv".format(bag_basename))
    msg_count = 0
    with open(out_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'autonomous_mode'])
        for topic, msg, t in bag.read_messages(topics=[TOPIC_STEER_CTRL_CMD]):
            msg_count += 1
            ts = stamp_to_sec(getattr(msg.header, 'stamp', None), fallback=t)
            # If steer_ctrl_cmd exists, it's autonomous mode
            autonomous_mode = 1
            w.writerow([ts, autonomous_mode])
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
            ts = stamp_to_sec(getattr(msg.header, 'stamp', None), fallback=t)
            
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

# ====== STEP 7: CALCULATE CONTROL KEY METRICS ========================
def step7_calculate_control_metrics(lat_ctrl_perf_csv, velocity_cmd_csv, acceleration_cmd_csv, autonomous_mode_csv):
    """
    Calculate control key metrics from the control data.
    Returns dict with all metrics.
    """
    metrics = {}
    
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
            
            # Estimate distance = average commanded speed during autonomous window * duration
            avg_speed_auto = 0.0
            if os.path.exists(velocity_cmd_csv):
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
        print("[INFO] Control metrics: max_cross_track_error={:.3f}m, max_velocity={:.2f}m/s, max_acceleration={:.2f}m/s^2".format(
            metrics['max_cross_track_error'], metrics['max_velocity_commanded'], metrics['max_acceleration_commanded']))
        print("[INFO] Autonomous mode: duration={:.1f}s, distance={:.1f}m".format(
            autonomous_duration, autonomous_distance))
    
    return metrics

# ====== STEP 8: WRITE CONTROL KEY METRICS CSV ========================
def step8_write_control_metrics_csv(metrics, bag_basename, bag_out_dir):
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
        w.writerow(['max_acceleration_commanded', metrics['max_acceleration_commanded']])
        w.writerow(['max_deceleration_commanded', metrics['max_deceleration_commanded']])
    
    print("[OK] Control key metrics CSV -> {}".format(out_csv))
    return out_csv

# ====== STEP 9: CREATE CONTROL PLOTS ==================================
def step9_create_control_plots(lat_ctrl_perf_csv, velocity_cmd_csv, acceleration_cmd_csv, bag_basename, bag_out_dir):
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
            
            # Calculate control metrics
            metrics = step7_calculate_control_metrics(lat_ctrl_perf_csv, velocity_cmd_csv, acceleration_cmd_csv, autonomous_mode_csv)
            
            # Write metrics and create plots
            step8_write_control_metrics_csv(metrics, bag_basename, bag_out_dir)
            step9_create_control_plots(lat_ctrl_perf_csv, velocity_cmd_csv, acceleration_cmd_csv, bag_basename, bag_out_dir)
        
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
