#!/usr/bin/env python3
"""Small ROS 2 bag compatibility helpers for the AVA analysis pipelines."""

import os

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class BagTime(object):
    """ROS1-like timestamp wrapper around ROS 2 bag nanoseconds."""

    def __init__(self, nanoseconds):
        self.nanoseconds = int(nanoseconds)

    def to_sec(self):
        return self.nanoseconds / 1e9


def stamp_to_sec(stamp, fallback=None):
    """Return seconds from ROS 2 builtin_interfaces/Time or a BagTime fallback."""
    if stamp is not None:
        sec = getattr(stamp, "sec", None)
        nanosec = getattr(stamp, "nanosec", None)
        if sec is not None and nanosec is not None and (sec != 0 or nanosec != 0):
            return float(sec) + float(nanosec) / 1e9
    if fallback is not None:
        return fallback.to_sec() if hasattr(fallback, "to_sec") else float(fallback)
    return 0.0


def normalize_topic(topic):
    if not topic:
        return topic
    return topic if topic.startswith("/") else "/" + topic


def infer_storage_id(uri):
    """Infer the rosbag2 storage id from metadata or bag file extension."""
    metadata = uri if os.path.basename(uri) == "metadata.yaml" else os.path.join(uri, "metadata.yaml")
    if os.path.exists(metadata):
        with open(metadata, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("storage_identifier:"):
                    return stripped.split(":", 1)[1].strip() or "sqlite3"

    if uri.endswith(".mcap"):
        return "mcap"
    return "sqlite3"


class TopicInfo(object):
    def __init__(self, msg_type, message_count):
        self.msg_type = msg_type
        self.message_count = message_count


class TypeAndTopicInfo(object):
    def __init__(self, topics):
        self.topics = topics


class Ros2Bag(object):
    """Minimal rosbag.Bag-style reader backed by rosbag2_py."""

    def __init__(self, uri):
        self.uri = uri
        self.storage_id = infer_storage_id(uri)
        self._topic_types = None
        self._topic_counts = None

    def __enter__(self):
        self._load_topic_metadata()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _open_reader(self):
        storage_options = rosbag2_py.StorageOptions(uri=self.uri, storage_id=self.storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        return reader

    def _load_topic_metadata(self):
        if self._topic_types is not None:
            return

        reader = self._open_reader()
        self._topic_types = {
            topic_metadata.name: topic_metadata.type
            for topic_metadata in reader.get_all_topics_and_types()
        }
        self._topic_counts = dict((topic, 0) for topic in self._topic_types)
        while reader.has_next():
            topic, _, _ = reader.read_next()
            self._topic_counts[topic] = self._topic_counts.get(topic, 0) + 1

    def get_type_and_topic_info(self):
        self._load_topic_metadata()
        return TypeAndTopicInfo(
            dict(
                (topic, TopicInfo(msg_type, self._topic_counts.get(topic, 0)))
                for topic, msg_type in self._topic_types.items()
            )
        )

    def read_messages(self, topics=None):
        self._load_topic_metadata()
        selected = None
        if topics is not None:
            selected = set()
            available_by_normalized = dict(
                (normalize_topic(topic), topic) for topic in self._topic_types
            )
            for topic in topics:
                actual = available_by_normalized.get(normalize_topic(topic), topic)
                selected.add(actual)

        reader = self._open_reader()
        msg_classes = {}
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            if selected is not None and topic not in selected:
                continue

            msg_type = self._topic_types.get(topic)
            if msg_type is None:
                continue
            if msg_type not in msg_classes:
                msg_classes[msg_type] = get_message(msg_type)

            yield topic, deserialize_message(data, msg_classes[msg_type]), BagTime(timestamp_ns)


def _rosbags_deps_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps")


def read_messages_rosbags(uri, topics=None):
    """
    Yield (topic, msg, BagTime) using rosbags and schemas embedded in the bag.
    Used when installed ROS types do not match the recorded yolo_msgs layout.
    """
    import sys
    from pathlib import Path

    deps = _rosbags_deps_path()
    if not os.path.isdir(deps):
        raise ImportError(
            "rosbags not found in {}; install with: "
            "python3 -m pip install --target=.deps 'numpy<2' rosbags".format(deps)
        )

    saved_path = sys.path[:]
    try:
        if deps not in sys.path:
            sys.path.insert(0, deps)
        from rosbags.highlevel import AnyReader

        selected = None
        if topics is not None:
            selected = set()
            for topic in topics:
                selected.add(topic)
                selected.add(normalize_topic(topic))

        with AnyReader([Path(uri)]) as reader:
            connections = reader.connections
            if selected is not None:
                connections = [
                    conn for conn in connections
                    if conn.topic in selected or normalize_topic(conn.topic) in selected
                ]
            for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, conn.msgtype)
                yield conn.topic, msg, BagTime(timestamp_ns)
    finally:
        sys.path[:] = saved_path


def iter_topic_messages(bag, topic):
    """
    Read messages for one topic, using rosbags for yolo_msgs when available.
    """
    bag._load_topic_metadata()
    msg_type = bag._topic_types.get(topic) or bag._topic_types.get(normalize_topic(topic))
    if msg_type and msg_type.startswith("yolo_msgs/"):
        try:
            yield from read_messages_rosbags(bag.uri, topics=[topic])
            return
        except ImportError as exc:
            print("[WARN] rosbags unavailable ({}); falling back to rclpy deserialize".format(exc))
    yield from bag.read_messages(topics=[topic])
