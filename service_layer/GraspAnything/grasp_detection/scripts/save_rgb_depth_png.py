#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import errno
from typing import Tuple
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


def _mkdir_p(path: str) -> None:
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class RgbDepthSaver:
    def __init__(self):
        self.bridge = CvBridge()

        self.rgb_topic = rospy.get_param("~rgb_topic", "/wrist_camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/wrist_camera/aligned_depth_to_color/image_raw")
        self.output_dir = rospy.get_param("~output_dir", "./example_data")
        self.prefix = rospy.get_param("~prefix", "frame")
        self.save_once = bool(rospy.get_param("~save_once", True))
        self.queue_size = int(rospy.get_param("~queue_size", 30))
        self.slop = float(rospy.get_param("~slop", 0.03))
        self.depth_float_scale_m = float(rospy.get_param("~depth_float_scale_m", 0.001))

        _mkdir_p(self.output_dir)
        self.counter = 0
        self.saved = False

        rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        ats = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=self.queue_size,
            slop=self.slop,
            allow_headerless=False,
        )
        ats.registerCallback(self.cb)

        rospy.loginfo("Saving synchronized RGB+Depth PNGs")
        rospy.loginfo("  rgb_topic:   %s", self.rgb_topic)
        rospy.loginfo("  depth_topic: %s", self.depth_topic)
        rospy.loginfo("  output_dir:  %s", os.path.abspath(self.output_dir))
        rospy.loginfo("  save_once:   %s", self.save_once)

    def _save_paths(self, stamp) -> Tuple[str, str]:
        t = stamp.to_sec()
        tag = f"{self.prefix}_{self.counter:06d}_{t:.6f}"
        rgb_path = os.path.join(self.output_dir, f"{tag}_color.png")
        depth_path = os.path.join(self.output_dir, f"{tag}_depth.png")
        return rgb_path, depth_path

    def _convert_rgb(self, msg: Image) -> np.ndarray:
        # Prefer exact encoding when provided; fallback to "bgr8" which is OpenCV-native.
        enc = msg.encoding.lower() if msg.encoding else ""
        if enc in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8"):
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        else:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if cv.ndim == 2:
            cv = cv2.cvtColor(cv, cv2.COLOR_GRAY2BGR)
        elif cv.shape[2] == 4:
            cv = cv2.cvtColor(cv, cv2.COLOR_BGRA2BGR)

        # Save as standard 3-channel BGR PNG (viewable everywhere).
        return cv

    def _convert_depth_to_u16(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower() if msg.encoding else ""

        if enc in ("16uc1", "mono16"):
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16, copy=False)
            return depth

        if enc in ("32fc1",):
            depth_f = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32, copy=False)
            # depth_float_scale_m: meters per unit in the float image; default assumes meters.
            # Convert to millimeters for 16-bit PNG.
            depth_m = depth_f * (1.0 if self.depth_float_scale_m == 0 else self.depth_float_scale_m / 0.001)
            depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
            return depth_mm

        # Fallback: try passthrough and coerce.
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth.dtype == np.uint16:
            return depth
        if depth.dtype == np.float32 or depth.dtype == np.float64:
            depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
            return depth_mm
        return depth.astype(np.uint16)

    def cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self.save_once and self.saved:
            return

        try:
            rgb = self._convert_rgb(rgb_msg)
            depth_u16 = self._convert_depth_to_u16(depth_msg)
        except Exception as e:
            rospy.logerr("Conversion failed: %s", str(e))
            return

        rgb_path, depth_path = self._save_paths(rgb_msg.header.stamp)

        ok1 = cv2.imwrite(rgb_path, rgb)
        ok2 = cv2.imwrite(depth_path, depth_u16)
        if not ok1 or not ok2:
            rospy.logerr("Failed to write png(s): rgb_ok=%s depth_ok=%s", ok1, ok2)
            return

        rospy.loginfo("Saved: %s", os.path.basename(rgb_path))
        rospy.loginfo("Saved: %s", os.path.basename(depth_path))
        self.counter += 1
        if self.save_once:
            self.saved = True
            rospy.loginfo("save_once=true, shutting down.")
            rospy.signal_shutdown("Saved one synchronized pair")


def main():
    rospy.init_node("save_rgb_depth_png", anonymous=False)
    RgbDepthSaver()
    rospy.spin()


if __name__ == "__main__":
    main()

