import argparse
import base64
import json
import sys

import cv2
import numpy as np
import requests
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage

DEFAULT_URL = "http://30.79.84.82:8017/detect"
DEFAULT_IMAGE_TOPIC = "/wrist_camera/color/image_raw"


def ros_image_to_bgr(msg: RosImage) -> np.ndarray:
    bridge = CvBridge()
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception:
        return bridge.imgmsg_to_cv2(msg)


def run_detect(url: str, img_b64: str, conf_thres: float, iou_thres: float) -> None:
    payload = {
        "image": img_b64,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    res_json = response.json()
    print(f"Detected {res_json['count']} objects:")
    print(json.dumps(res_json, indent=2))


def test_detect_from_ros(
    url: str,
    image_topic: str,
    timeout: float,
    conf_thres: float,
    iou_thres: float,
) -> None:
    rospy.init_node("yolo_detect_api_test", anonymous=True)
    try:
        msg = rospy.wait_for_message(image_topic, RosImage, timeout=timeout)
    except rospy.ROSException as e:
        print(f"未在 {timeout}s 内从话题收到图像: {image_topic}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    img = ros_image_to_bgr(msg)
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        print("cv2.imencode 失败", file=sys.stderr)
        sys.exit(1)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    try:
        run_detect(url, img_b64, conf_thres, iou_thres)
    except Exception as e:
        print("API request failed:", e)
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="从 ROS 图像话题取一帧并调用 YOLO 检测 HTTP API")
    p.add_argument("--url", default=DEFAULT_URL, help="检测服务地址")
    p.add_argument(
        "--image-topic",
        default=DEFAULT_IMAGE_TOPIC,
        help="sensor_msgs/Image 彩色图话题",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="等待首帧图像的超时（秒）",
    )
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--iou-thres", type=float, default=0.45)
    args = p.parse_args()

    test_detect_from_ros(
        url=args.url,
        image_topic=args.image_topic,
        timeout=args.timeout,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )


if __name__ == "__main__":
    main()
