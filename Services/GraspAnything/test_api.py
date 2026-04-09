import argparse
import base64
import json
from pathlib import Path

import requests

DEFAULT_URL = "http://127.0.0.1:8015/grasp/detect"
DEFAULT_K = "[[600.0,0.0,320.0],[0.0,600.0,240.0],[0.0,0.0,1.0]]"


def _encode_file_as_base64(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def run_detect(
    url: str,
    color_path: str,
    depth_path: str,
    camera_k_json: str,
    object_name: str,
    top_k: int,
    timeout: float,
) -> None:
    payload = {
        "color_image": _encode_file_as_base64(color_path),
        "depth_image": _encode_file_as_base64(depth_path),
        "camera_intrinsics": json.loads(camera_k_json),
        "object_name": object_name,
        "top_k": top_k,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call GraspAnything /grasp/detect with base64 RGB and depth images"
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Grasp service endpoint URL")
    parser.add_argument("--color", required=True, help="Path to color image (png/jpg)")
    parser.add_argument("--depth", required=True, help="Path to depth image (uint16 png recommended)")
    parser.add_argument("--camera-k", default=DEFAULT_K, help="3x3 camera intrinsics JSON string")
    parser.add_argument("--object-name", default="cup", help="Target object class name")
    parser.add_argument("--top-k", type=int, default=5, help="Max grasp candidates")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    run_detect(
        url=args.url,
        color_path=args.color,
        depth_path=args.depth,
        camera_k_json=args.camera_k,
        object_name=args.object_name,
        top_k=args.top_k,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
