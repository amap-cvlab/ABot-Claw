import base64
import json
from pathlib import Path

import requests


URL = "http://127.0.0.1:8014/critic"


def image_to_b64(image_path: str) -> str:
    raw = Path(image_path).read_bytes()
    return base64.b64encode(raw).decode("utf-8")


def test_critic() -> None:
    current_image = "evo_vlac/examples/images/test/595-139-565-0.jpg"
    reference_image = "evo_vlac/examples/images/ref/599-300-521-0.jpg"
    payload = {
        "image": image_to_b64(current_image),
        "reference_image": image_to_b64(reference_image),
        "task_description": "Scoop the rice into the rice cooker.",
        "batch_num": 1,
        "rich": False,
    }

    resp = requests.post(URL, json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_critic()
