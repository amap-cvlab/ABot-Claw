import base64
import requests
import cv2
import json

URL = "http://localhost:8017/detect"

def test_detect():
    # Create a dummy image or read one (here we use a dummy one for simple testing)
    img = cv2.imread("data/images/bus.jpg")
    if img is None:
        print("Test image not found, skipping visual test.")
        return

    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "image": img_b64,
        "conf_thres": 0.25,
        "iou_thres": 0.45
    }

    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        res_json = response.json()
        print(f"Detected {res_json['count']} objects:")
        print(json.dumps(res_json, indent=2))
    except Exception as e:
        print("API request failed:", e)

if __name__ == "__main__":
    test_detect()
