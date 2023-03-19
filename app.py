from flask import Flask, request, jsonify
from flask_cors import CORS
from flask.helpers import send_file
import logging
from PIL import Image

logging.basicConfig(level=logging.DEBUG)

import cv2
import onnxruntime as ort
import numpy as np
from box_utils import (
    predict,
    faceDetector,
    allowed_image_file,
    read_image_from_file,
)
import base64
import io

app = Flask(__name__, static_url_path="/", static_folder="web")
CORS(app)

face_detector_onnx = "version-RFB-640.onnx"
face_detector = ort.InferenceSession(face_detector_onnx)


def base64_to_image(base64_data):
    try:
        img_data = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception as e:
        print(f"Error: {e}")
        return None


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_byte_data = buffered.getvalue()
    return base64.b64encode(img_byte_data).decode("ascii")


def process_image(image):
    image_np = np.array(image)
    boxes, _, _ = faceDetector(image_np)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return Image.fromarray(image_np)


@app.route("/", methods=["GET"])
def index():
    return send_file("web/index.html")


@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    data = request.get_json()

    if "image" not in data:
        return jsonify({"error": "Missing image data"}), 400

    image_data = data["image"]
    image = base64_to_image(image_data)

    if image is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Process the image and detect faces
    boxes, _, _ = faceDetector(np.array(image))
    print(f"Boxes detected: {boxes}")

    detected_image = process_image(image)

    # Convert the processed image back to base64
    output_image_data = image_to_base64(detected_image)

    return jsonify({"image": output_image_data})
