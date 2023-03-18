from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.DEBUG)

import cv2
import onnxruntime as ort
import numpy as np
from box_utils import (
    predict,
    scale,
    faceDetector,
    allowed_image_file,
    read_image_from_file,
)
import base64
import io

app = Flask(__name__)
CORS(app)

face_detector_onnx = "version-RFB-640.onnx"
face_detector = ort.InferenceSession(face_detector_onnx)


@app.route("/", methods=["GET"])
def index():
    return "Hello World"


@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    if "image" not in request.files:
        return jsonify({"error": "No image file in the request"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No image file provided"}), 400

    if not allowed_image_file(image_file.filename):
        return jsonify({"error": "Invalid image file type"}), 400

    image = read_image_from_file(image_file)
    boxes, labels, probs = faceDetector(image)

    results = []

    for i in range(boxes.shape[0]):
        box = scale(boxes[i, :])
        box = [int(x) for x in box]  # Convert box coordinates to int
        results.append({"box": box})
    return jsonify(results)
