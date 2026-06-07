import os
import io
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# ── Load ONNX model (lightweight — no PyTorch needed) ─────────────────────
model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

CLASS_NAMES = ["Angry", "Engaged", "Happy", "Neutral"]

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((48, 48))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr[np.newaxis, np.newaxis, :, :]  # (1, 1, 48, 48)
    return arr

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        tensor = preprocess(file.read())
        output = session.run(None, {input_name: tensor})[0][0]
        probs = softmax(output).tolist()
        prediction_idx = probs.index(max(probs))

        return jsonify({
            "prediction": CLASS_NAMES[prediction_idx],
            "confidence": round(probs[prediction_idx] * 100, 1),
            "scores": {
                name: round(prob * 100, 1)
                for name, prob in zip(CLASS_NAMES, probs)
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
