import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# ── Model Architecture (matches training exactly) ──────────────────────────
class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.fc_layer(x)
        return x

# ── Load model ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
model = DeeperCNN()
model_path = os.path.join(os.path.dirname(__file__), "Main_Model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

CLASS_NAMES = ["Angry", "Engaged", "Happy", "Neutral"]

# Inference transform — no augmentation, just resize + normalize
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

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
        image = Image.open(io.BytesIO(file.read())).convert("L")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0].tolist()

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
