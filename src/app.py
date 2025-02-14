from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.ppo_torch import Agent

import json
import random

from utils.model import predict_from_image
from utils.utils import process_drawing
from utils.labels import labels

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize agent with the same parameters as used in training.
n_actions = 5  # 5 difficulty levels: 0 (Very Easy) to 4 (Very Hard)
input_dims = (2,)  # State: [confidence, detection_time]
alpha = 0.0003
n_epochs = 10
batch_size = 64

agent = Agent(
    n_actions=n_actions,
    input_dims=input_dims,
    alpha=alpha,
    n_epochs=n_epochs,
    batch_size=batch_size,
)

# Load the trained model checkpoints.
agent.load_models()


@app.route("/update_proficiency", methods=["POST"])
def predict():
    data = request.json
    try:
        confidence = float(data["confidence"])
        detection_time = float(data["time"])
    except (KeyError, ValueError):
        return json.dumps(
            {"error": "Please provide valid 'confidence' and 'time' fields."}
        ), 400

    state = [confidence, detection_time]

    action, prob, value = agent.choose_action(state)

    return json.dumps({"action": action, "probability": prob, "value": value})


@app.route("/recommend_label", methods=["GET"])
def recommend_label():
    recommended_label = random.choice(labels)
    return json.dumps({"recommended_label": recommended_label["label"]})


@app.route("/predict_drawing", methods=["POST"])
def predict_drawing():
    try:
        file = request.files["file"]
        prompt = request.form.get("prompt")

        image_array = process_drawing(file)

        recommended_label, confidence, preds = predict_from_image(image_array)

        if prompt == recommended_label["label"]:
            return json.dumps(
                {
                    "correct_label": recommended_label["label"],
                    "success": True,
                    "confidence": str(confidence),
                }
            )
        else:
            index = next((i for i, d in enumerate(labels) if d["label"] == prompt), -1)
            confidence_of_right_prompt = preds[0][index]

            return json.dumps(
                {
                    "correct_label": recommended_label["label"],
                    "success": False,
                    "confidence": str(confidence_of_right_prompt),
                }
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
