from flask import Flask, request, send_file, jsonify
from io import BytesIO
from flask_cors import CORS
from utils.ppo_torch import Agent
from utils.utils import plot_learning_curve

import base64
import json
import random

from utils.labels import labels
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize agent with the same parameters as used in training.
n_actions = 5              # 5 difficulty levels: 0 (Very Easy) to 4 (Very Hard)
input_dims = (2,)          # State: [confidence, detection_time]
alpha = 0.0003
n_epochs = 10
batch_size = 64

agent = Agent(n_actions=n_actions, input_dims=input_dims, alpha=alpha,
              n_epochs=n_epochs, batch_size=batch_size)

# Load the trained model checkpoints.
agent.load_models()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        confidence = float(data['confidence'])
        detection_time = float(data['time'])
    except (KeyError, ValueError):
        return jsonify({"error": "Please provide valid 'confidence' and 'time' fields."}), 400

    # Create the state vector as expected by the agent.
    state = [confidence, detection_time]

    # Get the action from the agent.
    action, prob, value = agent.choose_action(state)

    return jsonify({
        "action": action,
        "probability": prob,
        "value": value
    })

@app.route('/predict_query', methods=['GET'])
def predict_query():
    """
    Use query parameters (e.g., ?confidence=0.8&time=1.2) to pass data from the browser.
    """
    try:
        confidence = float(request.args.get('confidence'))
        detection_time = float(request.args.get('time'))
    except (TypeError, ValueError):
        return jsonify({"error": "Please provide valid 'confidence' and 'time' query parameters."}), 400

    # Create the state vector as expected by the agent.
    state = [confidence, detection_time]

    # Get the action from the agent.
    action, prob, value = agent.choose_action(state)

    return jsonify({
        "action": action,
        "probability": prob,
        "value": value
    })

@app.route("/recommend_label", methods=["GET"])
def recommend_label():
    recommended_label = random.choice(labels)
    return json.dumps({"recommended_label": recommended_label})

@app.route("/display_image", methods=["POST"])
def display_image():
    data = request.get_json()
    image_data = base64.b64decode(data["image"])
    image = BytesIO(image_data)
    image = Image.open(image)
    image = image.resize((255, 255))
    image = np.array(image)

    print(image)

    recommended_label = random.choice(labels)
    confidence = random.uniform(0.6, 1.0)
    return json.dumps(
        {"correct_label": recommended_label, "success": True, "confidence": confidence}
    )

if __name__ == "__main__":
    app.run(debug=True)
