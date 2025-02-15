from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.ppo_torch import Agent,DQNAgent
import numpy as np

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

# agent = Agent(
#     n_actions=n_actions,
#     input_dims=input_dims,
#     alpha=alpha,
#     n_epochs=n_epochs,
#     batch_size=batch_size,
# )


# Make sure your environment is defined somewhere:
class DrawingAdaptationEnv:
    def __init__(self):
        self.current_level = 2
        self.action_space = type("ActionSpace", (), {"n": 3})
        self.observation_space = type("ObsSpace", (), {"shape": (3,), "low": np.array([0, 0.0, 0.0]), "high": np.array([9, 1.0, 50.0])})
        
    def step(self, action):
        confidence = np.random.uniform(0, 1)
        detection_time = np.random.uniform(0, 50)
        if action == 0:
            self.current_level = max(0, self.current_level - 1)
        elif action == 2:
            self.current_level = min(9, self.current_level + 1)
        target_confidence = 0.7
        target_time = 20.0
        reward = -0.2 * abs(target_time - detection_time) - 0.8 * abs(target_confidence - confidence)
        if confidence > 0.7 and detection_time < 30 and action == 2:
            reward = 10
        elif confidence < 0.5 and detection_time > 20 and action == 0:
            reward = 10
        done = False
        state = np.array([self.current_level, confidence, detection_time], dtype=np.float32)
        return state, reward, done, {}
    
    def reset(self):
        self.current_level = 2
        confidence = np.random.uniform(0, 1)
        detection_time = np.random.uniform(0, 10)
        return np.array([self.current_level, confidence, detection_time], dtype=np.float32)


env = DrawingAdaptationEnv()
state_size = env.observation_space.shape[0]  
action_size = env.action_space.n 
agent = DQNAgent(state_size, action_size)

# Load model
agent.load_model("/src/utils/tmp/model/dqn_agent_model.h5")

@app.route("/update_proficiency", methods=["POST"])
def predict():
    data = request.json
    try:
        current_level = float(data["current_level"])
        confidence = float(data["confidence"])
        detection_time = float(data["time"])
    except (KeyError, ValueError):
        return json.dumps(
            {"error": "Please provide valid 'confidence' and 'time' fields."}
        ), 400

    state = [current_level,confidence, detection_time]

    action = agent.act(state)

    return json.dumps({"action": action})


@app.route("/update_proficiency_for_multiplayer", methods=["POST"])
def predict_proficiency_for_multiplayer():
    data = request.json
    try:
        detection_time = float(data["time"])
        currPoints = int(data["currPoint"])

    except (KeyError, ValueError):
        return json.dumps(
            {"error": "Please provide valid 'confidence' and 'time' fields."}
        ), 400

    if detection_time < 10:
        action = 2
    elif detection_time < 30:
        action = 1
    else:
        action = 0

    return json.dumps({"action": currPoints + action})


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
