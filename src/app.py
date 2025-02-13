from flask import Flask, request, send_file
from io import BytesIO
from flask_cors import CORS

import base64
import json
import random

from src.utils import labels

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route("/recommend_label", methods=["GET"])
def recommend_label():
    recommended_label = random.choice(labels)
    return json.dumps({"recommended_label": recommended_label})


@app.route("/display_image", methods=["POST"])
def display_image():
    data = request.get_json()
    image_data = base64.b64decode(data["image"])
    image = BytesIO(image_data)

    print(image)
    return send_file(image, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True)
