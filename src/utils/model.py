from tensorflow.keras.applications import MobileNet
import numpy as np
from utils.labels import labels

model = MobileNet(input_shape=(64, 64, 1), alpha=1.0, weights=None, classes=340)
model.load_weights("src/utils/tmp/model/model.h5")  # Change to your file path

def predict_from_image(image_array):
    preds = model.predict(image_array)

    confidence = np.max(preds, axis=-1)[0]
    predicted_class = np.argmax(preds, axis=-1)[0]

    recommended_label = labels[predicted_class]

    return recommended_label, confidence, preds