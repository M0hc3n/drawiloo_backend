import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


def process_drawing(file):
    image_pil = Image.open(io.BytesIO(file.read()))

    image_pil = image_pil.convert("L")  # Keep this if model expects grayscale

    for size in [500, 200, 64]:
        image_pil = image_pil.resize((size, size), Image.LANCZOS)

    image_array = np.array(image_pil) / 255.0  # Normalize to [0,1]

    image_array = np.expand_dims(image_array, axis=[0, -1])

    return image_array
