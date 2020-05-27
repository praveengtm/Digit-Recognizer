from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

OUTPUT_DIR = 'Images'

app = Flask(__name__)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    x = np.reshape(img, (1,784))
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        preds_max = np.argmax(preds)
        return render_template("show.html", preds_max=preds_max)
    return render_template("index.html")


if __name__ == '__main__':
    model = load_model('weights.h5')
    app.run(debug=True)
