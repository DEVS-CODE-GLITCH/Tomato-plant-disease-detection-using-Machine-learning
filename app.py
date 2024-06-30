import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Load your trained model
MODEL_PATH = "trained_tomato_disease_updated_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Septoria Leaf Spot', 'Yellow Leaf Curl Virus']

def predict_disease(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    print(preds)
    predicted_class = np.argmax(preds)
    print(predicted_class)
    result = class_names[predicted_class]

    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_disease(file_path, model)
        return preds

if __name__ == '__main__':
    app.run(port=5001, debug=True)
