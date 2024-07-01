convolutional neural network for image classification 

step 1 : Run trainingmini.ipynb 

step 2 : execute app.py 

NOTE: change the trainig_set and validation_set directory path in trainingmini.ipynb file 

![Screenshot 2024-07-01 132727](https://github.com/DEVS-CODE-GLITCH/Tomato-plant-disease-detection-using-Machine-learning/assets/174253524/d1ee1724-fe86-4b59-80c0-59963081cf0d)

-> The provided Jupyter notebook appears to focus on training and evaluating a convolutional neural network (CNN) for image classification. Below are the key algorithms and techniques used in the code:

1. **Data Loading and Preparation**:
   - `tf.keras.utils.image_dataset_from_directory`: Loads image data from directories for training, validation, and testing.

2. **Data Augmentation**:
   - `tf.keras.Sequential` with layers such as `tf.keras.layers.RandomFlip`, `tf.keras.layers.RandomRotation`, `tf.keras.layers.RandomZoom`, and `tf.keras.layers.RandomTranslation`: Applies random transformations to the training images to augment the dataset.

3. **Convolutional Neural Network (CNN)**:
   - `tf.keras.models.Sequential`: Defines a sequential CNN model.
   - `tf.keras.layers.Conv2D`: Adds convolutional layers with ReLU activation.
   - `tf.keras.layers.MaxPool2D`: Adds max pooling layers to downsample the feature maps.
   - `tf.keras.layers.Flatten`: Flattens the input.
   - `tf.keras.layers.Dense`: Adds fully connected (dense) layers.
   - `tf.keras.layers.Dropout`: Adds dropout layers to prevent overfitting.
   - `tf.keras.layers.Softmax`: Adds a softmax activation function for the output layer.

4. **Model Compilation and Training**:
   - `model.compile`: Configures the model for training with loss function (`categorical_crossentropy`), optimizer (`adam`), and metrics (`accuracy`).
   - `model.fit`: Trains the model on the training dataset and evaluates on the validation dataset.

5. **Model Evaluation**:
   - `model.evaluate`: Evaluates the trained model on the test dataset.
   - `model.predict`: Makes predictions on the test dataset.
   - `sklearn.metrics.confusion_matrix` and `classification_report`: Computes and displays the confusion matrix and classification report.

6. **Visualization**:
   - `matplotlib.pyplot`: Visualizes the training history (accuracy and loss) and the confusion matrix.
   - `seaborn.heatmap`: Plots the confusion matrix as a heatmap.
  

-> The `app.py` file is a Flask web application that loads a trained model to predict tomato diseases from uploaded images. Here are the key components and algorithms used in the code:

1. **Flask Web Framework**:
   - `Flask`: Used to create a web application.
   - `request`, `render_template`: Used to handle HTTP requests and render HTML templates.
   - `secure_filename`: Used to securely save uploaded files.

2. **TensorFlow and Keras**:
   - `tf.keras.models.load_model`: Loads a pre-trained model from a file.
   - `image.load_img`, `image.img_to_array`: Used to preprocess the input images.

3. **Prediction Logic**:
   - **Image Preprocessing**:
     - `image.load_img(img_path, target_size=(128, 128))`: Loads an image and resizes it to 128x128 pixels.
     - `image.img_to_array(img)`: Converts the image to a NumPy array.
     - `np.expand_dims(img_array, axis=0)`: Expands the dimensions of the array to match the input shape expected by the model.
   - **Model Prediction**:
     - `model.predict(img_array)`: Uses the loaded model to predict the disease from the preprocessed image.
     - `np.argmax(preds)`: Determines the class with the highest predicted probability.
   - **Class Mapping**:
     - Maps the predicted class index to the corresponding disease name using `class_names`.

4. **Flask Routes**:
   - `@app.route('/', methods=['GET'])`: Defines the route for the homepage.
   - `@app.route('/predict', methods=['POST'])`: Defines the route to handle file uploads and make predictions.

5. **Utility Functions**:
   - `predict_disease(img_path, model)`: A helper function that preprocesses the image, makes a prediction, and returns the predicted disease name.

In summary, the primary algorithms used in the code are image preprocessing, model inference using a convolutional neural network (CNN), and mapping predictions to class names. The Flask framework is utilized to create a web interface for uploading images and displaying predictions.

