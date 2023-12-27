from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the chest.h5 model
model_path = './chest.h5'
model = load_model(model_path)

def chestScanPrediction(img_path, model):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the input image using EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class = np.argmax(predictions)

    # Define the label names corresponding to the classes
    label_names = ['adenocarcinoma', 'large.cell.carcinoma',
                   'normal', 'squamous.cell.carcinoma']

    # Get the predicted label name
    predicted_label = label_names[predicted_class]

    return predicted_label

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['file']

    # Save the file temporarily
    img_path = 'temp.png'
    file.save(img_path)

    # Make prediction
    prediction = chestScanPrediction(img_path, model)

    # Remove the temporary file
    import os
    os.remove(img_path)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)