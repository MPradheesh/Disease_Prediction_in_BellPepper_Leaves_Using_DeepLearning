import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename



# Initialize Flask app with the custom templates folder
app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to save uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # Allowed file types

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = load_model('cnn_model.h5')

# Get class labels
class_labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Not Found']  # Replace with actual class names used during training

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Disable caching of templates in development
app.jinja_env.cache = {}

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # `index.html` will be fetched from `custom_template_folder`


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image part is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is allowed and save it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict the image class
        prediction, confidence = model_predict(filepath)

        if confidence < 0.8:  # Check if confidence is below 80%
            return jsonify({
                "message": "Prediction failure: Confidence too low",
                "prediction": prediction,
                "confidence": f"{confidence:.2f}"
            })

        return jsonify({
            "message": "Prediction success",
            "prediction": prediction,
            "confidence": f"{confidence:.2f}"
        })

    return jsonify({"error": "Invalid file type"}), 400

'''
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image part is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is allowed and save it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict the image class
        prediction, confidence = model_predict(filepath)

        if confidence < 0.8:  # Check if confidence is below 80%
            return jsonify({
                "message": "Prediction failure: Confidence too low",
                "confidence": f"{confidence:.2f}"
            })

        return jsonify({
            "message": "Prediction success",
            "prediction": prediction,
            "confidence": f"{confidence:.2f}"
        })

    return jsonify({"error": "Invalid file type"}), 400
'''
# Function to preprocess the image and predict the class
def model_predict(filepath):
    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(150, 150))  # Match the training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image as in training

    # Predict the class
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)  # Confidence score

    return class_labels[class_index[0]], confidence  # Return both class label and confidence

# Disable caching for all responses in development
@app.after_request
def add_header(response):
    # Ensure no caching
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

