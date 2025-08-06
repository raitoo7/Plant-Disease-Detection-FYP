import os
from keras.models import load_model

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resmodel.h5")

if os.path.exists(model_path):
    loaded_model = load_model(model_path)
    print("Model loaded successfully!")
else:
    loaded_model = None
    print(f"Model file '{model_path}' not found. The app cannot run predictions without it.")
    # Optional: raise error or disable prediction functionality gracefully

from flask import Flask, render_template, request, jsonify
import os
from keras.models import load_model

# Get the absolute directory where app.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the model
model_path = os.path.join(script_dir, "resmodel.h5")

# Load the model
loaded_model = load_model(model_path)
print("Model loaded successfully!")

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "resmodel.h5")
loaded_model = load_model(model_path)


app = Flask(__name__)

IMG_WIDTH = 224
IMG_HEIGHT = 224

model_path = "./resmodel.h5"
loaded_model = load_model(model_path)

labels = {
    0: 'Corn___Common_rust',
    1: 'Corn___Northern_Leaf_Blight',
    2: 'Corn___healthy',
    3: 'Not_a_Leaf',
    4: 'Potato___Early_blight',
    5: 'Potato___Late_blight',
    6: 'Potato___healthy',
    7: 'Strawberry___Leaf_scorch',
    8: 'Strawberry___healthy',
}

# Dictionary to store solution messages for each class
solution_messages = {
    'Corn___Common_rust': 'Utilize resistant hybrids, fungicides, rotation, and residue destruction',
    'Corn___Northern_Leaf_Blight': 'Opt for resistant hybrids, rotate crops, and use fungicides',
    'Corn___healthy': 'No action needed. Plant is healthy',
    'Not_a_Leaf': 'Please Provide the  Plant Image',
    'Potato___Early_blight': 'Choose resistant varieties, avoid overhead irrigation, rotate crops, and balance nutrients',
    'Potato___Late_blight': 'Choose resistant varieties, avoid overhead watering,dispose of plant debris away from growing areas',
    'Potato___healthy': 'No action needed. Plant is healthy.',
    'Strawberry___Leaf_scorch': 'Use resistant varieties, remove infected leaves, ensure proper air circulation, and avoid overhead irrigation.',
    'Strawberry___healthy': 'No action needed. Plant is healthy.'
}
reason_messages = {
    'Corn___Common_rust': 'Fungus Puccinia sorghi',
    'Corn___Northern_Leaf_Blight': 'Fungus Exserohilum turcicum.',
    'Corn___healthy': 'No issue, It is healthy',
    'Not_a_Leaf': 'Please Provide the  Plant Image.',
    'Potato___Early_blight': 'Fungus Alternaria solani.',
    'Potato___Late_blight': 'Oomycete pathogen Phytophthora infestans.',
    'Potato___healthy': 'No issue, It is healthy',
    'Strawberry___Leaf_scorch': 'Fungus Diplocarpon earliana.',
    'Strawberry___healthy': 'No issue, It is healthy',
}

# Function to extract reason message based on predicted class name
def get_reason_message(predicted_class_name):
    return reason_messages.get(predicted_class_name, None)

# Function to extract solution message based on predicted class name
def get_solution_message(predicted_class_name):
    return solution_messages.get(predicted_class_name, None)

current_working_directory = os.getcwd()
temp_dir = os.path.join(current_working_directory, 'temp_images')

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
@app.route('/', methods=['GET', 'POST'])
def aboutus():
    return render_template('about.html')  # This will show the About Us page by default.

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['img_file']
        if file:
            try:
                img_url = save_and_process_image(file)
                result = process_image(img_url)
                return render_template('index.html', result=result, img_url=img_url)
            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                print(f"Error: {error_message}")
                return render_template('index.html', result=None, img_url=None, error_message=error_message)

    return render_template('index.html', result=None, img_url=None)

def save_and_process_image(file):
    img_path = os.path.join(temp_dir, f"temp_{file.filename}")
    file.save(img_path)
    return img_path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['img_file']
        if file:
            img_url = save_and_process_image(file)
            result = process_image(img_url)
            return jsonify(result)
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(f"Error: {error_message}")
        return jsonify({'error': error_message})

    return jsonify({'error': 'Invalid request'})




def process_image(img_url):
    try:
        specific_image = image.load_img(img_url, target_size=(IMG_WIDTH, IMG_HEIGHT))
        specific_image_array = image.img_to_array(specific_image)
        specific_image_array = np.expand_dims(specific_image_array, axis=0)
        specific_image_array = specific_image_array.astype("float32") / 255.0  # Normalize

        prediction = loaded_model.predict(specific_image_array)

        if prediction is None or len(prediction) == 0 or np.isnan(prediction).any():
            return {'error': 'Model returned an invalid prediction'}

        # Get the top prediction
        predicted_class_index = int(np.argmax(prediction))
        confidence_score = float(np.max(prediction))

        # Prevent NaN confidence values
        if np.isnan(confidence_score) or confidence_score <= 0.0:
            confidence_score = 0.01  # Set a small default value to avoid NaN%

        predicted_class_name = labels.get(predicted_class_index, "Unknown Class")

        reason_message = get_reason_message(predicted_class_name)
        solution_message = get_solution_message(predicted_class_name)

        #  Show Second-Best Prediction ONLY if Confidence < 75%
        second_best_prediction = ""
        second_best_confidence = ""
        if confidence_score < 0.77:
            sorted_indices = np.argsort(prediction[0])[-2:]  # Get top 2 class indices
            top_2_classes = [labels[idx] for idx in reversed(sorted_indices)]
            top_2_scores = [float(prediction[0][idx]) for idx in reversed(sorted_indices)]

            second_best_prediction = top_2_classes[1]
            second_best_confidence = f"{top_2_scores[1] * 100:.2f}%"

        confidence_warning = ""
        if confidence_score < 0.77:
            confidence_warning = "⚠️ Model is less confident about this prediction.Please verify."

        result = {
            'predicted_class_name': predicted_class_name,
            'confidence_score': f"{confidence_score * 100:.2f}%",
            'confidence_warning': confidence_warning,
            'reason_message': reason_message,
            'solution_message': solution_message,
        }

        #  Add second-best prediction ONLY if confidence <75%
        if confidence_score < 0.77:
            result['second_best_prediction'] = second_best_prediction
            result['second_best_confidence'] = second_best_confidence

        print("Final JSON response:", result)  # Debugging

        return result

   
    

   
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(f"Error: {error_message}")
        return {'error': error_message}

if __name__ == '__main__':
    app.run(debug=True)
