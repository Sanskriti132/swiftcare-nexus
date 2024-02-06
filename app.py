from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model4")

@app.route('/')
def home():
    return render_template("D:\\pagal\\after.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()
    
    # Extract features
    age = float(data['age'])
    gender = float(data['gender'])
    
    # Make prediction
    prediction = model.predict([[age, gender]])
    
    # Convert prediction to human-readable form
    if prediction[0] == 0:
        prediction_label = "No Liver Disease"
    else:
        prediction_label = "Liver Disease Detected"
    
    # Return the prediction as JSON
    return jsonify({'prediction': prediction_label})

@app.route('/model_output')
def model_output():
    # Perform prediction on some sample data
    sample_age = 45
    sample_gender = 1  # Female
    prediction = model.predict([[sample_age, sample_gender]])
    
    # Convert prediction to human-readable form
    if prediction[0] == 0:
        prediction_label = "No Liver Disease"
    else:
        prediction_label = "Liver Disease Detected"
    
    return render_template('model_output.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
