from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('bagging_regressor_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')  # Your 'index.html' form already exists

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form (ensure form fields correspond to your feature names)
        features = [float(x) for x in request.form.values()]
        features_array = np.array([features])

        # Scale the input
        features_scaled = scaler.transform(features_array)

        # Predict using the trained model
        prediction = model.predict(features_scaled)

        # Return result to the page
        return render_template('index.html', prediction_text=f'Predicted Price: ${prediction[0]:.2f}')

    except Exception as e:
        # Return an error message if something goes wrong
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

