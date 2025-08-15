from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\HP\VS Code Projects\Machine Learning\KNN-Breast-Cancer-prediction\KNN-Breast_Cancer_Model.pkl')
scaler = joblib.load(r'C:\Users\HP\VS Code Projects\Machine Learning\KNN-Breast-Cancer-prediction\KNN-Breast_Cancer_Scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form and convert to float
        features = [float(request.form[f]) for f in request.form]
        input_data = np.array([features])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = 'Malignant' if prediction == 4 else 'Benign'

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
