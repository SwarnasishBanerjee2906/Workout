from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('svr_model.pkl')
scaler = joblib.load('scaler_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    gender = data['Gender']
    age = int(data['Age'])
    height = float(data['Height'])
    weight = float(data['Weight'])
    duration = int(data['Duration'])
    heart_rate = float(data['Heart_Rate'])
    body_temp = float(data['Body_Temp'])
    
    # Encode the gender
    gender_encoded = label_encoder.transform([gender])[0]
    
    # Features
    features = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make a prediction
    prediction = model.predict(features_scaled)
    
    return render_template('result.html', prediction=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
