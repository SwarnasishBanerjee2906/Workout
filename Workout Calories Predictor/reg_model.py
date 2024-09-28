import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Upload the dataset
workout_calories = pd.read_excel(r'C:/Users/SWARNASASH/Downloads/Desktop Material/Workout_Calory_Predictor/Workout_calories.xlsx')

# Assuming 'Gender' is a categorical variable that needs encoding
label_encoder = LabelEncoder()
workout_calories['Gender'] = label_encoder.fit_transform(workout_calories['Gender'])

# Splitting the data into features and target
X = workout_calories.drop(columns=["User_ID","Calories"], axis=1)
y = workout_calories['Calories']
# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training the SVR model
svr = SVR(C=20,kernel='rbf')
svr.fit(X_train, y_train)

# Saving the model and the scaler
joblib.dump(svr, 'svr_model.pkl')
joblib.dump(scaler,'scaler_model.pkl')
joblib.dump(label_encoder,'label_encoder.pkl')

