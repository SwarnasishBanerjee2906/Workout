# **Workout Activity Recognition and Calorie Prediction System**

This project integrates two key components: a machine learning-based **Workout Calories Prediction** and a deep learning-based **Workout Image Classification**. The solution is further deployed as a **Flask web application**, allowing users to predict calories burned during workouts based on input parameters and classified workout types.

## **1. Workout Calories Prediction**

**Objective**:  
Predict the calories burned by a workout trainee based on various physiological parameters including:
- Gender
- Height
- Weight
- Workout Duration
- Heart Rate
- Body Temperature

**Approach**:  
This is a **regression problem**, where multiple regression models are used to predict the calories burned. The dataset is preprocessed to handle outliers, and the most influential features (Duration, Heart Rate, Body Temperature) are identified to enhance the accuracy of the model.

**Model**:  
A Support Vector Regressor (SVR) with an optimized hyperparameter (C=20) is implemented, achieving:
- **Accuracy**: 99.98%
- **MAE**: 0.407
- **MSE**: 0.596

The model has been deployed via a Flask application to provide real-time calorie predictions based on user inputs.

## **2. Workout Image Classification**

**Objective**:  
Classify different types of workout images from a dataset containing 22 distinct workout types:
- ['barbell biceps curl', 'bench press', 'chest fly machine', 'deadlift', 'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press', 'lat pulldown', 'lateral raises', 'leg extension', 'leg raises', 'plank', 'pull up', 'push up', 'romanian deadlift', 'russian twist', 'shoulder press', 'squat', 't bar row', 'tricep dips', 'tricep pushdown']

**Approach**:  
This is a **classification problem** solved using PyTorch, a popular deep learning framework. OpenCV is used to visualize the workout images for preprocessing and exploration.

**Model**:  
The classification model achieves an accuracy of **82.75%**, ensuring reliable identification of various workout activities from the image dataset.

## **3. Flask Application**

The final solution is deployed as a **Flask web application**, enabling:
- **Real-time calorie prediction**: Based on user-provided parameters such as gender, weight, workout duration, heart rate, and body temperature.

## **Technologies Used**
- **Languages**: Python
- **Libraries**: PyTorch, OpenCV, Numpy, Pandas, Matplotlib, Seaborn, Scikit-Learn
- **Web Framework**: Flask
- **Machine Learning Models**: Support Vector Regressor (SVR), Deep Learning with PyTorch

---

