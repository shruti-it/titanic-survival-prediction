# Titanic Survival Prediction 🚢

Predicts whether a passenger survived the Titanic disaster or not, using Machine Learning based on features like Age, Gender, Ticket Class, and Fare.

## Dataset
- Titanic dataset from Kaggle
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Target: Survived (0 = No, 1 = Yes)

## What I Did
- Data Cleaning - Handled missing values in Age, Cabin & Embarked
- Exploratory Data Analysis (EDA) - Analyzed survival rate by Gender, Class & Age
- Feature Engineering - Converted Sex & Embarked to numeric values
- Model Building - Logistic Regression / Random Forest Classifier
- Model Evaluation - Accuracy, Confusion Matrix, Classification Report

## Files in this Project
- `app.py` / `titanic.py` - Main code file
- `train.csv` / `Titanic.csv` - Dataset
- `model.pkl` - Trained Model
- `requirements.txt` - Libraries used

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run
pip install -r requirements.txt
python app.py

## Result
Model predicts survival with good accuracy and shows that Gender and Ticket Class were the most important factors.
