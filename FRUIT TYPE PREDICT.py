import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv(r'C:\Users\gauth\PycharmProjects\major\fruits.csv')


# Display the first few rows of the dataframe
print(data.head())

# Initialize the LabelEncoder
le = LabelEncoder()

# Encode the 'fruit_name' column
data['fruit_name'] = le.fit_transform(data['fruit_name'])

# Features and target variable
X = data[['mass', 'width', 'height', 'color_score']]  # Use appropriate features
y = data['fruit_name']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# EDA (Exploratory Data Analysis)
# Basic statistics
print(data.describe())

# Plot the distribution of different fruits
plt.figure(figsize=(10, 6))
sns.countplot(x='fruit_name', data=data)
plt.title('Distribution of Fruits')
plt.xlabel('Fruit Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Scatter plot for color features
plt.figure(figsize=(12, 8))
sns.scatterplot(x='mass', y='color_score', hue='fruit_name', data=data, palette='tab10')
plt.title('Mass vs Color Score of Fruits')
plt.xlabel('Mass')
plt.ylabel('Color Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Model Building and Evaluation
# Initialize classifiers
models = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'Accuracy': accuracy,
        'Classification Report': report,
        'Confusion Matrix': cm
    }

# Display results
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['Accuracy']:.2f}")
    print(f"Classification Report:\n{metrics['Classification Report']}")
    print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")
    print("\n" + "=" * 50 + "\n")

# Save the best model
best_model = max(results, key=lambda k: results[k]['Accuracy'])
joblib.dump(models[best_model], '/content/best_model.pkl')
joblib.dump(le, '/content/label_encoder.pkl')

# Streamlit App (Optional)
import streamlit as st
import numpy as np

# Load the trained model
model = joblib.load('/content/best_model.pkl')
le = joblib.load('/content/label_encoder.pkl')

# Define the Streamlit app
st.title('Fruit Classification App')
st.write('Enter the features of the fruit to predict its type.')

# Input fields for features
mass = st.number_input('Mass', min_value=0.0, max_value=1000.0, value=100.0)
width = st.number_input('Width', min_value=0.0, max_value=100.0, value=7.0)
height = st.number_input('Height', min_value=0.0, max_value=100.0, value=7.0)
color_score = st.number_input('Color Score', min_value=0.0, max_value=1.0, value=0.5)

# Predict the fruit type
if st.button('Predict'):
    features = np.array([[mass, width, height, color_score]])
    prediction = model.predict(features)
    fruit_name = le.inverse_transform(prediction)
    st.write(f'The predicted fruit is: {fruit_name[0]}')

# To run the Streamlit app, save this code in a file named 'app.py' and run:
# !streamlit run app.py
