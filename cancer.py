import flask
import render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train_model')
def train_model():
    # Load data
    data = pd.read_csv("C:\\Users\\sansk\\Downloads\\ML-MT-WebApp-master\\ML-MT-WebApp-master\\cancer.csv")

    # Dropping unnecessary columns
    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

    # One-hot encode the diagnosis column
    diagnosis_dummies = pd.get_dummies(data["diagnosis"])

    # Concatenate the one-hot encoded columns with the original data
    cancer = pd.concat([data, diagnosis_dummies], axis=1)

    # Drop the original diagnosis column and the "B" column
    cancer.drop(["diagnosis", "B"], axis=1, inplace=True)

    # Rename the "M" column to "Malignant/Benign"
    cancer.rename(columns={"M": "Malignant/Benign"}, inplace=True)

    # Separate features (X) and target (y)
    y = cancer[["Malignant/Benign"]]
    X = cancer.drop(["Malignant/Benign"], axis=1)

    # Print the number of features before scaling
    print("Number of features before scaling:", X.shape[1])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the logistic regression model
    logreg = LogisticRegression(max_iter=1000)  # Increased max_iter
    logreg.fit(X_train_scaled, y_train.values.ravel())  # Using .ravel() to convert y to a 1d array if needed

    # Make predictions on the test set
    y_pred = logreg.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)

    # Save the trained model
    joblib.dump(logreg, "model")

    return accuracy

if __name__ == '__main__':
    app.run(debug=True)
