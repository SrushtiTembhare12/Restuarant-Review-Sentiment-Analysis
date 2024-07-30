from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Initialize the Flask application
app = Flask(__name__)

# Load the data
df = pd.read_csv("C:\\Users\\bhush\\Downloads\\Restaurant_Reviews.csv")

# Extract features and target variable
X = df["Review"]
y = df["Liked"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Function to predict the sentiment of a review
def predict_review(review):
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    return "Positive Review" if prediction == 1 else "Negative Review"

# Define a route for the default URL, which loads the form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for handling the review submission
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    prediction = predict_review(review)
    return render_template('index.html', review=review, prediction=prediction)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
