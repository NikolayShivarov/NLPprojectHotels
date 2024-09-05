# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV files
df = pd.read_csv("tripadvisor_reviews_processed_discrete.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["Review"], df["Rating"], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_vect, y_train)

# Predict ratings on the test set
y_pred = classifier.predict(X_test_vect)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.xlabel("Predicted Rating")
plt.ylabel("True Rating")
plt.title("Confusion Matrix TripAdvisor")
plt.show()

# Example usage:
new_reviews = [
    "The hotel room was clean and comfortable.",
    "Awful experience, terrible service.",
    "Very nice hotel",
    "It was very noisy at night",
    "The room was dirty",
    "Delicious food"
]
new_reviews_vect = vectorizer.transform(new_reviews)
predicted_ratings = classifier.predict(new_reviews_vect)
for review, rating in zip(new_reviews, predicted_ratings):
    print(f"Review: {review}\nPredicted rating: {rating}\n")
