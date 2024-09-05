# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data from CSV files
df = pd.read_csv("booking_reviews_processed_numerical.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["Review"], df["Rating"], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a ridge regression model
regressor = Ridge()
regressor.fit(X_train_vect, y_train)

# Predict ratings on the test set
y_pred = regressor.predict(X_test_vect)

# Evaluate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, s = 3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Ratings")
plt.ylabel("Residuals")
plt.title("Residual Plot Booking")
plt.grid(True)
plt.show()

# Example usage:
new_reviews = [
    "The hotel room was clean and comfortable.",
    "Awful experience, terrible service.",
    "Very nice hotel",
    "It was very noisy at night",
    "The room was dirty",
    "Delicious food",
    "Deliciuos food, clean and comfortable room, nice place overall",
    "Delicious food, dirty room"
]
new_reviews_vect = vectorizer.transform(new_reviews)
predicted_ratings = regressor.predict(new_reviews_vect)
for review, rating in zip(new_reviews, predicted_ratings):
    print(f"Review: {review}\nPredicted rating: {rating}\n")
    
def convert_rating(rating):
    if rating > 10:
        return 10 
    elif rating < 1:
        return 1
    else:
        return rating

convert_rating_vec = np.vectorize(convert_rating)
y_pred = convert_rating_vec(y_pred)
    
# Evaluate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error, after bounds:", mse)       
    
    