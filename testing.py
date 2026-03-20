import pickle
import numpy as np

# -----------------------------
# 1. Load Saved Model and Vectorizer
# -----------------------------
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully.\n")

# -----------------------------
# 2. Accept User Input for Review
# -----------------------------
user_input = input("Enter a product review: ")

# -----------------------------
# 3. Preprocess and Transform Review
# -----------------------------
# Transform the review using the saved vectorizer
review_vec = vectorizer.transform([user_input])

# -----------------------------
# 4. Predict Sentiment
# -----------------------------
prediction = model.predict(review_vec)

# -----------------------------
# 5. Display Result
# -----------------------------
if prediction == 1:
    print("Prediction: Positive Review")
else:
    print("Prediction: Negative Review")