import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("shopping_reviews_dataset.csv", sep="\t")

print("Dataset loaded successfully\n")
print(df.head())


# -----------------------------
# 2. Extract Review Text using Regex
# -----------------------------
def extract_review_text(review_block):

    pattern = r"<review-text>(.*?)</review-text>"
    match = re.search(pattern, review_block, re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


df["review_text"] = df["html_review"].apply(extract_review_text)

print("\nExtracted review text sample:")
print(df[["review_text", "label"]].head())


# -----------------------------
# 3. Prepare Data
# -----------------------------
X = df["review_text"]
y = df["label"]


# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 5. Vectorization Experiments
# -----------------------------
vectorizers = {
    "Count Unigram": CountVectorizer(ngram_range=(1,1)),
    "Count 1-2gram": CountVectorizer(ngram_range=(1,2)),
    "TFIDF Unigram": TfidfVectorizer(ngram_range=(1,1)),
    "TFIDF 1-2gram": TfidfVectorizer(ngram_range=(1,2))
}


best_model = None
best_vectorizer = None
best_f1 = 0
best_name = ""


# -----------------------------
# 6. Evaluation Function
# -----------------------------
def evaluate(name, y_true, y_pred):

    print("\n=================================")
    print("Experiment:", name)
    print("=================================")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy :", round(acc,4))
    print("Precision:", round(prec,4))
    print("Recall   :", round(rec,4))
    print("F1 Score :", round(f1,4))

    print("\nConfusion Matrix")
    print(cm)

    return f1


# -----------------------------
# 7. Run Experiments
# -----------------------------
for name, vectorizer in vectorizers.items():

    print("\nRunning experiment:", name)

    # Fit vectorizer ONLY on training data
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform test data
    X_test_vec = vectorizer.transform(X_test)

    # Train Linear SVM
    model = LinearSVC()

    model.fit(X_train_vec, y_train)

    # Predict
    y_pred = model.predict(X_test_vec)

    # Evaluate
    f1 = evaluate(name, y_test, y_pred)

    # Track best model
    if f1 > best_f1:

        best_f1 = f1
        best_model = model
        best_vectorizer = vectorizer
        best_name = name


# -----------------------------
# 8. Best Model
# -----------------------------
print("\n=================================")
print("BEST MODEL:", best_name)
print("BEST F1 SCORE:", round(best_f1,4))
print("=================================")


# -----------------------------
# 9. Save Model and Vectorizer
# -----------------------------
with open("svm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(best_vectorizer, f)

print("\nSaved objects:")
print("svm_model.pkl")
print("vectorizer.pkl")
