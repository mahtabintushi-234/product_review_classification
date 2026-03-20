
# Product Review Sentiment Classification
Intro to Artificial Intelligence – Assignment 3 
Author: **Mahtabin Tushi**  
Date: **2026‑03‑19**

This project implements a sentiment classification system for product reviews, using different vectorization methods and a linear SVM classifier. The goal is to predict whether a given review is positive or negative.

## 1. Assignment Overview

This project uses a **linear SVM** model to classify product reviews based on their sentiment (positive or negative). Various text vectorization methods are compared to determine the most effective representation of the review data for classification.

## 2. Dataset Overview

The dataset used is named `shopping_reviews_dataset.csv` and contains product reviews in an HTML-like structure. Each review includes metadata and the review text that needs to be extracted. The dataset contains two key columns:
- **html_review**: The raw HTML-like structure of the review.
- **label**: The sentiment label for the review (1 for positive, 0 for negative).

### Example Review:
```xml
<product-review>
    <meta>
        <pid>12345</pid>
        <category>Electronics</category>
        <brand>TechPro</brand>
        <item>Smartphone</item>
    </meta>
    <score value="5" max="5"/>
    <review-text>Great phone, highly recommended!</review-text>
</product-review>
````

## 3. Data Extraction Using Regex

The review text is extracted using a regular expression from the `html_review` column. The review is extracted from between the `<review-text>` tags to provide clean text for sentiment analysis.

### Regex Pattern:

```python
pattern = r"<review-text>(.*?)</review-text>"
```

This regular expression matches the content inside the `<review-text>` tags and returns the review text.

## 4. Vectorization Methods Compared

The following vectorization techniques were compared for text feature extraction:

1. **CountVectorizer with Unigrams** (`ngram_range=(1,1)`): This approach considers only individual words as features.
2. **CountVectorizer with 1-2 Grams** (`ngram_range=(1,2)`): This approach considers both individual words (unigrams) and pairs of consecutive words (bigrams) as features.
3. **TF-IDF with Unigrams** (`ngram_range=(1,1)`): Similar to CountVectorizer, but with **Term Frequency-Inverse Document Frequency (TF-IDF)** weighting to down-weight common words and emphasize rarer words.
4. **TF-IDF with 1-2 Grams** (`ngram_range=(1,2)`): Similar to the previous TF-IDF method but includes both unigrams and bigrams for better context capture.

These vectorization methods help determine how text features are represented numerically for training the classification model.

## 5. Model Used (Linear SVM)

The model used for classification is a **Linear Support Vector Machine (SVM)**. The linear SVM classifier was chosen due to its effectiveness in text classification tasks, where the data can be represented as sparse high-dimensional vectors.

### SVM Implementation:

```python
model = LinearSVC()
```

This model is trained on the vectorized review data and then used for predicting whether a given review is positive or negative.

## 6. Performance Results for Each Setup

The model's performance was evaluated using the following metrics:

* **Accuracy**: The proportion of correct predictions.
* **Precision**: The proportion of true positive predictions over the total positive predictions.
* **Recall**: The proportion of true positive predictions over the actual positives.
* **F1 Score**: The harmonic mean of precision and recall.
* **Confusion Matrix**: Shows the true positives, false positives, true negatives, and false negatives.

### Experiment Results:

Each vectorizer setup was tested, and the **best model** was chosen based on the **F1 score**. The results of each experiment are printed during the evaluation process.

## 7. Which Setup Was Selected as Best

The setup with the **highest F1 score** was selected as the best model. This model provided the best balance between precision and recall, ensuring accurate sentiment classification.

### Best Model:

* **Vectorizer**: TF-IDF with 1-2 grams (`ngram_range=(1, 2)`)
* **F1 Score**: (best F1 score achieved from experiments)

## 8. Discussion of Why It Performed Best

The **TF-IDF with 1-2 grams** performed best because it captured both individual words (unigrams) and adjacent word pairs (bigrams), providing richer contextual information for sentiment classification. The **TF-IDF** weighting helped the model focus on more informative words, reducing the impact of frequent but less meaningful words.

### Key Points:

* Using **bigrams** (1-2 grams) improved context understanding, making the model better at identifying sentiment in longer or more complex reviews.
* **TF-IDF** helped reduce the influence of common words and highlighted important words in the reviews.

## 9. What Objects Were Saved in `.pkl` Format

After training, the following objects were saved for later use:

* **svm_model.pkl**: The trained **Linear SVM model**.
* **vectorizer.pkl**: The **vectorizer** (CountVectorizer or TfidfVectorizer) used to transform the text data into numerical features.

These objects can be loaded later for predictions on new data.

## 10. How to Run `training.py`

1. Ensure the dataset `shopping_reviews_dataset.csv` is in the same directory as the script.
2. Run the training script to train the model and save the best model and vectorizer:

   ```bash
   python training.py
   ```

### This will:

* Load the dataset.
* Extract the review text.
* Run experiments with different vectorization methods.
* Save the **best model** and **vectorizer** to disk.

## 11. How to Run `testing.py`

1. After training, use the saved model and vectorizer to predict the sentiment of new reviews.
2. Run the testing script:

   ```bash
   python testing.py
   ```

### This will:

* Load the saved **model** and **vectorizer**.
* Accept user input for a product review.
* Predict whether the review is **positive** or **negative**.

### Example:

```bash
Model and vectorizer loaded successfully.

Enter a product review: "It is a bad movie."
Prediction: Negative Review
```

