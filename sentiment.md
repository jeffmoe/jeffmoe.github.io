---
title: Building a Sentiment Analysis Model
parent: Natural Language Processing
nav_order: 2
---

## Overview
This project implements a sentiment analysis classifier for movie reviews using a Linear Support Vector Machine (SVM) with TF-IDF feature extraction. The model is trained on the IMDB movie reviews dataset (50,000 reviews) to predict whether a review expresses positive or negative sentiment.

---
## Features
- Text Preprocessing:
  - Contraction expansion (e.g., "don't" → "do not")
  - Tokenization using NLTK
  - Stop word removal
  - Lemmatization using WordNetLemmatizer
  - Special character removal

- Machine Learning Pipeline:
  - TF-IDF vectorization for feature extraction
  - Linear SVM classifier
  - Grid Search with cross-validation (3-fold) for hyperparameter tuning

- Hyperparameter Tuning:
  - max_df: [0.75, 0.85, 1.0]
  - min_df: [1, 2, 5]
  - ngram_range: [(1,1), (1,2)]
  - sublinear_tf: [True, False]
  - svm__C: [0.01, 0.1, 1.0, 10.0]

- Evaluation Metrics:
  - 1 = positive review, 0 = negative review  
  - Classification report (precision, recall, f1-score)
  - Confusion matrix visualization
  - Precision-Recall curve
  - ROC curve with AUC score
  - Decision boundary visualization (2D PCA projection)
 
  ---
## Installation
  1. Clone the repository
  2. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
  3. Download required NLTK data:
      ```python
      import nltk
      nltk.download('popular')
      ```
      
  ---
## Basic Use
### Load and preprocess data
```python
tokens, labels, mapping = prepare_data_from_csv(
    file_path='IMDB Dataset.csv',
    text_column='review',
    label_column='sentiment'
)

# Train the model
results = svm_pipeline(tokens, labels)

# Analyze results
analyze_svm_results(results)

# Make predictions on new reviews
preds, scores = results["predict_fn"](["Your review text here..."])
```

---
## Code File - Jupyter Notebook
```python
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import contractions
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import nltk
from nltk.downloader import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_recall_curve, roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

download('popular')
```
```python
def prepare_data_from_csv(
    file_path: str,
    text_column: str,
    label_column: str,
):
    """
    Loads CSV, preprocesses text, and encodes labels to 0/1.

    Args:
        file_path: path to CSV file
        text_column: column name containing text
        label_column: column name containing labels
        positive_label: value representing positive sentiment

    Returns:
        tokens_list: list of tokenized documents
        labels: list of 0/1 labels
    """
    # creating progress bar
    tqdm.pandas()

    # using pandas to load in the csv file of comments
    df = pd.read_csv(file_path)

    # remove missing values from the dataset
    df = df[[text_column,label_column]].dropna()

    # init the stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        try:
            # Handle input types
            if isinstance(text, list):
                text = ' '.join(text)
            elif not isinstance(text, str):
                return []

            # expand contractions
            text = contractions.fix(text)

            # tokenize
            tokens = nltk.word_tokenize(text)

            # normalize and clean
            tokens = [
                re.sub(r'[^a-zA-Z0-9]', '', w.lower())
                for w in tokens
            ]

            # remove empties and stop words
            tokens = [w for w in tokens if w and w not in stop_words]

            # lemmatize the words
            tokens = [lemmatizer.lemmatize(w) for w in tokens]

            return tokens

        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return []

    # apply the text preprocessing
    print("Preprocessing text data...")
    tokens_list = df[text_column].progress_apply(preprocess).tolist()

    # encode labels for analysis in the SVM pipeline
    unique_labels = sorted(df[label_column].astype(str).str.lower().unique())
    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected binary labels, but found: {unique_labels}"
        )
    label_mapping = {
        unique_labels[0]: 0,
        unique_labels[1]: 1
    }
    print(f"Label mapping: {label_mapping}")
    labels = df[label_column].astype(str).str.lower().map(label_mapping).tolist()

    return tokens_list, labels, label_mapping
```
```python
def svm_pipeline(tokens: Union[List[List[str]], Tuple[List[str]]],
                 labels: List[int]) -> Dict[str, Any]:
    """
    Builds a TF-IDF + Linear SVM pipeline with GridSearchCV.

    Args:
        tokens: list of tokenized documents (each doc = list of words)
        labels: list of binary labels (1 = positive, 0 = negative)

    Returns:
        dict containing:
            - best_model
            - best_params
            - X_train, X_test, y_train, y_test
            - y_pred
            - decision_scores
            - classification_report
            - confusion_matrix
            - predict_fn (for new tokens)
    """

    # Join tokens into strings for splitting data
    print("Preparing text data...")
    text_data = [' '.join(t) for t in tqdm(tokens, desc="Joining Tokens")]

    # splitting our data into training and testing sets for the model
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        text_data,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Creating a repeatable pipeline for model retraining using sklearn
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', LinearSVC(random_state=42, dual=False))
    ])

    # Using a hyperparameter grid to determine the best values for the model
    parameter_grid = {
        'tfidf__max_df': [0.75, 0.85, 1.0],
        'tfidf__min_df': [1, 2, 5],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__sublinear_tf': [True, False],
        'svm__C': [0.01, 0.1, 1.0, 10.0]
    }
    total_fits = (
        len(parameter_grid["tfidf__max_df"]) *
        len(parameter_grid["tfidf__min_df"]) *
        len(parameter_grid["tfidf__ngram_range"]) *
        len(parameter_grid["tfidf__sublinear_tf"]) *
        len(parameter_grid["svm__C"]) *
        3
    )



    # setting up our Grid search to implement cross-validation and determing the hyperparameters
    print(f"Running our grid search (~{total_fits} fits)...")
    gridsearch = GridSearchCV(
        pipeline,
        parameter_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    with tqdm_joblib(tqdm(desc="Grid Search Progress", total=total_fits)):
        # model training
        gridsearch.fit(X_train, y_train)
    print("Grid search is now done.")

    best_model = gridsearch.best_estimator_

    # using the best model from the grid search for our predictions
    print("Getting predictions...")
    y_pred = best_model.predict(X_test)

    # decision function is a good built in for visualizing hyperplanes with scatter plots
    decision_scores = best_model.decision_function(X_test)

    # getting the accuracy metrics for the model
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print("Predictions ready for analysis.")
    
    # function when wanting to test new data to see what the predicted outcome would be
    def predict_new(raw_text: List[str]):
        """
        Predict sentiment for raw text input.
        
        Args:
            raw_texts: list of raw text strings to classify
            
        Returns:
            preds: predicted labels (0 or 1)
            scores: decision function scores
        """
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        def preprocess(text):
            try:
                # Handle input types
                if isinstance(text, list):
                    text = ' '.join(text)
                elif not isinstance(text, str):
                    return []
                # expand contractions
                text = contractions.fix(text)
                # tokenize
                tokens = nltk.word_tokenize(text)
                # normalize and clean
                tokens = [
                    re.sub(r'[^a-zA-Z0-9]', '', w.lower())
                    for w in tokens
                ]
                # remove empties and stop words
                tokens = [w for w in tokens if w and w not in stop_words]
                # lemmatize the words
                tokens = [lemmatizer.lemmatize(w) for w in tokens]
                return tokens
            except Exception as e:
                print(f"Error preprocessing text: {e}")
                return []
        new_text = [preprocess(t) for t in raw_text]
        print(new_text)   
        new_text = [' '.join(t) for t in new_text]
        preds = best_model.predict(new_text)
        scores = best_model.decision_function(new_text)
        return preds, scores

    return {
        "best_model": best_model,
        "best_params": gridsearch.best_params_,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "decision_scores": decision_scores,
        "classification_report": report,
        "confusion_matrix": cm,
        "predict_fn": predict_new
    }
```
```python
def analyze_svm_results(results: dict):
    """
    Generates evaluation visualizations and metrics
    from svm_pipeline output.

    Args:
        results: dictionary returned from svm_pipeline()
    """
    # unpacking the results from the svm pipeline for analysis
    y_test = np.array(results["y_test"])
    y_pred = np.array(results["y_pred"])
    scores = np.array(results["decision_scores"])
    X_test = results["X_test"]

    best_model = results["best_model"]
    tfidf = best_model.named_steps["tfidf"]
    svm = best_model.named_steps["svm"]

    # transforming the test data using the tfidf vectorizer for visualization
    X_test_tfidf = tfidf.transform(X_test)
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_test_2d = svd.fit_transform(X_test_tfidf)

    # getting the hyperplane coefficients for visualization
    w = svm.coef_.ravel()
    w_2d = svd.components_.dot(w)
    bias = svm.intercept_[0]

    # visualizing the decision boundary of the SVM in the 2D PCA space
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_test_2d[:, 0], X_test_2d[:, 1], # plotting the 2D PCA projection of the test data
        c=y_test, # coloring points by their true labels
        cmap='coolwarm',
        alpha=0.6,
        edgecolors='k',
        s=35
    )
    misclassified = y_test != y_pred
    if misclassified.any():
        # highlight misclassified points with yellow edges
        plt.scatter(
            X_test_2d[misclassified, 0],
            X_test_2d[misclassified, 1],
            facecolors='none',
            edgecolors='yellow',
            s=90,
            linewidths=1.5,
            label='Misclassified'
        )

    # plotting the hyperplane (decision boundary) of the SVM
    x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
    line_x = np.linspace(x_min, x_max, 200)
    if abs(w_2d[1]) > 1e-6:
        line_y = -(w_2d[0] * line_x + bias) / w_2d[1]
        plt.plot(line_x, line_y, 'k--', linewidth=2, label='Hyperplane (approx)')
    else:
        plt.axvline(-bias / w_2d[0], color='k', linestyle='--', linewidth=2, label='Hyperplane (approx)')

    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.title('SVM decision boundary on 2D TF-IDF projection')
    plt.colorbar(scatter, label='True label', ticks=[0, 1])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

    # creating the precision recall curve 
    precision, recall, _ = precision_recall_curve(y_test, scores)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()

    # creating the roc curve (receiver operating characteristic) and area under the curve (auc) for analysis
    # receiver operating characteristic: True pos. rate (TPR) vs the false pos rate (FPR)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # using seaborn to display the confusion matrix from the model
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # displaying the classification report of the model
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
```
```python
tokens, labels, mapping = prepare_data_from_csv(
    file_path='IMDB Dataset.csv',
    text_column='review',
    label_column='sentiment'
)
print(mapping)
results = svm_pipeline(tokens,labels)
```
```text
Preprocessing text data...
100%|██████████| 50000/50000 [01:24<00:00, 589.38it/s]
Label mapping: {'negative': 0, 'positive': 1}
{'negative': 0, 'positive': 1}
Preparing text data...
Joining Tokens: 100%|██████████| 50000/50000 [00:00<00:00, 419704.05it/s]
Splitting data...
Running our grid search (~432 fits)...
Grid Search Progress:   0%|          | 0/432 [00:00<?, ?it/s]
  0%|          | 0/432 [00:00<?, ?it/s]
Grid search is now done.
Getting predictions...
Predictions ready for analysis.
```
```python
analyze_svm_results(results)
```
<p><img width="815" height="624" alt="image" src="https://github.com/user-attachments/assets/bb3cbc42-ad6e-4eb5-b728-aaad12fce021" /></p>
<p><img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/f3a7f9c1-ecec-43f6-a70b-a809d1fe1a28" /></p>
<p><img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/6af3e585-58be-411e-852f-6a98cd9f87ba" /></p>
<p><img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/cd951fda-a0de-4d0d-aa87-837ac9578129" /></p>

```text
Classification Report:

              precision    recall  f1-score   support

           0       0.92      0.90      0.91      5000
           1       0.90      0.93      0.92      5000

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000
```
```python
preds, scores = results["predict_fn"]([
    "This movie was amazing!",
    "This movie was terrible and I hated it."
])
print(f"Predictions for new texts: {preds}")
print(f"Decision scores for new texts: {scores}")
```
```text
[['movie', 'amazing'], ['movie', 'terrible', 'hated']]
Predictions for new texts: [1 0]
Decision scores for new texts: [ 1.98126893 -3.37981217]
```
