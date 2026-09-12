"""
Intent Classification Module.
Implements:
1. TF-IDF + Logistic Regression Intent Classifier
2. Baseline 1: Majority Class Classifier
3. Baseline 2: Keyword Rule-Based Classifier
4. Comparative Model Evaluation (Accuracy, Precision, Recall, F1)
"""

import os
import pickle
import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Sequence, Union
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from src.preprocess import INTENT_LABELS, INTENT_KEYWORDS, assign_intent

logger = logging.getLogger(__name__)


class MajorityBaselineClassifier:
    """Baseline 1: Always predicts the most common intent in the training set."""
    def __init__(self):
        self.majority_class: str = "Other"

    def fit(self, X: pd.Series, y: pd.Series):
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, X: Union[pd.Series, Sequence[str]]):
        return [self.majority_class] * len(X)

    def predict_proba(self, X: Union[pd.Series, Sequence[str]]):
        # Uniform or dummy probability
        return [[1.0 / len(INTENT_LABELS)] * len(INTENT_LABELS) for _ in range(len(X))]


class KeywordBaselineClassifier:
    """Baseline 2: Rule-based regex keyword matching classifier."""
    def fit(self, X: pd.Series, y: pd.Series):
        return self

    def predict(self, X: Union[pd.Series, Sequence[str]]):
        return [assign_intent(str(text)) for text in X]


class IntentClassifier:
    """Production Intent Classifier using TF-IDF and tuned Multiclass Logistic Regression."""
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            stop_words="english",
            strip_accents="unicode",
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=1.5,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
        self.is_trained = False
        self.classes_ = list(INTENT_LABELS)

    def fit(self, X: pd.Series, y: pd.Series):
        logger.info("Fitting TF-IDF Vectorizer...")
        X_vec = self.vectorizer.fit_transform(X)
        logger.info("Training Logistic Regression classifier...")
        self.classifier.fit(X_vec, y)
        self.classes_ = list(self.classifier.classes_)
        self.is_trained = True
        return self

    def predict(self, X: Union[pd.Series, Sequence[str]]):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict(X_vec)

    def predict_proba(self, X: Union[pd.Series, Sequence[str]]):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting probabilities.")
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vec)

    def predict_single(self, text: str) -> Tuple[str, float]:
        """Returns the top predicted intent and confidence probability score."""
        probs = self.predict_proba([text])[0]
        max_idx = probs.argmax() if hasattr(probs, "argmax") else max(range(len(probs)), key=lambda i: probs[i])
        predicted_intent = self.classes_[max_idx]
        confidence = float(probs[max_idx])
        return predicted_intent, confidence

    def save(self, file_path: str = "data/intent_model.joblib"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier, "classes": self.classes_}, f)
        logger.info(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str = "data/intent_model.joblib"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            import importlib
            joblib = importlib.import_module("joblib")
            data = joblib.load(file_path)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.classifier = data["classifier"]
        instance.classes_ = data["classes"]
        instance.is_trained = True
        logger.info(f"Model loaded from {file_path}")
        return instance


def compute_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, float]:
    """Computes Accuracy, Precision, Recall, and F1 (macro and weighted)."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }


def train_and_evaluate(df: pd.DataFrame, model_save_path: str = "data/intent_model.joblib") -> Dict[str, Any]:
    """
    Trains TF-IDF + Logistic Regression and compares against Majority & Keyword Baselines.
    Returns comparative evaluation metrics dictionary.
    """
    X = df["clean_text"]
    y = df["intent"]

    # Stratified Train/Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Train TF-IDF + Logistic Regression
    lr_model = IntentClassifier()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_metrics = compute_metrics(y_test.values, lr_preds)
    lr_model.save(model_save_path)

    # 2. Baseline 1: Majority Class
    majority_model = MajorityBaselineClassifier()
    majority_model.fit(X_train, y_train)
    maj_preds = majority_model.predict(X_test)
    majority_metrics = compute_metrics(y_test.values, maj_preds)

    # 3. Baseline 2: Keyword Rule-Based
    keyword_model = KeywordBaselineClassifier()
    keyword_model.fit(X_train, y_train)
    kw_preds = keyword_model.predict(X_test)
    keyword_metrics = compute_metrics(y_test.values, kw_preds)

    detailed_report = classification_report(y_test, lr_preds, zero_division=0)

    results = {
        "logistic_regression": lr_metrics,
        "baseline_majority": majority_metrics,
        "baseline_keyword": keyword_metrics,
        "detailed_classification_report": detailed_report,
        "classes": list(lr_model.classes_),
    }

    return results


if __name__ == "__main__":
    from src.preprocess import load_and_preprocess_data
    df = load_and_preprocess_data()
    eval_results = train_and_evaluate(df)
    print("\n--- BASELINE & MODEL EVALUATION ---")
    print(f"Majority Baseline F1:       {eval_results['baseline_majority']['f1_weighted']}")
    print(f"Keyword Baseline F1:        {eval_results['baseline_keyword']['f1_weighted']}")
    print(f"Logistic Regression F1:     {eval_results['logistic_regression']['f1_weighted']}")
    print("\nClassification Report:\n", eval_results["detailed_classification_report"])
