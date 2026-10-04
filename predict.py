
import os
import joblib
import numpy as np

from preprocessing import clean_text


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.70


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)

classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "risk_classifier.pkl"
    )
)


# --------------------------------------------------
# Predict document risk
# --------------------------------------------------

def predict_document(text):

    cleaned = clean_text(text)

    features = vectorizer.transform(
        [cleaned]
    )

    prediction = classifier.predict(
        features
    )[0]

    probabilities = classifier.predict_proba(
        features
    )[0]

    probability_dict = {
        class_name: float(probability)
        for class_name, probability
        in zip(
            classifier.classes_,
            probabilities
        )
    }

    confidence = float(
        np.max(probabilities)
    )

    manual_review = (
        confidence < CONFIDENCE_THRESHOLD
    )

    return {
        "risk": prediction,
        "confidence": confidence,
        "manual_review": manual_review,
        "probabilities": probability_dict
    }


# --------------------------------------------------
# Explain prediction
# --------------------------------------------------

def explain_prediction(text, top_n=10):

    cleaned = clean_text(text)

    features = vectorizer.transform(
        [cleaned]
    )

    prediction = classifier.predict(
        features
    )[0]

    class_index = list(
        classifier.classes_
    ).index(prediction)

    coefficients = classifier.coef_[
        class_index
    ]

    feature_values = features.toarray()[0]

    contributions = (
        feature_values * coefficients
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    indices = np.argsort(
        contributions
    )[::-1]

    important_terms = []

    for index in indices:

        if feature_values[index] > 0:

            important_terms.append({
                "Term": feature_names[index],
                "Contribution": round(
                    float(contributions[index]),
                    4
                )
            })

        if len(important_terms) >= top_n:
            break

    return important_terms
