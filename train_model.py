
import os
import random
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import clean_text


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(42)


# --------------------------------------------------
# Synthetic document templates
# --------------------------------------------------

low_templates = [
    "The company announced a general employee event for all staff.",
    "Employees are invited to participate in the annual sports competition.",
    "The marketing department announced a public product campaign.",
    "The company newsletter contains information about upcoming activities.",
    "The organization will conduct a general employee training session.",
    "The company published information about an upcoming public event.",
    "Employees can participate in the annual company celebration.",
    "The organization announced new workplace activities.",
    "The company website contains general information about its products.",
    "The organization published a general announcement for employees.",
    "The office will remain closed during the scheduled public holiday.",
    "Employees are invited to attend a general awareness workshop.",
    "The company has announced a new employee engagement program.",
    "The organization published its quarterly public newsletter.",
    "The company will organize a general team-building activity."
]

medium_templates = [
    "The vendor agreement describes service levels and operational responsibilities.",
    "The supplier contract contains payment terms and service requirements.",
    "Employees must follow the company's internal operating procedures.",
    "The service agreement defines responsibilities between the company and vendor.",
    "Managers are responsible for reviewing internal business processes.",
    "The company has established procedures for managing business operations.",
    "The agreement describes delivery schedules and vendor responsibilities.",
    "The internal policy defines procedures that employees must follow.",
    "The supplier must follow the company's operational requirements.",
    "The business agreement contains terms related to services and payments.",
    "The vendor must comply with the organization's operational requirements.",
    "The internal procedure defines responsibilities for business activities.",
    "The contract specifies service delivery and reporting requirements.",
    "Managers must review operational processes on a regular basis.",
    "The supplier agreement establishes responsibilities for both parties."
]

high_templates = [
    "This document contains confidential customer personal information.",
    "Customer financial information must be protected from unauthorized access.",
    "The document contains sensitive banking and financial information.",
    "Personal data must only be accessed by authorized personnel.",
    "Confidential customer records must not be shared with unauthorized parties.",
    "The document contains sensitive personal and financial records.",
    "Access to customer information must be restricted to authorized employees.",
    "Sensitive information must be protected using appropriate security controls.",
    "The confidential agreement contains customer financial information.",
    "Personal information and financial records require restricted access.",
    "The document contains personally identifiable information about customers.",
    "Confidential banking records must be protected from unauthorized disclosure.",
    "The file contains sensitive customer account information.",
    "Access to confidential financial records is strictly restricted.",
    "Customer identity and financial information must remain confidential."
]


# --------------------------------------------------
# Generate synthetic documents
# --------------------------------------------------

def generate_documents(templates, label, count):

    documents = []

    for _ in range(count):

        num_sentences = random.randint(3, 6)

        selected = random.choices(
            templates,
            k=num_sentences
        )

        text = " ".join(selected)

        documents.append({
            "text": text,
            "label": label
        })

    return documents


low_data = generate_documents(
    low_templates,
    "low",
    500
)

medium_data = generate_documents(
    medium_templates,
    "medium",
    500
)

high_data = generate_documents(
    high_templates,
    "high",
    500
)


# --------------------------------------------------
# Create dataset
# --------------------------------------------------

all_data = low_data + medium_data + high_data

df = pd.DataFrame(all_data)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save dataset
# --------------------------------------------------

dataset_path = os.path.join(
    DATASET_DIR,
    "risk_documents.csv"
)

df[["text", "label"]].to_csv(
    dataset_path,
    index=False
)


# --------------------------------------------------
# Clean text
# --------------------------------------------------

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

classifier = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

classifier.fit(
    X_train_tfidf,
    y_train
)


# --------------------------------------------------
# Save models
# --------------------------------------------------

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

classifier_path = os.path.join(
    MODEL_DIR,
    "risk_classifier.pkl"
)

joblib.dump(
    vectorizer,
    vectorizer_path
)

joblib.dump(
    classifier,
    classifier_path
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)

print(f"Total documents : {len(df)}")
print(f"Training docs   : {len(X_train)}")
print(f"Testing docs    : {len(X_test)}")

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nFiles created:")
print(dataset_path)
print(vectorizer_path)
print(classifier_path)
