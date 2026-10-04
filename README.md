
# 🛡️ AI Document Risk & Compliance Classifier

An AI/ML project that analyzes business documents and classifies their potential information risk as:

- 🟢 LOW
- 🟡 MEDIUM
- 🔴 HIGH

The application also provides a confidence score and flags uncertain predictions for manual human review.

---

## 🚀 Features

- Upload PDF documents
- Extract text automatically
- Classify document risk
- Display prediction confidence
- Show LOW/MEDIUM/HIGH probabilities
- Flag low-confidence predictions for manual review
- Show important terms influencing the prediction
- Display document statistics
- Preview extracted document text
- Streamlit web interface

---

## 🧠 Machine Learning Pipeline

The project uses:

1. PDF text extraction using PyMuPDF
2. Text cleaning using Python Regular Expressions
3. TF-IDF text vectorization
4. Logistic Regression classification
5. Probability-based confidence scoring
6. Human-in-the-loop manual review

### Pipeline

PDF → Text Extraction → Cleaning → TF-IDF → Logistic Regression → Risk Prediction → Confidence → Manual Review

---

## 📊 Risk Categories

### LOW

General business information that does not contain obvious sensitive information.

### MEDIUM

Operational, vendor, contract, or internal business information.

### HIGH

Documents containing potentially sensitive information such as:

- Personal information
- Customer information
- Financial information
- Banking information
- Confidential records
- Restricted information

---

## 🛠️ Technologies

- Python
- Streamlit
- Scikit-learn
- PyMuPDF
- Pandas
- NumPy
- Joblib
- TF-IDF
- Logistic Regression

---

## 📁 Project Structure

```text
AI-Document-Risk-Classifier/
│
├── app.py
├── train_model.py
├── predict.py
├── preprocessing.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── risk_documents.csv
│
└── models/
    ├── tfidf_vectorizer.pkl
    └── risk_classifier.pkl
