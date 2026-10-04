
import io
import os

import fitz
import streamlit as st

from predict import (
    predict_document,
    explain_prediction,
    CONFIDENCE_THRESHOLD
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Document Risk Classifier",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .risk-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #ddd;
    }

    .metric-title {
        font-size: 14px;
        color: #666;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# PDF TEXT EXTRACTION
# ==================================================

def extract_pdf_text(file_bytes):

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:

        page_text = page.get_text()

        if page_text:
            pages.append(page_text)

    document.close()

    return "\n".join(pages)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🛡️ AI Document Risk Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload a business document to automatically classify
    its potential information risk level.
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(
        "**Model:** TF-IDF + Logistic Regression"
    )

    st.write(
        "**Classes:** LOW / MEDIUM / HIGH"
    )

    st.write(
        f"**Review threshold:** "
        f"{CONFIDENCE_THRESHOLD:.0%}"
    )

    st.divider()

    st.info(
        """
        Documents with low prediction confidence
        are flagged for manual review.
        """
    )

    st.divider()

    st.caption(
        "⚠️ This project is a portfolio demonstration "
        "using synthetic training data."
    )


# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "📄 Upload a PDF document",
    type=["pdf"]
)


# ==================================================
# PROCESS DOCUMENT
# ==================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    with st.spinner("Reading and analyzing document..."):

        try:

            extracted_text = extract_pdf_text(
                file_bytes
            )

        except Exception as error:

            st.error(
                f"Could not read the PDF: {error}"
            )

            st.stop()


    # ------------------------------------------------
    # Check extracted text
    # ------------------------------------------------

    if not extracted_text.strip():

        st.error(
            "No selectable text was found in this PDF."
        )

        st.warning(
            "This may be a scanned/image-only PDF. "
            "OCR support would be required."
        )

        st.stop()


    # ------------------------------------------------
    # Document statistics
    # ------------------------------------------------

    character_count = len(extracted_text)

    word_count = len(
        extracted_text.split()
    )

    line_count = len(
        extracted_text.splitlines()
    )


    # ==================================================
    # DOCUMENT STATISTICS
    # ==================================================

    st.subheader("📊 Document Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Characters",
            f"{character_count:,}"
        )

    with col2:

        st.metric(
            "Words",
            f"{word_count:,}"
        )

    with col3:

        st.metric(
            "Lines",
            f"{line_count:,}"
        )


    # ==================================================
    # RISK PREDICTION
    # ==================================================

    result = predict_document(
        extracted_text
    )

    risk = result["risk"]
    confidence = result["confidence"]
    manual_review = result["manual_review"]
    probabilities = result["probabilities"]


    st.divider()

    st.subheader("🔍 Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Risk Level",
            risk.upper()
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

    with col3:

        if manual_review:

            st.metric(
                "Manual Review",
                "REQUIRED"
            )

        else:

            st.metric(
                "Manual Review",
                "NOT REQUIRED"
            )


    # ==================================================
    # CONFIDENCE BAR
    # ==================================================

    st.write("### Confidence")

    st.progress(
        min(confidence, 1.0)
    )

    if manual_review:

        st.warning(
            f"⚠️ Confidence is below the "
            f"{CONFIDENCE_THRESHOLD:.0%} threshold. "
            "Manual review is recommended."
        )

    else:

        st.success(
            "✅ Prediction confidence is above "
            "the manual-review threshold."
        )


    # ==================================================
    # CLASS PROBABILITIES
    # ==================================================

    st.subheader("📈 Risk Probabilities")

    probability_columns = st.columns(3)

    class_order = [
        "low",
        "medium",
        "high"
    ]

    for column, class_name in zip(
        probability_columns,
        class_order
    ):

        probability = probabilities.get(
            class_name,
            0.0
        )

        with column:

            st.metric(
                class_name.upper(),
                f"{probability:.2%}"
            )

            st.progress(
                min(probability, 1.0)
            )


    # ==================================================
    # EXPLAINABILITY
    # ==================================================

    st.divider()

    st.subheader(
        "💡 Important Terms"
    )

    important_terms = explain_prediction(
        extracted_text,
        top_n=10
    )

    if important_terms:

        st.dataframe(
            important_terms,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No important terms could be identified."
        )


    # ==================================================
    # DOCUMENT PREVIEW
    # ==================================================

    st.divider()

    st.subheader(
        "📄 Document Preview"
    )

    preview_length = 5000

    preview_text = extracted_text[
        :preview_length
    ]

    st.text_area(
        "Extracted text",
        preview_text,
        height=300
    )

    if character_count > preview_length:

        st.caption(
            f"Showing the first {preview_length:,} "
            f"characters of {character_count:,}."
        )


else:

    st.info(
        "👆 Upload a PDF above to begin the analysis."
    )
