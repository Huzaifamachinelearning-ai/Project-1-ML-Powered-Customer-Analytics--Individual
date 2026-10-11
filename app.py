
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration and constants
# --------------------------------------------------

st.set_page_config(
    page_title="Churn Risk Advisor",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
META_PATH = BASE_DIR / "model_meta.json"

RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges",
]

SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

PAYMENTS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


# --------------------------------------------------
# Load the trained model and metadata
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {META_PATH.name}"
        )

    loaded_model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if "feature_columns" not in metadata:
        raise ValueError(
            "model_meta.json must contain feature_columns."
        )

    return loaded_model, metadata


try:
    model, meta = load_artifacts()
    COLS = meta["feature_columns"]
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()


# --------------------------------------------------
# Prepare raw data exactly as the lab specifies
# --------------------------------------------------

def prepare_input(raw, columns):
    missing = [c for c in RAW_COLS if c not in raw.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = raw[RAW_COLS].copy()

    # Ensure numerical columns are actually numeric.
    for col in ["SeniorCitizen", "tenure",
                "MonthlyCharges", "TotalCharges"]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    if data[["SeniorCitizen", "tenure",
             "MonthlyCharges", "TotalCharges"]].isna().any().any():
        raise ValueError(
            "Numeric customer fields contain invalid or missing values."
        )

    # IMPORTANT: do not use drop_first=True during serving.
    encoded = pd.get_dummies(data)

    # Match the exact feature columns and order used for training.
    encoded = encoded.reindex(columns=columns, fill_value=0)

    return encoded.astype(float)


def score(raw):
    features = prepare_input(raw, COLS)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "This app requires a model with predict_proba()."
        )

    probabilities = model.predict_proba(features)

    # Find the probability for the churn class.
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "steps"):
        final_estimator = model.steps[-1][1]
        classes = getattr(final_estimator, "classes_", None)

    if classes is not None:
        churn_indices = [
            i for i, value in enumerate(classes)
            if str(value).strip().lower() in
            {"yes", "1", "true", "churn"}
        ]

        if churn_indices:
            return probabilities[:, churn_indices[0]]

    # Fallback for the lab's usual binary target encoding:
    # 0 = No churn, 1 = Churn.
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]

    raise ValueError(
        "Unable to identify the churn class in model.classes_."
    )


# --------------------------------------------------
# Sidebar: individual customer profile
# --------------------------------------------------

with st.sidebar:
    st.header("Customer profile")

    tenure = st.slider("Tenure (months)", 0, 72, 4)

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )

    monthly = st.number_input(
        "Monthly charges (USD)",
        min_value=18.0,
        max_value=120.0,
        value=85.0,
        step=0.5,
    )

    internet = st.selectbox(
        "Internet service",
        ["Fiber optic", "DSL", "No"],
    )

    if internet != "No":
        services = st.multiselect(
            "Add-on services",
            SERVICES,
            default=["StreamingTV"],
        )
    else:
        services = []

    phone = st.radio(
        "Phone service", ["Yes", "No"], horizontal=True
    )

    multi = "No phone service"
    if phone == "Yes":
        multi = st.radio(
            "Multiple lines", ["No", "Yes"],
            horizontal=True,
        )

    payment = st.selectbox("Payment method", PAYMENTS)

    paperless = st.radio(
        "Paperless billing", ["Yes", "No"],
        horizontal=True,
    )

    with st.expander("Demographics"):
        gender = st.radio(
            "Gender", ["Female", "Male"],
            horizontal=True,
        )
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has partner")
        dependents = st.checkbox("Has dependents")

    st.divider()

    threshold = st.slider(
        "Contact threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(meta.get("threshold", 0.30)),
        step=0.05,
    )


# Build one raw customer row using the original dataset columns.
row = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": "Yes" if partner else "No",
    "Dependents": "Yes" if dependents else "No",
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multi,
    "InternetService": internet,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly,
}

for service in SERVICES:
    if internet == "No":
        row[service] = "No internet service"
    else:
        row[service] = "Yes" if service in services else "No"

customer = pd.DataFrame([row], columns=RAW_COLS)


# --------------------------------------------------
# Main application
# --------------------------------------------------

st.title("Customer Churn Risk Advisor")

cv_auc = meta.get("cv_auc")
cv_std = meta.get("cv_auc_std")

if cv_auc is not None and cv_std is not None:
    st.caption(
        f"{meta.get('model_name', 'Model')} | "
        f"CV AUC {float(cv_auc):.3f} "
        f"+/- {float(cv_std):.3f} | "
        "Decision support only"
    )
else:
    st.caption(
        f"{meta.get('model_name', 'Model')} | "
        "Decision support only"
    )

tab1, tab2, tab3 = st.tabs(
    ["One customer", "Batch scoring", "About"]
)


# --------------------------------------------------
# Tab 1: individual prediction and what-if analysis
# --------------------------------------------------

with tab1:
    try:
        p = float(score(customer)[0])

        band = (
            "HIGH" if p >= threshold
            else "WATCH" if p >= threshold / 2
            else "LOW"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn probability", f"{p:.1%}")
        c2.metric("Risk band", band)
        c3.metric(
            "Suggested action",
            "Contact for review" if band == "HIGH"
            else "Monitor" if band == "WATCH"
            else "Routine retention",
        )

        st.progress(min(max(p, 0.0), 1.0))

        st.subheader("What would change the risk?")
        st.caption(
            "These are model associations, not guaranteed causal effects."
        )

        changes = [
            ("Contract", value)
            for value in ["One year", "Two year"]
            if value != contract
        ]

        if payment == "Electronic check":
            changes.append(
                ("PaymentMethod", "Credit card (automatic)")
            )

        if internet != "No" and "TechSupport" not in services:
            changes.append(("TechSupport", "Yes"))

        results = []

        for feature, value in changes:
            alternative = customer.copy()
            alternative.loc[:, feature] = value

            new_probability = float(score(alternative)[0])

            results.append({
                "Change": f"{feature}: {value}",
                "New probability": round(new_probability, 3),
                "Difference": round(new_probability - p, 3),
            })

        if results:
            st.dataframe(
                pd.DataFrame(results),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No alternative scenarios to show.")

        # Optional explanation for a compatible Logistic Regression pipeline.
        final_estimator = (
            model.steps[-1][1]
            if hasattr(model, "steps")
            else model
        )

        if (
            hasattr(model, "steps")
            and hasattr(final_estimator, "coef_")
        ):
            with st.expander(
                "Why this score? (Logistic Regression)"
            ):
                try:
                    transformed = model[:-1].transform(
                        prepare_input(customer, COLS)
                    )[0]

                    contributions = pd.Series(
                        final_estimator.coef_[0] * transformed,
                        index=COLS,
                    )

                    contributions = contributions.reindex(
                        contributions.abs()
                        .sort_values(ascending=False)
                        .index
                    ).head(8)

                    st.bar_chart(contributions)
                    st.caption(
                        "Positive contributions push the model score "
                        "toward churn; negative contributions push it away."
                    )
                except Exception as exc:
                    st.info(
                        "The explanation is not compatible with this "
                        f"model's preprocessing: {exc}"
                    )

    except Exception as exc:
        st.error(f"Could not score this customer: {exc}")


# --------------------------------------------------
# Tab 2: batch scoring
# --------------------------------------------------

with tab2:
    st.write(
        "Upload a CSV containing the original Telco customer columns."
    )

    uploaded_file = st.file_uploader(
        "Customer CSV file", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            missing = [
                c for c in RAW_COLS if c not in data.columns
            ]

            if missing:
                st.error(f"Missing required columns: {missing}")
            elif data.empty:
                st.error("The uploaded CSV contains no customers.")
            else:
                # Allow blank TotalCharges cells to be treated as zero,
                # matching the lab instructions.
                data["TotalCharges"] = pd.to_numeric(
                    data["TotalCharges"], errors="coerce"
                ).fillna(0)

                probabilities = score(data)

                data["p_churn"] = np.round(
                    probabilities, 3
                )

                data["risk_band"] = np.where(
                    data["p_churn"] >= threshold,
                    "HIGH",
                    np.where(
                        data["p_churn"] >= threshold / 2,
                        "WATCH",
                        "LOW",
                    ),
                )

                data["contact"] = np.where(
                    data["p_churn"] >= threshold,
                    "Yes",
                    "",
                )

                st.write(
                    f"{(data['contact'] == 'Yes').sum()} of "
                    f"{len(data)} customers are above the threshold."
                )

                st.dataframe(
                    data.sort_values(
                        "p_churn", ascending=False
                    ).head(50),
                    hide_index=True,
                    use_container_width=True,
                )

                csv_output = data.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download scored CSV",
                    data=csv_output,
                    file_name="scored_customers.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")


# --------------------------------------------------
# Tab 3: metadata and limitations
# --------------------------------------------------

with tab3:
    st.subheader("Model information")

    public_metadata = {
        key: value
        for key, value in meta.items()
        if key != "feature_columns"
    }

    st.json(public_metadata)

    st.markdown(
        """
        **Intended use:** Help a telecom retention team prioritize
        customers for review.

        **Limitations:** The model was trained on a US telecom dataset.
        It has not been validated on Pakistani telecom customers.
        What-if results show associations learned from data, not proof
        that changing a contract or payment method will prevent churn.

        **Not for:** Credit decisions, service denial, or other
        high-impact automated decisions.
        """
    )
