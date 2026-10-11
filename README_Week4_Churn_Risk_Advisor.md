# Churn Risk Advisor
### ML-Powered Customer Analytics | Project 1 — Week 4

> **Purpose:** A machine-learning decision-support app that estimates customer churn probability, highlights risk bands, explores hypothetical customer-profile changes, and scores customer batches from a CSV file.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Project-Week%204-informational.svg)]()

---

## Table of Contents
- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Repository Structure](#repository-structure)
- [Run Locally](#run-locally)
- [Deploy to Streamlit Community Cloud](#deploy-to-streamlit-community-cloud)
- [Using the App](#using-the-app)
- [Training–Serving Parity](#trainingserving-parity)
- [Model Card](#model-card)
- [Screenshots](#screenshots)
- [Check Your Understanding](#check-your-understanding)
- [Known Limitations](#known-limitations)
- [Week 4 Deliverables Checklist](#week-4-deliverables-checklist)

---

## Overview

**Churn Risk Advisor** packages the customer-churn model developed in Weeks 1–3 into a Streamlit web application. It is designed to help a telecom retention team **prioritize customers for review** by showing estimated churn probabilities and allowing users to compare hypothetical customer profiles.

The app is built to use the same feature columns and preprocessing logic as the trained model. A parity test is used to check that predictions generated through the app's serving pipeline match the predictions from the training notebook.

**Important:** This is a portfolio and educational project. The output is a model estimate—not a guarantee that a customer will leave or stay, and not a substitute for human judgment.

## Live Demo

- **Public app:** `ADD_YOUR_STREAMLIT_APP_URL_HERE`
- **GitHub repository:** `ADD_YOUR_GITHUB_REPOSITORY_URL_HERE`
- **Week 1–3 notebooks:** `ADD_YOUR_NOTEBOOK_OR_KAGGLE_LINK_HERE`

Replace these placeholders with your actual links before publishing or submitting the project.

## Features

### 1. Single-customer scoring
- Enter a customer profile using the sidebar controls.
- Estimate the customer's churn probability.
- Display a risk band: **LOW**, **WATCH**, or **HIGH**.
- Compare the estimated probability with a configurable contact threshold.

### 2. What-if analysis
- Compare the current profile with selected alternatives, such as a different contract type, payment method, or technical-support option.
- Display the new estimated probability and the difference from the baseline.
- Treat these comparisons as **associations learned from historical data**, not proof that a change will cause a customer to stay.

### 3. Model explanation
- If the selected model is a compatible Logistic Regression pipeline, show the largest feature contributions available from its coefficients.
- Positive contributions push the model score toward churn; negative contributions push it away.
- This explanation is model-specific and should not be interpreted as causal evidence.

### 4. Batch scoring
- Upload a CSV containing the original customer columns.
- Validate that required columns are present.
- Calculate churn probabilities and flag records above the selected threshold.
- Review the highest-scoring customers and download the scored CSV.

### 5. Model metadata
- Read model information from `model_meta.json`, including the model name, feature-column order, scikit-learn version, threshold, and evaluation metrics.
- Keep the training and deployment environment aligned by pinning the required library versions.

## How It Works

1. **Train:** Fit and evaluate the model in the Week 3 notebook.
2. **Package:** Save the trained estimator as `churn_model.joblib` and its metadata as `model_meta.json`.
3. **Prepare inputs:** Convert raw customer fields into dummy-encoded features, align them to the model's exact training columns, and fill missing encoded columns with zero.
4. **Predict:** Use `predict_proba` to calculate the estimated probability of churn.
5. **Present:** Display the score, risk band, what-if comparisons, and optional explanation in Streamlit.
6. **Batch score:** Apply the same preprocessing and scoring pipeline to each row of an uploaded CSV.

### Risk bands

The app uses the user-selected contact threshold `t`:

- **HIGH:** probability is greater than or equal to `t`.
- **WATCH:** probability is at least half of `t`, but below `t`.
- **LOW:** probability is below half of `t`.

Changing the threshold changes the risk band and contact flag, **not the model's underlying probability**.

## Model Performance

Enter the actual results from your Week 3 notebook and `model_meta.json`. Do not use example values from the lab instructions.

| Metric | Your result |
|---|---|
| Final model | `ADD_ACTUAL_MODEL_NAME` |
| Cross-validation AUC (mean) | `ADD_ACTUAL_CV_AUC_MEAN` |
| Cross-validation AUC (standard deviation) | `ADD_ACTUAL_CV_AUC_STD` |
| Test AUC | `ADD_ACTUAL_TEST_AUC` |
| Contact threshold | `ADD_ACTUAL_THRESHOLD` |
| scikit-learn version used for training | `ADD_ACTUAL_SKLEARN_VERSION` |
| Training dataset | IBM Telco Customer Churn |
| Dataset size | 7,043 customers (as described in the lab instructions) |

**Interpretation:** ROC AUC measures how well the model ranks churners above non-churners across thresholds. Report the cross-validation mean and standard deviation to communicate variability; do not present a single score as certainty about future performance.

## Repository Structure

The repository should contain the following project files:

```text
churn-risk-advisor/
├── app.py
├── churn_model.joblib
├── model_meta.json
├── requirements.txt
├── sample_customers.csv
├── README.md
└── .gitignore
```

- `app.py` — Streamlit user interface, preprocessing, scoring, what-if analysis, and batch scoring.
- `churn_model.joblib` — Serialized trained model or pipeline.
- `model_meta.json` — Model name, ordered feature columns, library version, threshold, metrics, and version.
- `requirements.txt` — Runtime dependencies, including the exact training-time scikit-learn version.
- `sample_customers.csv` — Small sample file for demonstrating batch scoring.
- `.gitignore` — Excludes local virtual environments and other files that should not be committed.

## Run Locally

### Requirements
- Python 3.11 is recommended by the lab instructions.
- Git (optional, for cloning the repository).
- The model artifact and metadata files listed above.

### 1. Clone the repository

```bash
git clone ADD_YOUR_GITHUB_REPOSITORY_URL_HERE
cd churn-risk-advisor
```

If you have not pushed the repository yet, create the project folder and place the required files inside it.

### 2. Create and activate a virtual environment

**Windows (PowerShell or Command Prompt):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure `requirements.txt` pins the **exact scikit-learn version used to train the model**. If the model is XGBoost, include a compatible, pinned `xgboost` dependency as well. The dependency list must match the estimator serialized in `churn_model.joblib`.

### 4. Launch the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## Deploy to Streamlit Community Cloud

1. Push the project files to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Create an app using the repository, the `main` branch, and `app.py`.
4. In the advanced settings, select a Python version compatible with the pinned dependencies—Python 3.11 is recommended by the lab instructions.
5. Deploy and review the build logs.
6. Open the public URL in a private/incognito browser window to verify that the app works outside your own session.
7. Add the working URL to the [Live Demo](#live-demo) section above.

### Deployment troubleshooting

| Error or symptom | Likely cause | What to check |
|---|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | The serialized model depends on XGBoost, but it is not installed in the deployment environment. | Add a compatible pinned `xgboost` version to `requirements.txt`, commit, and redeploy. Only do this if the final model is XGBoost. |
| `InconsistentVersionWarning` or an unpickling error | scikit-learn version differs from the training environment. | Pin the exact training version and select a compatible Python version. |
| `FileNotFoundError: churn_model.joblib` | The model artifact is missing or stored at a different path. | Commit the file next to `app.py`, or update the path consistently. |
| `ModuleNotFoundError` for another package | A runtime dependency is missing. | Add the needed package to `requirements.txt`, pin it when appropriate, then redeploy. |
| App loads but predictions appear identical | Input encoding or feature alignment may be wrong. | Check the serving preprocessing function and run the full parity test again. |
| A dependency cannot be installed | The pinned version may not support the selected Python version. | Choose a compatible Python version or a compatible dependency version, then re-test model loading and prediction parity. |

## Using the App

### Single customer
1. Set the customer profile in the sidebar.
2. Review churn probability and risk band.
3. Adjust the contact threshold to see how the decision band changes.
4. Review the what-if table to compare alternative profiles.
5. If available, expand the Logistic Regression explanation.

### Batch scoring
1. Open **Batch scoring**.
2. Upload a CSV containing the original customer fields expected by the app.
3. If columns are missing, use the displayed error to correct the file.
4. Review the scored records and contact flags.
5. Download the resulting `scored_customers.csv`.

The sample CSV is intended for a demonstration. For real use, verify data quality, column meanings, missing-value handling, and compatibility with the model's training data.

## Training–Serving Parity

A key Week 4 objective is preventing **training–serving skew**: a model can run without errors and still produce different predictions if serving-time features are encoded differently from training-time features.

The serving function should:
- Encode categorical fields without dropping the first category.
- Reindex features to the exact column list stored in model metadata.
- Fill absent dummy columns with zero.
- Preserve the numeric feature representation expected by the model.

The parity test compares probabilities from the notebook's training feature matrix with probabilities generated through the serving preprocessing function for **every customer row**. The test should pass with a maximum difference of zero or a very small floating-point tolerance (around `1e-15`).

Example acceptance check from the lab workflow:

```python
p_train = model.predict_proba(X[cols])[:, 1]
p_serve = model.predict_proba(prepare_input(raw, cols))[:, 1]

print("Largest difference:", np.abs(p_train - p_serve).max())
assert np.allclose(p_train, p_serve), "Training-serving skew detected!"
```

Record your actual largest difference and parity-test result here:

- **Largest probability difference:** `ADD_ACTUAL_MAX_DIFFERENCE`
- **Rows checked:** `ADD_ACTUAL_ROW_COUNT`
- **Result:** `PASS / FAIL — replace with your actual result`

## Model Card

### Churn Risk Advisor v1.0

**Intended use**  
Rank existing telecom customers by estimated churn risk so a retention team can prioritize which customers to review or contact. The app is a decision-support tool; it does not automatically make retention decisions.

**Out-of-scope uses**  
Do not use this model for credit decisions, customer pricing, eligibility decisions, or decisions that deny a person a service. Do not treat the score as a definitive judgment about an individual customer.

**Data**  
IBM Telco Customer Churn dataset, described in the lab instructions as 7,043 customers. The dataset represents a US telecom context and may not represent other populations or current market conditions.

**Model and features**  
- **Model:** `ADD_ACTUAL_MODEL_NAME`
- **Selection method:** 5-fold cross-validation, as described in the lab workflow.
- **Feature columns:** loaded from `model_meta.json`; report the actual count from your artifact.
- **Decision threshold:** `ADD_ACTUAL_THRESHOLD`, selected from the project's cost analysis.

**Performance**  
- **Cross-validation ROC AUC:** `ADD_MEAN ± ADD_STD`
- **Test ROC AUC:** `ADD_TEST_AUC`
- **Test-set policy:** the test set should be used for final evaluation, not repeatedly used to tune the model.

**Decision costs**  
The lab's example cost analysis uses a false negative (FN) cost of PKR 6,000 and a false positive (FP) cost of PKR 1,000. Confirm that these are the values used in your own analysis before reporting them as project results.

**Limitations and risks**
- Trained on one historical US telecom dataset; it has not been validated on Pakistani telecom customers.
- Customer behavior, plans, and market conditions may change, causing model drift.
- Model associations do not establish causal effects. A what-if result does not prove that changing a contract or service will prevent churn.
- Predictions may be inaccurate for unusual, incomplete, or out-of-distribution customer profiles.
- Fairness across groups has not been established. A future check should compare recall across gender and senior-citizen groups, as proposed in the lab instructions.
- Human review is required before taking action based on a prediction.

**Owner and version**
- **Owner:** `ADD_YOUR_NAME`
- **Version:** `1.0`
- **Last updated:** `ADD_DATE`

## Screenshots

Add two screenshots captured from your running app. Keep customer information fictional or anonymized.

### Single-customer result
<!-- Replace this line with: ![Single-customer scoring screenshot](screenshots/single_customer.png) -->

`ADD_SINGLE_CUSTOMER_SCREENSHOT_HERE`

### Batch-scoring result
<!-- Replace this line with: ![Batch scoring screenshot](screenshots/batch_scoring.png) -->

`ADD_BATCH_SCORING_SCREENSHOT_HERE`

Suggested layout:

```text
screenshots/
├── single_customer.png
└── batch_scoring.png
```

## Check Your Understanding

These answers address the six Week 4 questions from the lab instructions.

**1. Why can `get_dummies(drop_first=True)` break predictions for a single customer?**  
When dummy encoding is performed on one row, the category present in that row may become the dropped reference category. This can erase the distinction between categories and produce a feature vector that does not match the representation used during training.

**2. What does the parity test compare, and why should it cover every training row?**  
It compares predicted churn probabilities from the training feature matrix with probabilities generated through the serving-time preprocessing function. Checking every row is more reliable than checking a few examples because encoding mismatches can affect only particular categories or combinations of features.

**3. Should the model use `st.cache_resource` or `st.cache_data`? Why?**  
Use `st.cache_resource` for the loaded model because the model is a shared resource that should be loaded once rather than reloaded on every interaction. `st.cache_data` is generally intended for caching data-returning computations.

**4. Why derive `TotalCharges` instead of asking the user to type it?**  
Deriving it keeps the input interface simpler and avoids inconsistent values between tenure, monthly charges, and total charges. The app's derivation is an approximation, however, and should be validated against the dataset's actual definition of `TotalCharges` before use beyond the lab.

**5. Why pin the exact scikit-learn version used in training?**  
Serialized estimators may not be compatible across scikit-learn versions. Pinning the training version reduces environment mismatch and helps the deployed app load the model reproducibly.

**6. If a two-year contract lowers predicted risk from 0.79 to 0.27, does offering one make the customer stay? How could a company find out?**  
No. The comparison is an association produced by the model, not evidence that changing the contract causes retention. A company could evaluate the offer through a properly designed experiment, such as a randomized controlled test, while monitoring outcomes and customer impact.

## Known Limitations

- This app is an educational prototype and has not been validated for operational use.
- Results depend on the training data, preprocessing, model artifact, and configured threshold.
- The hypothetical `TotalCharges` calculation may not match real billing history.
- Batch predictions require the expected original input columns and compatible values.
- A successful deployment does not by itself prove model quality, fairness, calibration, or business value.

## Week 4 Deliverables Checklist

Use this list before submitting the project.

- [ ] Training–serving skew demo completed and both probabilities documented.
- [ ] Parity test passes on every customer row.
- [ ] `model_meta.json` contains the actual feature columns, library version, threshold, and metrics.
- [ ] `requirements.txt` pins the actual scikit-learn version used for training and includes any required estimator library.
- [ ] `app.py` passes the single-customer, threshold, what-if, explanation (when supported), and batch-scoring checks.
- [ ] Public Streamlit URL opens in a private/incognito browser window.
- [ ] README model card contains actual values and clearly states out-of-scope uses.
- [ ] Single-customer and batch-scoring screenshots added.
- [ ] Week 1–3 notebook links added.
- [ ] Six understanding questions reviewed.
- [ ] Five-minute presentation prepared, including a backup demo video.
- [ ] LinkedIn post includes the live URL, a real bug fixed, and performance numbers with uncertainty.

---

## Acknowledgements

Prepared for **Introduction to Applied AI — BS 8th Semester, Department of Electronics**, as the Week 4 documentation for *Project 1: ML-Powered Customer Analytics*.

**Before submission:** replace every `ADD_...` placeholder with your real project information, verify the metrics against your notebook and metadata, and make sure the public URL and screenshots are genuine.
