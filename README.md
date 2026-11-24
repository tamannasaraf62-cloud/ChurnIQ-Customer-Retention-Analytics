
# ChurnIQ – Customer Churn & Retention Analytics Platform

An End-to-End Machine Learning Project for Bank Customer Churn Prediction using XGBoost and Streamlit.

---

## 1. Project Overview

ChurnIQ is an end-to-end Machine Learning project designed to analyze banking customer behavior and predict whether a customer is likely to leave a bank.

Customer churn is a major challenge for banking institutions because losing existing customers can affect revenue, customer relationships, and long-term business growth.

ChurnIQ uses historical customer data and an XGBoost classification model to identify customers who may leave the bank. It also provides an interactive Streamlit dashboard to explore customer data, understand churn patterns, visualize important features, and generate churn predictions.

The main objective is to transform customer data into meaningful insights that can support customer retention strategies and data-driven decision-making.

---

## 2. Business Problem

Banks invest significant resources in acquiring and maintaining customers. However, some customers eventually stop using banking services or move to competing institutions.

Identifying customers who are likely to leave in advance can help banks:

- Understand customer churn behavior.
- Identify customers who may be at risk of leaving.
- Analyze factors associated with customer churn.
- Improve customer retention strategies.
- Support better business decisions through data analysis.

ChurnIQ addresses this problem using Machine Learning.

---

## 3. Project Objectives

The primary objectives of this project are:

1. Perform data cleaning and preprocessing.
2. Explore customer demographics and banking behavior.
3. Perform Exploratory Data Analysis (EDA).
4. Prepare relevant features for Machine Learning.
5. Train and evaluate classification models.
6. Develop an XGBoost-based churn prediction model.
7. Analyze feature importance and model performance.
8. Build an interactive Streamlit web application.
9. Present customer churn and retention insights through visualizations.

---

## 4. Dataset Information

The project uses a Bank Customer Churn dataset.

### Dataset Summary

| Attribute | Description |
|---|---|
| Total Records | 10,000 |
| Original Columns | 14 |
| Target Variable | Exited |
| Problem Type | Binary Classification |
| Churned Customers | 2,037 |
| Retained Customers | 7,963 |
| Churn Percentage | 20.37% |

### Features Description

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country of residence |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Number of years with the bank |
| Balance | Customer account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Customer's estimated salary |
| Exited | Target variable indicating customer churn |

### Target Variable

The target column is `Exited`.

- `0` – Customer retained by the bank.
- `1` – Customer exited the bank.

The dataset contains an imbalanced target distribution, with approximately 20.37% churned customers and 79.63% retained customers.

---

## 5. Project Workflow

The project follows an end-to-end Machine Learning workflow.

### Step 1: Data Cleaning

- Load the original customer dataset.
- Inspect dataset structure and data types.
- Remove irrelevant identifier columns.
- Check for missing values.
- Check for duplicate records.
- Prepare the cleaned dataset.

### Step 2: Exploratory Data Analysis (EDA)

- Analyze customer demographics.
- Explore customer age and credit scores.
- Study customer distribution across geographical regions.
- Analyze account balance and estimated salary.
- Examine banking product usage.
- Investigate active membership.
- Understand customer churn distribution.
- Identify patterns associated with churn.

### Step 3: Feature Engineering

- Select relevant input features.
- Separate independent and dependent variables.
- Prepare numerical and categorical features.
- Apply appropriate preprocessing.
- Prepare data for model training.

### Step 4: Model Building

- Train classification models.
- Evaluate model performance.
- Develop an XGBoost classification model.
- Compare prediction performance using evaluation metrics.
- Select the final trained model.

### Step 5: Model Evaluation

Evaluate the model using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### Step 6: Deployment

- Save the trained model.
- Integrate the model into a Streamlit application.
- Develop an interactive dashboard.
- Deploy the application using Streamlit Community Cloud.

---

## 6. Machine Learning Model

### XGBoost Classifier

ChurnIQ uses XGBoost (Extreme Gradient Boosting), an ensemble learning algorithm based on gradient-boosted decision trees.

XGBoost builds multiple decision trees sequentially, where each new tree attempts to improve the predictions made by previous trees.

It is used for the customer churn prediction task because it can model complex relationships between customer features and churn behavior.

### Prediction Output

The application predicts whether a customer is likely to:

- Stay with the bank.
- Leave the bank.

The prediction probability and selected classification threshold are used to determine the final churn prediction.

---

## 7. Model Performance

The following are the held-out test results at the selected classification threshold of 0.35.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 85.65% |
| Precision | 65.87% |
| Recall | 61.18% |
| F1-Score | 63.44% |
| ROC-AUC | 86.82% |

These metrics provide different perspectives on the model's performance.

- Accuracy measures the overall proportion of correct predictions.
- Precision measures how many predicted churn cases were actually churn cases.
- Recall measures how many actual churn cases were identified.
- F1-Score balances precision and recall.
- ROC-AUC measures the model's ability to distinguish between the two classes across classification thresholds.

---

## 8. Feature Importance

Feature importance helps identify which customer attributes contribute most to the XGBoost model's predictions.

The three highest-ranked features in the current model are:

1. Age
2. NumOfProducts
3. IsActiveMember

Other customer features also contribute to the model's predictions.

Feature importance represents model reliance on input features. It does not establish a causal relationship between a feature and customer churn.

---

## 9. Streamlit Application Features

ChurnIQ provides an interactive web application built using Streamlit.

### Dashboard

- Overall customer statistics.
- Customer churn and retention overview.
- Dataset information.
- Column descriptions.
- Data quality overview.
- Customer distribution visualizations.
- Statistical summaries.

### Churn Prediction

- Enter customer details.
- Generate churn predictions.
- View predicted customer status.
- Explore churn probability.

### Analytics

- Explore customer behavior.
- Visualize churn patterns.
- Analyze customer demographics.
- Understand relationships between customer features and churn.

### Model Insights

- View model evaluation metrics.
- Explore feature importance.
- Understand the selected classification threshold.
- Review model limitations.

---

## 10. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive charts |
| Scikit-learn | Data preprocessing and model evaluation |
| XGBoost | Machine Learning model |
| Streamlit | Web application development |
| Jupyter Notebook | Experimentation and development |
| Git & GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

---

## 11. Project Structure

```text
ChurnIQ-Customer-Retention-Analytics/
│
├── app.py
├── requirements.txt
│
├── best_churn_model.pkl
├── churn_features.pkl
├── churn_threshold.pkl
├── churn_scaler.pkl
│
├── churn_cleaned.csv
├── Churn_Modelling.csv
├── churn_prepared_data.npz
│
├── 01_Data_Cleaning.ipynb
├── 02_EDA_churn.ipynb
├── 03_Feature_Engineering.ipynb
├── 04_Model_Building.ipynb
├── 05_Churn_Prediction.ipynb
│
└── README.md
```

---

## 12. Installation and Setup

Follow these steps to run ChurnIQ locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/tamannasaraf62-cloud/ChurnIQ-Customer-Retention-Analytics.git
```

### Step 2: Navigate to the Project Folder

```bash
cd ChurnIQ-Customer-Retention-Analytics
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 13. Deployment

The application is intended to be deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the project files to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Select `app.py` as the main application file.
4. Deploy the application.

**Live Application:** Add your deployed Streamlit URL here.

---

## 14. Future Improvements

Potential improvements include:

- Experimenting with additional classification algorithms.
- Further hyperparameter optimization.
- Improving churn detection and recall.
- Adding more detailed customer segmentation.
- Developing customer-level retention recommendations.
- Integrating explainable AI techniques.
- Improving dashboard interactivity and user experience.

---

## 15. Project Disclaimer

This project is developed for educational and analytical purposes using a historical bank customer dataset.

The model's predictions are estimates based on learned patterns in the dataset. They should not be treated as guaranteed outcomes or used as the sole basis for real banking decisions.

---

## 16. Author

**Tamanna Saraf**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: [tamannasaraf62-cloud](https://github.com/tamannasaraf62-cloud)

---

If you find this project interesting, feel free to explore the repository and its implementation.
