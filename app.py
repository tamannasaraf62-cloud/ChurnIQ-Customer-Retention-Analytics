
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("best_churn_model.pkl")
    threshold = joblib.load("churn_threshold.pkl")
    return model, threshold

@st.cache_data
def load_data():
    return pd.read_csv("churn_cleaned.csv")

model, threshold = load_model()
data = load_data()

# --------------------------------------------------
# DESIGN SYSTEM
# --------------------------------------------------

st.markdown("""
<style>

/* Application */
.stApp {
    background: #F5F7FA;
    color: #172B4D;
}

.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #101D36;
    border-right: 1px solid #263957;
}

[data-testid="stSidebar"] * {
    color: #E8EEF7;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 12px;
    border-radius: 8px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #203553;
}

/* Typography */
h1, h2, h3, h4 {
    color: #172B4D !important;
    font-weight: 700 !important;
}

p, label {
    color: #52627A;
}

[data-testid="stCaptionContainer"] p {
    color: #718096;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 3px 10px rgba(20, 40, 70, 0.035);
}

[data-testid="stMetricLabel"] {
    color: #718096 !important;
}

[data-testid="stMetricValue"] {
    color: #172B4D !important;
    font-weight: 700 !important;
}

/* Input labels */
[data-testid="stWidgetLabel"] p {
    color: #34445E !important;
    font-weight: 600 !important;
}

/* Input fields */
.stNumberInput input,
.stTextInput input {
    background: #FFFFFF !important;
    color: #172B4D !important;
}

[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #172B4D !important;
}

/* Buttons */
.stFormSubmitButton button,
.stButton button {
    background: #087F8C;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stFormSubmitButton button:hover,
.stButton button:hover {
    background: #066873;
    border: none;
}

/* Custom header */
.page-header {
    margin-bottom: 25px;
}

.page-eyebrow {
    color: #087F8C;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.page-description {
    color: #718096;
    font-size: 15px;
    margin-top: -8px;
}

/* Custom cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 15px;
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 14px;
    padding: 28px;
    margin-top: 20px;
}

.result-label {
    color: #718096;
    font-size: 13px;
    font-weight: 600;
}

.result-value {
    color: #172B4D;
    font-size: 38px;
    font-weight: 750;
    margin-top: 5px;
}

.result-status {
    font-size: 16px;
    font-weight: 700;
    margin-top: 12px;
}

/* Dividers */
hr {
    border-color: #E4EAF2;
}

/* Hide Streamlit footer */
footer {
    visibility: hidden;
}

/* Personal Footer */
.page-footer {
    margin-top: 50px;
    padding: 22px 0 12px 0;
    border-top: 1px solid #E4EAF2;
    text-align: center;
}

# ------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------

st.divider()

st.markdown("### Feature Importance")
st.caption(
    "Relative importance of customer features "
    "used by the trained XGBoost model."
)

# Extract model components
preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

# Get feature names and importance scores
feature_names = preprocessor.get_feature_names_out()
importances = xgb_model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Clean feature names
importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

# Convert to percentage
importance_df["Importance"] *= 100

# Sort features
importance_df = importance_df.sort_values(
    "Importance", ascending=True
)

# Create horizontal bar chart
fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Importance",
    color_discrete_sequence=["#087F8C"]
)

fig_importance.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
    cliponaxis=False
)

fig_importance.update_layout(
    height=500,
    xaxis_title="Relative Importance (%)",
    yaxis_title="",
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=20, r=70, t=20, b=20),
    xaxis_range=[
        0,
        importance_df["Importance"].max() * 1.25
    ]
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

st.info(
    "Age, Number of Products and Active Membership "
    "are the three highest-ranked features. "
    "Feature importance indicates model reliance, "
    "not causation."
)

/* Improve caption visibility */
.stCaption {
    color: #334155 !important;
    font-size: 15px !important;
}

/* Improve general text visibility */
.stMarkdown p {
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="padding: 15px 0 25px 0;">
        <div style="font-size:27px;font-weight:800;color:white;">
            CHURN<span style="color:#22C5B5;">IQ</span>
        </div>
        <div style="font-size:12px;color:#A9B8CC;margin-top:5px;">
            CUSTOMER INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Churn Prediction",
            "Analytics",
            "Model Insights"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size:12px;color:#A9B8CC;line-height:2;">
        <b style="color:#FFFFFF;">MODEL STATUS</b><br>
        ● XGBoost Loaded<br>
        ● Prediction Engine Ready<br>
        <br>
        <b style="color:#FFFFFF;">DECISION THRESHOLD</b><br>
        0.35
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:relative;margin-top:45px;
    font-size:11px;color:#8495AD;">
        ChurnIQ · Customer Retention Analytics
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# REUSABLE PAGE HEADER
# --------------------------------------------------

def page_header(eyebrow, title, description):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-eyebrow">{eyebrow}</div>
        <h1 style="margin-top:8px;">{title}</h1>
        <div class="page-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    page_header(
        "OVERVIEW",
        "Customer Intelligence",
        "A summary of customer retention and churn patterns."
    )

    total = len(data)
    churned = int(data["Exited"].sum())
    retained = total - churned
    churn_rate = (churned / total) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Historical Churn Rate", f"{churn_rate:.2f}%")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churned Customers", f"{churned:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Customer Retention Overview")

        chart_data = pd.DataFrame({
            "Customer Status": ["Retained", "Churned"],
            "Customers": [retained, churned]
        })

        fig = px.pie(
            chart_data,
            names="Customer Status",
            values="Customers",
            hole=0.68,
            color="Customer Status",
            color_discrete_map={
                "Retained": "#087F8C",
                "Churned": "#F08A72"
            }
        )

        fig.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Key Business Observations")

        st.markdown("""
        <div class="info-card">
            <div style="color:#087F8C;font-weight:700;font-size:13px;">
                CUSTOMER CHURN
            </div>
            <h2 style="margin:8px 0;">20.37%</h2>
            <p style="margin:0;">Historical customers who exited the bank.</p>
        </div>

        <div class="info-card">
            <div style="color:#087F8C;font-weight:700;font-size:13px;">
                CUSTOMER RETENTION
            </div>
            <h2 style="margin:8px 0;">79.63%</h2>
            <p style="margin:0;">Historical customers who remained.</p>
        </div>

        <div class="info-card">
            <div style="color:#087F8C;font-weight:700;font-size:13px;">
                PREDICTION ENGINE
            </div>
            <h2 style="margin:8px 0;">XGBoost</h2>
            <p style="margin:0;">Trained binary classification model.</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "All dashboard statistics are calculated from the historical "
        "Bank Customer Churn dataset. They are not live banking statistics."
    )

    

        # --------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------

    st.divider()

    st.markdown("### Dataset Overview")
    st.caption(
        "An overview of the dataset used to train the customer churn prediction model."
    )

    total_rows = data.shape[0]
    total_columns = data.shape[1]
    input_features = total_columns - 1

    d1, d2, d3 = st.columns(3)

    d1.metric("Total Records", f"{total_rows:,}")
    d2.metric("Total Columns", total_columns)
    d3.metric("Input Features", input_features)

    st.markdown("#### About the Dataset")

    st.markdown("""
    <div class="info-card">
        This dataset contains historical bank customer information.
        It includes customer demographics, account details and banking
        activity. The target column <b>Exited</b> indicates whether a
        customer left the bank.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Column Information")

    column_descriptions = {
        "CreditScore": "Customer's credit score",
        "Geography": "Country where the customer resides",
        "Gender": "Customer's gender",
        "Age": "Customer's age",
        "Tenure": "Number of years the customer has been with the bank",
        "Balance": "Customer's account balance",
        "NumOfProducts": "Number of bank products used by the customer",
        "HasCrCard": "Whether the customer has a credit card (1 = Yes, 0 = No)",
        "IsActiveMember": "Whether the customer is an active member (1 = Yes, 0 = No)",
        "EstimatedSalary": "Estimated annual salary of the customer",
        "Exited": "Target variable (1 = Churned, 0 = Retained)"
    }

    column_info = pd.DataFrame({
        "Column Name": data.columns,
        "Description": [
            column_descriptions.get(col, "Customer dataset feature")
            for col in data.columns
        ],
        "Data Type": [
            str(data[col].dtype)
            for col in data.columns
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The model uses 10 customer features to predict the target "
        "variable Exited."
    )

        # --------------------------------------------------
    # DATA QUALITY OVERVIEW
    # --------------------------------------------------

    st.markdown("#### Data Quality Overview")

    missing_values = int(data.isnull().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    numerical_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Missing Values", f"{missing_values:,}")
    q2.metric("Duplicate Rows", f"{duplicate_rows:,}")
    q3.metric("Numerical Columns", len(numerical_columns))
    q4.metric("Categorical Columns", len(categorical_columns))

# --------------------------------------------------
# CHURN PREDICTION
# --------------------------------------------------

elif page == "Churn Prediction":

    page_header(
        "PREDICTIVE ANALYTICS",
        "Customer Risk Assessment",
        "Enter customer information to estimate the probability of churn."
    )

    with st.container(border=True):

        st.markdown("### Customer Information")
        st.caption("Provide the customer's account and profile details.")

        with st.form("churn_form"):

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown("**PERSONAL & ACCOUNT DETAILS**")

                credit_score = st.number_input(
                    "Credit Score", 300, 850, 650
                )

                geography = st.selectbox(
                    "Geography", ["France", "Germany", "Spain"]
                )

                gender = st.selectbox(
                    "Gender", ["Male", "Female"]
                )

                age = st.number_input(
                    "Age", 18, 100, 35
                )

                tenure = st.number_input(
                    "Tenure (Years)", 0, 10, 3
                )

            with col2:
                st.markdown("**FINANCIAL & MEMBERSHIP DETAILS**")

                balance = st.number_input(
                    "Account Balance",
                    min_value=0.0,
                    value=50000.0,
                    step=1000.0
                )

                products = st.selectbox(
                    "Number of Products", [1, 2, 3, 4]
                )

                card = st.selectbox(
                    "Has Credit Card", ["Yes", "No"]
                )

                active = st.selectbox(
                    "Active Member", ["Yes", "No"]
                )

                salary = st.number_input(
                    "Estimated Salary",
                    min_value=0.0,
                    value=75000.0,
                    step=1000.0
                )

            submitted = st.form_submit_button(
                "Generate Churn Prediction",
                use_container_width=True
            )

    if submitted:

        customer = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": 1 if card == "Yes" else 0,
            "IsActiveMember": 1 if active == "Yes" else 0,
            "EstimatedSalary": salary
        }])

        probability = float(model.predict_proba(customer)[0][1])
        prediction = probability >= threshold

        st.markdown("### Prediction Result")

        if prediction:
            status = "HIGH CHURN RISK"
            message = "This customer is predicted to leave."
            color = "#C2413B"
            background = "#FFF1F0"
        else:
            status = "LOW CHURN RISK"
            message = "This customer is predicted to stay."
            color = "#087F5B"
            background = "#EAF8F1"

        st.markdown(f"""
        <div class="result-card" style="border-left:5px solid {color};">
            <div class="result-label">ESTIMATED CHURN PROBABILITY</div>
            <div class="result-value">{probability * 100:.2f}%</div>
            <div style="background:#E8EDF3;height:9px;
            border-radius:10px;margin-top:18px;">
                <div style="background:{color};
                width:{probability * 100:.2f}%;
                height:9px;border-radius:10px;"></div>
            </div>
            <div style="background:{background};padding:13px;
            border-radius:8px;margin-top:20px;">
                <div class="result-status" style="color:{color};">
                    {status}
                </div>
                <div style="color:#52627A;font-size:14px;margin-top:4px;">
                    {message}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            f"Decision threshold: {threshold:.2f} | "
            "Predictions are estimates from the trained model."
        )

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------


elif page == "Analytics":

    page_header(
        "BUSINESS ANALYTICS",
        "Customer Churn Analysis",
        "Explore customer behavior, identify churn patterns and discover retention opportunities."
    )

    # ------------------------------------------
    # 1. CHURN BY GEOGRAPHY
    # ------------------------------------------

    geo = data.groupby("Geography").agg(
        Churn_Rate=("Exited", "mean"),
        Customers=("Exited", "count")
    ).reset_index()

    geo["Churn_Rate"] *= 100

    # 2. CHURN BY GENDER
    gender = data.groupby("Gender").agg(
        Churn_Rate=("Exited", "mean"),
        Customers=("Exited", "count")
    ).reset_index()

    gender["Churn_Rate"] *= 100

    # 3. CHURN BY MEMBERSHIP
    active = data.groupby("IsActiveMember").agg(
        Churn_Rate=("Exited", "mean"),
        Customers=("Exited", "count")
    ).reset_index()

    active["Membership"] = active["IsActiveMember"].map({
        0: "Inactive",
        1: "Active"
    })

    active["Churn_Rate"] *= 100

    # 4. CHURN BY PRODUCTS
    products = data.groupby("NumOfProducts").agg(
        Churn_Rate=("Exited", "mean"),
        Customers=("Exited", "count")
    ).reset_index()

    products["Churn_Rate"] *= 100

    # 5. AGE DISTRIBUTION
    age_data = data.copy()
    age_data["Customer Status"] = age_data["Exited"].map({
        0: "Retained",
        1: "Churned"
    })

    # ------------------------------------------
    # ANALYTICS CHART LAYOUT
    # ------------------------------------------

    st.markdown("### Geographic & Demographic Analysis")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Churn Rate by Geography**")

        fig1 = px.bar(
            geo,
            x="Geography",
            y="Churn_Rate",
            text="Churn_Rate",
            hover_data=["Customers"],
            color="Geography",
            color_discrete_sequence=["#087F8C", "#4C78A8", "#55A7A0"]
        )

        fig1.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig1.update_layout(
            height=350,
            showlegend=False,
            xaxis_title="Geography",
            yaxis_title="Churn Rate (%)",
            yaxis_range=[0, 40],
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**Churn Rate by Gender**")

        fig2 = px.bar(
            gender,
            x="Gender",
            y="Churn_Rate",
            text="Churn_Rate",
            hover_data=["Customers"],
            color="Gender",
            color_discrete_sequence=["#4C78A8", "#087F8C"]
        )

        fig2.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig2.update_layout(
            height=350,
            showlegend=False,
            xaxis_title="Gender",
            yaxis_title="Churn Rate (%)",
            yaxis_range=[0, 32],
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ------------------------------------------
    # CUSTOMER ENGAGEMENT & PRODUCTS
    # ------------------------------------------

    st.markdown("### Customer Behavior Analysis")

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("**Churn by Membership Status**")

        fig3 = px.bar(
            active,
            x="Membership",
            y="Churn_Rate",
            text="Churn_Rate",
            hover_data=["Customers"],
            color="Membership",
            color_discrete_map={
                "Active": "#087F8C",
                "Inactive": "#F08A72"
            }
        )

        fig3.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig3.update_layout(
            height=350,
            showlegend=False,
            xaxis_title="Membership Status",
            yaxis_title="Churn Rate (%)",
            yaxis_range=[0, 35],
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Churn by Number of Products**")

        fig4 = px.bar(
            products,
            x="NumOfProducts",
            y="Churn_Rate",
            text="Churn_Rate",
            hover_data=["Customers"],
            color="Churn_Rate",
            color_continuous_scale=["#A8DADC", "#087F8C"]
        )

        fig4.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig4.update_layout(
            height=350,
            showlegend=False,
            xaxis_title="Number of Products",
            yaxis_title="Churn Rate (%)",
            coloraxis_showscale=False,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ------------------------------------------
    # AGE DISTRIBUTION
    # ------------------------------------------

    st.markdown("### Customer Age Distribution")

    fig5 = px.histogram(
        age_data,
        x="Age",
        color="Customer Status",
        nbins=25,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={
            "Retained": "#087F8C",
            "Churned": "#F08A72"
        }
    )

    fig5.update_layout(
        height=380,
        xaxis_title="Customer Age",
        yaxis_title="Number of Customers",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text=""
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # ------------------------------------------
    # CORRELATION HEATMAP
    # ------------------------------------------

    st.markdown("### Feature Correlation Analysis")

    numeric_data = data.select_dtypes(include="number")
    correlation = numeric_data.corr()

    fig6 = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    fig6.update_layout(
        height=550,
        paper_bgcolor="white"
    )

    st.plotly_chart(fig6, use_container_width=True)

    st.caption(
        "Correlation measures linear relationships. "
        "It does not establish causation."
    )

    st.divider()

    # ------------------------------------------
    # BUSINESS INSIGHTS
    # ------------------------------------------

    st.markdown("### Business Insights & Recommendations")

    st.caption(
        "Key findings from historical customer data and "
        "their potential business implications."
    )

    insight1, insight2 = st.columns(2, gap="large")

    with insight1:

        st.markdown("""
        <div class="info-card">
            <div style="color:#087F8C;font-size:12px;font-weight:700;">
                GEOGRAPHIC RISK
            </div>
            <h2 style="margin:8px 0;">32.44%</h2>
            <p><b>Germany has the highest churn rate</b> among
            the three geographical segments.</p>
            <hr>
            <p><b>Recommendation:</b> Investigate customer
            satisfaction, product preferences and retention
            strategies in Germany.</p>
        </div>

        <div class="info-card">
            <div style="color:#087F8C;font-size:12px;font-weight:700;">
                CUSTOMER ENGAGEMENT
            </div>
            <h2 style="margin:8px 0;">26.85%</h2>
            <p><b>Inactive members show higher churn</b> than
            active members.</p>
            <hr>
            <p><b>Recommendation:</b> Consider personalized
            engagement campaigns and account activity programs.</p>
        </div>
        """, unsafe_allow_html=True)

    with insight2:

        st.markdown("""
        <div class="info-card">
            <div style="color:#087F8C;font-size:12px;font-weight:700;">
                CUSTOMER DEMOGRAPHICS
            </div>
            <h2 style="margin:8px 0;">25.07%</h2>
            <p><b>Female customers show a higher churn rate</b>
            than male customers in this dataset.</p>
            <hr>
            <p><b>Recommendation:</b> Investigate differences in
            customer needs and experiences before designing
            retention initiatives.</p>
        </div>

        <div class="info-card">
            <div style="color:#087F8C;font-size:12px;font-weight:700;">
                CUSTOMER AGE
            </div>
            <h2 style="margin:8px 0;">44.84 Years</h2>
            <p><b>Average age of churned customers</b> compared
            with 37.41 years for retained customers.</p>
            <hr>
            <p><b>Recommendation:</b> Study age-related customer
            preferences and develop suitable retention strategies.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Product observation: Customers with 3 or 4 products "
        "show very high churn rates. However, these groups have "
        "small sample sizes, so their results should be interpreted "
        "with caution."
    )

    st.caption(
        "Business recommendations are exploratory suggestions, "
        "not proven causes of customer churn."
    )


# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------

elif page == "Model Insights":

    page_header(
        "MODEL INTELLIGENCE",
        "Model Performance",
        "Evaluation results of the selected machine learning model."
    )

    st.markdown("""
    <div class="info-card">
        <div style="color:#087F8C;font-size:12px;
        font-weight:700;letter-spacing:1px;">
            SELECTED MODEL
        </div>
        <h2 style="margin:8px 0;">XGBoost Classifier</h2>
        <p style="margin:0;">Binary classification · Customer churn prediction</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "85.65%")
    col2.metric("Precision", "65.87%")
    col3.metric("Recall", "61.18%")

    col4, col5, col6 = st.columns(3)

    col4.metric("F1 Score", "63.44%")
    col5.metric("ROC-AUC", "86.82%")
    col6.metric("Threshold", "0.35")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Model Configuration")

        st.markdown("""
        <div class="info-card">
            <p><b>Algorithm</b><br>XGBoost</p>
            <hr>
            <p><b>Problem Type</b><br>Binary Classification</p>
            <hr>
            <p><b>Target Variable</b><br>Exited</p>
            <hr>
            <p><b>Dataset Size</b><br>10,000 Customers</p>
            <hr>
            <p><b>Models Evaluated</b><br>7</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("### Threshold Optimization")

        st.markdown("""
        <div class="info-card">
            <div style="font-size:13px;color:#718096;">
                SELECTED DECISION THRESHOLD
            </div>
            <div style="font-size:38px;font-weight:750;
            color:#087F8C;margin:8px 0;">0.35</div>
            <p>
                The threshold was selected using cross-validation
                to improve the F1-score and balance precision
                with recall.
            </p>
            <hr>
            <p style="font-size:13px;">
                A lower threshold increases the model's sensitivity
                to potential churn, while potentially increasing
                false-positive predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Accuracy, precision, recall and F1-score are from the "
        "final test evaluation at threshold 0.35. ROC-AUC is "
        "calculated from prediction probabilities."
    )


    # ------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------
    
    st.divider()
    
    st.markdown("### Feature Importance")
    st.caption(
        "Relative importance of customer features "
        "used by the trained XGBoost model."
    )
    
    # Extract model components
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]
    
    # Get feature names and importance scores
    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    
    # Clean feature names
    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    
    # Convert to percentage
    importance_df["Importance"] *= 100
    
    # Sort features
    importance_df = importance_df.sort_values(
        "Importance", ascending=True
    )
    
    # Create horizontal bar chart
    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color_discrete_sequence=["#087F8C"]
    )
    
    fig_importance.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        textfont=dict(color="#1E293B", size=13),
        cliponaxis=False
    )
    
    fig_importance.update_layout(
        height=550,
        xaxis_title="Relative Importance (%)",
        yaxis_title="",
        font=dict(
            family="Arial",
            size=14,
            color="#1E293B"
        ),
        xaxis=dict(
            title_font=dict(color="#1E293B", size=15),
            tickfont=dict(color="#334155", size=13),
            showgrid=True,
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            tickfont=dict(color="#1E293B", size=13)
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=70, t=20, b=20),
        xaxis_range=[
            0,
            importance_df["Importance"].max() * 1.25
        ]
    )
    
    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )
    
    st.info(
        "Age, Number of Products and Active Membership "
        "are the three highest-ranked features. "
        "Feature importance indicates model reliance, "
        "not causation."
    )

    # ------------------------------------------
    # MODEL LIMITATIONS
    # ------------------------------------------

    st.divider()

    st.markdown("### Model Limitations")

    st.markdown("""
    - **Class Imbalance:** Only 20.37% of customers in the dataset
      belong to the churn class.

    - **Dataset Limitation:** The model is trained on a sample
      dataset and has not been validated on live banking data.

    - **Threshold Trade-off:** The 0.35 threshold improves churn
      detection but may increase false positive predictions.

    - **Feature Interpretation:** Feature importance shows
      model reliance, not cause-and-effect relationships.

    - **Generalization:** Model performance may differ when
      applied to customers from other banks or time periods.
    """)
# --------------------------------------------------
# FOOTER
# --------------------------------------------------


def page_footer():
    st.markdown("""
        <div class="page-footer">
            <div class="footer-name">Tamanna Saraf</div>
            <div class="footer-role">Data Scientist</div>
            <div class="footer-project">
                ChurnIQ | Customer Intelligence Platform
            </div>
        </div>
    """, unsafe_allow_html=True)


# Display footer on every page
page_footer()