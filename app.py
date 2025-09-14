import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load dataset
file_path = "IEA Global EV Data 2024.xlsx"
df = pd.read_excel(file_path)

# Drop missing rows
df = df.dropna()

# Define features & target
X = df.drop(columns=['value'])
y = df['value']

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train / Load Model
model_file = "ev_model.pkl"

if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)

# -------------------------------
# Sidebar Navigation (moved up)
# -------------------------------
st.sidebar.title("🔎 Navigation")
nav_choice = st.sidebar.radio(
    "Go to Section:",
    ["Prediction", "Feature Descriptions"],
    index=0  # Default = Prediction
)

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("📊 Dataset & Model Info")

# Dataset summary
st.sidebar.subheader("Dataset Overview")
st.sidebar.write("Brought to you by [Kaggle - IEA Global EV Data](https://www.kaggle.com/datasets/patricklford/global-ev-sales-2010-2024/data).")

if "year" in df.columns:
    st.sidebar.write(f"Year Range: {int(df['year'].min())} - {int(df['year'].max())}")
if "region" in df.columns:
    st.sidebar.write(f"Regions: {df['region'].nunique()}")

# Model performance
st.sidebar.subheader("Model Performance (Test Set)")
y_pred = model.predict(X_test)
st.sidebar.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
st.sidebar.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# Sidebar Visualization
if "region" in df.columns and "year" in df.columns:
    st.sidebar.subheader("📈 Trends by Region")
    regions = ["All"] + sorted(df["region"].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region for Trend", regions)

    if selected_region == "All":
        trend_data = df.groupby("year")["value"].sum().reset_index()
    else:
        trend_data = df[df["region"] == selected_region].groupby("year")["value"].sum().reset_index()

    st.sidebar.line_chart(trend_data.set_index("year"))

# -------------------------------
# Feature Descriptions Setup
# -------------------------------
feature_descriptions = {
    "region": "Country or region name.",
    "category": "Data type (e.g., Historical, Projection).",
    "parameter": "What is being measured (e.g., EV sales, EV stock share).",
    "mode": "Vehicle type (Cars, Buses, Trucks, etc.).",
    "powertrain": "Vehicle drivetrain (EV, BEV, PHEV, FCEV).",
    "year": "Year of observation.",
    "unit": "Unit of measure (Vehicles, Percent, etc.).",
    "value": "Numerical value (target variable).",
    "Acronyms": """
    - **EV** = Electric Vehicle  
    - **BEV** = Battery Electric Vehicle  
    - **PHEV** = Plug-in Hybrid Electric Vehicle  
    - **FCEV** = Fuel Cell Electric Vehicle
    """
}

# -------------------------------
# Main UI
# -------------------------------
st.title("🔋 EV Data Prediction App")
st.write("Predict EV-related values using the IEA Global EV dataset and Random Forest Regression.")

# Show Feature Descriptions
if nav_choice == "Feature Descriptions":
    st.subheader("📖 Feature & Acronym Descriptions")
    for feature, desc in feature_descriptions.items():
        st.markdown(f"**{feature}**: {desc}")
    st.markdown("---")

# Show Prediction UI
elif nav_choice == "Prediction":
    st.subheader("Enter Input Features")
    user_inputs = {}

    # Using categorical values from Excel
    if "region" in df.columns:
        user_inputs["region"] = st.selectbox("Select Region", sorted(df["region"].unique()))
    if "powertrain" in df.columns:
        user_inputs["powertrain"] = st.selectbox("Select Powertrain", sorted(df["powertrain"].unique()))
    if "mode" in df.columns:
        user_inputs["mode"] = st.selectbox("Select Mode", sorted(df["mode"].unique()))
    if "unit" in df.columns:
        user_inputs["unit"] = st.selectbox("Select Unit", sorted(df["unit"].unique()))

    # Numeric (Year)
    if "year" in df.columns:
        min_year, max_year = int(df["year"].min()), int(df["year"].max())
        user_inputs["year"] = st.slider("Select Year", min_year, max_year, int(df["year"].mean()))

    # Prediction
    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_inputs])

            # Applying same encoding as training
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=X.columns, fill_value=0)

            pred = model.predict(input_df)[0]

            st.success(f"✅ Predicted Value: {pred:.2f}")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
