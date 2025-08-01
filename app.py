import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("student-scores.csv")

df = load_data()

# Define subject columns
score_cols = ["math_score", "history_score", "physics_score", "chemistry_score", "biology_score", "english_score", "geography_score"]

# Calculate average score and label pass/fail
df["average_score"] = df[score_cols].mean(axis=1)
df["pass"] = df["average_score"] >= 60

# Prepare features and target
X = df.drop(columns=["id", "first_name", "last_name", "email", "average_score", "pass"])
X = pd.get_dummies(X, drop_first=True)
y = df["pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Streamlit UI ---

st.title("🎓 Student Performance Predictor")
st.markdown("This app predicts whether a student will **pass or fail** based on their scores and details.")

st.subheader("📊 Dataset Sample")
st.dataframe(df.head())

st.subheader("✅ Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

# Chart: Pass vs Fail
st.subheader("📈 Pass/Fail Distribution")
fig = px.histogram(df, x="pass", title="Pass vs Fail", labels={"pass": "Passed"})
st.plotly_chart(fig, use_container_width=True)

# --- Prediction Form ---
st.subheader("🧠 Predict Student Performance")

input_data = {}

input_data["gender"] = st.selectbox("Gender", df["gender"].unique())
input_data["part_time_job"] = st.selectbox("Part-Time Job", [True, False])
input_data["absence_days"] = st.number_input("Absence Days", min_value=0, max_value=100, value=3)
input_data["extracurricular_activities"] = st.selectbox("Extracurricular Activities", [True, False])
input_data["weekly_self_study_hours"] = st.number_input("Weekly Self Study Hours", min_value=0, max_value=100, value=5)
input_data["career_aspiration"] = st.selectbox("Career Aspiration", df["career_aspiration"].unique())

for subject in score_cols:
    input_data[subject] = st.slider(subject.replace("_", " ").title(), 0, 100, 75)

# Predict button
if st.button("Predict Pass/Fail"):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.subheader("🎯 Prediction Result")
    if prediction:
        st.success("✅ Student is likely to PASS!")
    else:
        st.error("❌ Student is likely to FAIL.")
