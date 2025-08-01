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
    df = pd.read_csv("students-scores.csv")
    return df

df = load_data()

# Calculate average score
score_cols = ["math_score", "history_score", "physics_score", "chemistry_score", "biology_score", "english_score", "geography_score"]
df["average_score"] = df[score_cols].mean(axis=1)

# Define pass/fail
df["pass"] = df["average_score"] >= 60  # You can adjust threshold

# Prepare features
X = df.drop(columns=["id", "first_name", "last_name", "email", "average_score", "pass"])
X = pd.get_dummies(X, drop_first=True)
y = df["pass"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Streamlit App ---

st.title("🎓 Student Performance Predictor")
st.markdown("This app predicts whether a student will **pass or fail** based on their data.")

st.subheader("📈 Dataset Overview")
st.dataframe(df.head())

st.subheader("✅ Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

# Plot: Pass vs Fail
fig = px.histogram(df, x="pass", title="Pass vs Fail Count", labels={"pass": "Passed"})
st.plotly_chart(fig, use_container_width=True)

# --- Prediction Form ---
st.subheader("🧠 Predict a New Student's Performance")
input_data = {}

# Collect inputs for features
input_data["gender"] = st.selectbox("Gender", df["gender"].unique())
input_data["part_time"] = st.selectbox("Part Time", [True, False])
input_data["absence_days"] = st.number_input("Absence Days", min_value=0, max_value=100, value=3)
input_data["extracurricular"] = st.selectbox("Extracurricular", [True, False])
input_data["weekly_self_study_hours"] = st.number_input("Weekly Self Study Hours", min_value=0, max_value=100, value=5)
input_data["career_aspiration"] = st.selectbox("Career Aspiration", df["career_aspiration"].unique())

for subject in score_cols:
    input_data[subject] = st.slider(subject.replace("_", " ").title(), min_value=0, max_value=100, value=70)

# Submit and predict
if st.button("Predict Pass/Fail"):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    st.subheader("🎯 Prediction Result")
    st.success("✅ Student is likely to PASS!" if prediction else "❌ Student is likely to FAIL.")

