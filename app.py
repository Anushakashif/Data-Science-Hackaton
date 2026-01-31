import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load model & features
# ===============================
model = joblib.load("student_performance_model.pkl")
features = joblib.load("model_features.pkl")

label_map = {
    0: "Low Performance",
    1: "Medium Performance",
    2: "High Performance"
}

# ===============================
# Preprocessing (SAME AS TRAINING)
# ===============================
def preprocess_input(df):

    binary_mappings = {
        'Semester': {'F': 0, 'S': 1},
        'Relation': {'Father': 0, 'Mum': 1},
        'ParentAnsweringSurvey': {'No': 0, 'Yes': 1},
        'ParentschoolSatisfaction': {'Bad': 0, 'Good': 1},
        'StudentAbsenceDays': {'Under-7': 0, 'Above-7': 1},
        'gender': {'M': 1, 'F': 0}
    }

    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # StageID encoding
    if 'StageID' in df.columns:
        stage_mapping = {
            'lowerlevel': 0,
            'MiddleSchool': 1,
            'HighSchool': 2
        }
        df['StageID'] = df['StageID'].map(stage_mapping)

    # GradeID encoding
    if 'GradeID' in df.columns:
        df['GradeID'] = (
            df['GradeID']
            .astype(str)
            .str.replace('G-', '', regex=False)
            .astype(int)
        )

    # One-hot encoding
    one_hot_cols = ['NationalITy', 'PlaceofBirth', 'SectionID', 'Topic']
    df = pd.get_dummies(df, columns=one_hot_cols)

    # Fill missing values
    df = df.fillna(0)

    # Align with training features
    df = df.reindex(columns=features, fill_value=0)

    return df

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.write(
    "Predict student academic performance using machine learning models."
)

st.sidebar.header("📌 Input Mode")
mode = st.sidebar.radio(
    "Choose Prediction Type",
    ["Upload CSV File", "Single Student Input"]
)

# ===============================
# CSV UPLOAD MODE
# ===============================
if mode == "Upload CSV File":

    st.subheader("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload student data", type=["csv"])

    if uploaded_file:
        raw_data = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data Preview")
        st.dataframe(raw_data.head())

        processed_data = preprocess_input(raw_data)

        predictions = model.predict(processed_data)

        raw_data["Predicted Performance"] = [
            label_map[p] for p in predictions
        ]

        st.success("✅ Prediction completed successfully")

        st.write("### Prediction Results")
        st.dataframe(raw_data)

        st.download_button(
            "⬇️ Download Predictions",
            raw_data.to_csv(index=False),
            "student_predictions.csv"
        )

# ===============================
# SINGLE STUDENT MODE
# ===============================
else:
    st.subheader("🧑‍🎓 Single Student Prediction")

    user_input = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        with cols[i % 3]:
            user_input[feature] = st.number_input(
                feature, value=0.0
            )

    if st.button("🔍 Predict Performance"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        st.success(
            f"🎯 Predicted Performance Level: **{label_map[prediction]}**"
        )

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(
    "🚀 Hackathon Project | Machine Learning Based Student Performance Prediction"
)
