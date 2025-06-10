
# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration first
st.set_page_config(
    page_title="HR Employee Attrition Analysis",
    page_icon="ðŸ“Š",
    layout="wide"
)

def main():
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Data Analysis", "Model Training", "Predictions", "About Dataset"]
    )

    if page == "Home":
        st.title("HR Employee Attrition Analysis")
        st.write("Welcome to the HR Employee Attrition Analysis Dashboard!")
        st.write("This interactive dashboard helps analyze employee attrition patterns and predict future attrition.")
        st.write("The goal is to predict which employees are likely to leave the company based on these features.")
        st.write("Understanding employee attrition patterns can help organizations take proactive measures to retain valuable employees.")
        st.write("Navigate through different sections using the sidebar to explore the data and build predictive models.")

    elif page == "Data Analysis":
        st.header("Data Analysis")
        try:
            df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
            st.success("Dataset loaded successfully.")
        except:
            st.error("Failed to load dataset. Please place 'WA_Fn-UseC_-HR-Employee-Attrition.csv' in the project folder.")
            return

        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Basic Statistics")
        st.write(df.describe())

        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Attrition', data=df)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        st.pyplot(fig)

    elif page == "Model Training":
        st.header("Model Training")
        try:
            df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        except:
            st.error("Dataset not found.")
            return

        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes

        features = df.columns.tolist()
        features.remove('Attrition')
        X = df[features]
        y = df['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("### Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
        st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    elif page == "Predictions":
        st.header("Make Predictions")
        try:
            df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        except:
            st.error("Dataset not found.")
            return

        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes

        features = df.columns.tolist()
        features.remove('Attrition')
        X = df[features]
        y = df['Attrition']

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        st.subheader("Enter Employee Details:")
        inputs = {}
        for feature in features:
            if df[feature].dtype in ['int64', 'float64']:
                inputs[feature] = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

        user_df = pd.DataFrame([inputs])
        if st.button("Predict"):
            pred = model.predict(user_df)[0]
            prob = model.predict_proba(user_df)[0][1]
            if pred == 1:
                st.error(f"âš ï¸ Employee is likely to leave. Probability: {prob:.2%}")
            else:
                st.success(f"âœ… Employee is likely to stay. Probability: {1 - prob:.2%}")

    elif page == "About Dataset":
        st.header("About the Dataset")
        st.markdown("""
        - Dataset: **IBM HR Analytics Employee Attrition**
        - Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/IBM+HR+Analytics+Employee+Attrition+%28using+Watson+Analytics%29)
        - Goal: Predict whether an employee will leave the company based on features like job role, satisfaction, income, etc.
        """)

if __name__ == "__main__":
    main()