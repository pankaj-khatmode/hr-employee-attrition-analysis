import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="HR Employee Attrition Analysis", page_icon="📊")

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
        
        # Load the dataset
        try:
            # Try different paths for the dataset
            possible_paths = [
                "data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
                "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
                "WA_Fn-UseC_-HR-Employee-Attrition.csv"
            ]
            
            dataset_path = None
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    dataset_path = path
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    st.error(f"Error loading dataset from {path}: {str(e)}")
                    return
            
            if dataset_path:
                st.success(f"Dataset loaded successfully from {dataset_path}")
            else:
                st.error("Dataset not found. Please upload the dataset file to the 'data' directory.")
                return
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return
        
        # Convert categorical variables
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
        
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Show basic statistics
        st.subheader("Basic Statistics")
        st.write(df.describe())
        
        # Show attrition distribution
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='Attrition')
        plt.title('Distribution of Attrition')
        st.pyplot(fig)
        
        # Show correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        
        # Show top correlated features
        st.subheader("Top Correlated Features with Attrition")
        attrition_corr = corr['Attrition'].sort_values(ascending=False)
        top_corr = attrition_corr[attrition_corr.abs() > 0.1]
        st.bar_chart(top_corr)

    # Title and description
    st.title("HR Employee Attrition Analysis")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Predictions", "About Dataset"])

    # Main content based on selected page

    if page == "Home":
        st.header("Home")
        st.write("Welcome to the HR Employee Attrition Analysis Dashboard!")
        st.write("This interactive dashboard helps analyze employee attrition patterns and predict future attrition.")
        st.write("The goal is to predict which employees are likely to leave the company based on these features.")
        st.write("Understanding employee attrition patterns can help organizations take proactive measures to retain valuable employees.")
        st.write("Navigate through different sections using the sidebar to explore the data and build predictive models.")

    elif page == "Model Training":
        st.header("Model Training")
        
        # Load and preprocess data
        try:
            df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        except FileNotFoundError:
            st.error("Dataset not found. Please make sure the dataset is in the 'data' directory.")
            return
        
        # Convert categorical variables
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
        
        # Select features
        features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        # Split the data
        X = df[features]
        y = df['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        # Display metrics
        st.write("### Model Performance Metrics")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        
        # Show classification report
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        # Show confusion matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
        # Show feature importance
        st.write("### Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.coef_[0]
        }).sort_values('Importance', ascending=False)
        st.bar_chart(feature_importance.set_index('Feature'))

    elif page == "Predictions":
        st.header("Predict Employee Attrition")
        
        # Load and preprocess data
        try:
            df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        except FileNotFoundError:
            st.error("Dataset not found. Please make sure the dataset is in the 'data' directory.")
            return
        
        # Convert categorical variables
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
        
        # Select features
        features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        # Create input form
        st.write("Enter employee details to predict attrition:")
        
        # Create input fields
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        distance_from_home = st.number_input("Distance from Home", min_value=0, max_value=100, value=10)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=2)
        
        # Create input data
        input_df = pd.DataFrame({
            'Age': [age],
            'DistanceFromHome': [distance_from_home],
            'Education': [education],
            'JobLevel': [job_level],
            'MonthlyIncome': [monthly_income],
            'YearsAtCompany': [years_at_company]
        })
        
        # Train the model
        X = df[features]
        y = df['Attrition']
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Make prediction
        if st.button("Predict Attrition"):
            try:
                # Align input features with training features
                input_features = input_df[features]
                
                # Make prediction
                prediction = model.predict(input_features)
                probability = model.predict_proba(input_features)[:, 1][0]
                
                # Display result
                if prediction[0] == 1:
                    st.error(f"This employee is likely to leave the company.")
                else:
                    st.success(f"This employee is likely to stay with the company.")
                
                st.write(f"Probability of attrition: {probability:.2%}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    elif page == "About Dataset":
        st.header("About the Dataset")
        
        # Load and preprocess data
        try:
            df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        except FileNotFoundError:
            st.error("Dataset not found. Please make sure the dataset is in the 'data' directory.")
            return
        
        # Convert categorical variables
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
        
        # Show dataset information
        st.write("### Dataset Information")
        st.write("This dataset contains employee information and attrition status.")
        st.write("The goal is to predict whether an employee will leave the company based on various features.")
        
        # Show feature descriptions
        st.write("### Feature Descriptions")
        feature_descriptions = {
            'Age': 'Employee age in years',
            'BusinessTravel': 'Travel required for job',
            'Department': 'Employee department',
            'DistanceFromHome': 'Distance from home to work in miles',
            'Education': 'Education level',
            'JobLevel': 'Job level in the organization',
            'MonthlyIncome': 'Monthly income',
            'YearsAtCompany': 'Years of service at the company'
        }
        
        for feature, description in feature_descriptions.items():
            st.write(f"- **{feature}**: {description}")
        
        # Show data statistics
        st.write("### Data Statistics")
        st.write(f"Number of employees: {len(df)}")
        st.write(f"Number of features: {len(df.columns)}")
        st.write(f"Number of employees who left: {df['Attrition'].sum()}")
        st.write(f"Number of employees who stayed: {len(df) - df['Attrition'].sum()}")

if __name__ == "__main__":
    main()

# Run the app
if __name__ == "__main__":
    main()
