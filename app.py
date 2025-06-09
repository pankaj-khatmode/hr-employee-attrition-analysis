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

# Title and description
st.title("HR Employee Attrition Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Predictions", "About Dataset"])

# Main content based on selected page
if page == "Home":
    st.header("Welcome to HR Employee Attrition Analysis")
    st.write("This application helps analyze employee attrition patterns and predict which employees are likely to leave.")
    st.write("Navigate through different sections using the sidebar to explore the data and build predictive models.")

elif page == "About Dataset":
    st.header("About the Dataset")
    st.write("This dataset contains employee information and their attrition status.")
    st.write("Key features include:")
    st.markdown("- **Age**: Age of the employee")
    st.markdown("- **Department**: Department in which the employee works")
    st.markdown("- **Job Role**: Role of the employee")
    st.markdown("- **Monthly Income**: Monthly salary of the employee")
    st.markdown("- **Years at Company**: Number of years the employee has been with the company")
    st.markdown("- **Attrition**: Whether the employee has left the company (Yes/No)")
    
    st.write("The goal is to predict which employees are likely to leave the company based on these features.")
    st.write("Understanding employee attrition patterns can help organizations take proactive measures to retain valuable employees.")
    st.write("Navigate through different sections using the sidebar to explore the data and build predictive models.")

elif page == "Data Analysis":
    st.header("Data Analysis")
    
    # Load the dataset
    df = pd.read_csv("C:\\Users\\Krynorcxy\\Downloads\\archive\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    # Convert categorical variables
    categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
    
    # Display basic information
    st.subheader("Dataset Overview")
    st.write("Number of employees:", len(df))
    st.write("Number of features:", len(df.columns))
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Show basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())
    
    # Visualizations
    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Attrition')
    plt.title('Employee Attrition Distribution')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    # Calculate correlation only for numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show top correlated features
    st.subheader("Top Correlated Features")
    abs_corr = corr.abs()
    top_corr = abs_corr.unstack().sort_values(ascending=False).drop_duplicates()
    st.write("Top 10 most correlated features:")
    st.write(top_corr[:10])

elif page == "Model Training":
    st.header("Model Training")
    
    # Load and preprocess data
    df = pd.read_csv("C:\\Users\\Krynorcxy\\Downloads\\archive\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    # Convert categorical variables
    categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
    
    # Select features and target
    features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    target = 'Attrition'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    if st.button("Train Model"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Show detailed performance metrics
        st.subheader("Model Performance")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        # Display metrics in a table
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': [f'{accuracy:.2f}', f'{precision:.2f}', f'{recall:.2f}', f'{f1:.2f}']
        })
        st.table(metrics_df)
        
        # Show classification report
        report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
        st.text("\nClassification Report:\n")
        st.text(report)
        
        # Show confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df.head(10).plot(kind='barh', x='Feature', y='Importance', ax=ax)
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        st.pyplot(fig)

elif page == "Predictions":
    st.header("Make Predictions")
    
    # Load and preprocess data
    df = pd.read_csv("C:\\Users\\Krynorcxy\\Downloads\\archive\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
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
    input_data = {
        'Age': age,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'JobLevel': job_level,
        'MonthlyIncome': monthly_income,
        'YearsAtCompany': years_at_company
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    # Use only the features we have in the form
    form_features = ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome', 'YearsAtCompany']
    X = df[form_features]
    y = df['Attrition']
    model.fit(X, y)
    
    # Make prediction
    if st.button("Predict Attrition"):
        # Preprocess input data
        for col in categorical_columns[:-1]:  # Exclude Attrition
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category').cat.codes
        
        # Use the same features as in training
        input_features = input_df[form_features]
        
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)[0][1]
        
        if prediction[0] == 1:  # 1 represents 'Yes' after encoding
            st.error(f"This employee is likely to leave. Probability: {probability:.2f}")
        else:
            st.success(f"This employee is likely to stay. Probability: {1 - probability:.2f}")
