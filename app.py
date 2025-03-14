import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Step 1: Load Dataset from User Input
st.title("Zepto Sales Analysis")
file = st.file_uploader("Upload a CSV, XLSX, XLS, or TXT file", type=["csv", "xlsx", "xls", "txt"])

if file is not None:
    file_extension = file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(file)
    elif file_extension in ["xls", "xlsx"]:
        df = pd.read_excel(file)
    elif file_extension == "txt":
        df = pd.read_csv(file, delimiter="\t")  # Assuming tab-separated values
    else:
        st.error("Unsupported file format")
        st.stop()
    
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Available columns:", df.columns.tolist())
    
    df.drop(columns=['Row ID', 'Postal Code'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    # Step 2: Exploratory Data Analysis
    st.subheader("Data Visualizations")
    if 'Order Date' in df.columns and 'Sales' in df.columns:
        plt.figure(figsize=(10,5))
        sns.lineplot(x='Order Date', y='Sales', data=df)
        plt.title('Sales Over Time')
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.warning("Column 'Order Date' or 'Sales' not found. Skipping sales over time visualization.")
    
    if 'Category' in df.columns and 'Sales' in df.columns:
        plt.figure(figsize=(8,5))
        sns.barplot(x='Category', y='Sales', data=df, estimator=np.sum)
        plt.title('Sales by Category')
        st.pyplot(plt)
    else:
        st.warning("Column 'Category' or 'Sales' not found. Skipping category-based visualization.")
    
    if 'Sales' in df.columns and 'Profit' in df.columns:
        plt.figure(figsize=(8,5))
        sns.scatterplot(x='Sales', y='Profit', data=df)
        plt.title('Profit vs. Sales')
        st.pyplot(plt)
    else:
        st.warning("Columns 'Sales' or 'Profit' not found. Skipping scatter plot.")
    
    # Step 3: Model Training
    numeric_cols = ['Quantity', 'Discount', 'Profit', 'Shipping Cost']
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
    categorical_cols = [col for col in df.columns if 'Category' in col or 'Segment' in col or 'Region' in col]
    feature_cols = available_numeric_cols + categorical_cols
    
    if feature_cols and 'Sales' in df.columns:
        X = df[feature_cols]
        y = df['Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) * 100
        
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}%")
        
        if r2 < 75:
            st.warning("Model accuracy is low. Feature selection improved, but you can further refine it.")
        elif r2 >= 100:
            st.warning("Model might be overfitting. Consider reducing complexity.")
        else:
            st.success("Model accuracy is good.")

        # Step 4: Streamlit GUI for Prediction
        st.subheader("Superstore Sales Prediction")
        input_features = {}
        for col in available_numeric_cols:
            input_features[col] = st.number_input(f"Enter {col}", value=0.0)
        for col in categorical_cols:
            input_features[col] = int(st.checkbox(f"{col}"))
        
        if st.button("Predict Sales"):
            prediction = model.predict([list(input_features.values())])
            st.write(f"Predicted Sales: ${prediction[0]:.2f}")
    else:
        st.warning("Required columns not found. Skipping model training and prediction.")
