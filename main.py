import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache_data  
def load_data(file_path):
    try:
        dataset = pd.read_csv(file_path)
        return dataset
    except FileNotFoundError:
        st.error("Dataset file not found. Please make sure 't20.csv' is in the same directory as 'main.py'.")
        st.stop()

# Preprocess data
def preprocess_data(dataset):
    # Define features and target
    X = dataset[['overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'venue', 'bat_team', 'bowl_team']]
    y = dataset['total']
    
    # One-Hot Encoding for categorical features
    categorical_features = ['venue', 'bat_team', 'bowl_team']
    numeric_features = ['overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5']
    
    # Define transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.25, random_state=0)
    
    return X_train, X_test, y_train, y_test, preprocessor

# Function to calculate custom accuracy
def custom_accuracy(y_test, y_pred, threshold):
    right = np.sum(np.abs(y_pred - y_test) <= threshold)
    return (right / len(y_pred)) * 100

# Define the Streamlit app
def main():
    st.title("Cricket Match Score Prediction System")
    
    # Load and preprocess data
    dataset = load_data('t20.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(dataset)
    
    # Define and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Button to show the dataset
    if st.button("View Dataset"):
        st.write("### Dataset")
        st.dataframe(dataset)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    custom_acc = custom_accuracy(y_test, y_pred, 20)
    
    # Display model performance metrics in a table
    metrics_df = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Squared Error', 'Custom Accuracy (±20 runs)', 'Predicted Score'],
        'Value': [f"{r2:.2f}", f"{mse:.2f}", f"{custom_acc:.2f}%", "N/A"]
    })
    
    # User inputs
    st.sidebar.header('User Input')
    overs = st.sidebar.number_input('Overs Played', min_value=0.0, max_value=20.0, value=10.0)
    runs = st.sidebar.number_input('Current Runs', min_value=0, max_value=500, value=100)
    wickets = st.sidebar.number_input('Wickets Lost', min_value=0, max_value=10, value=2)
    runs_last_5 = st.sidebar.number_input('Runs in Last 5 Overs', min_value=0, max_value=100, value=50)
    wickets_last_5 = st.sidebar.number_input('Wickets in Last 5 Overs', min_value=0, max_value=10, value=1)
    
    # Input for venue, batting team, and bowling team
    venue = st.sidebar.selectbox('Venue', options=dataset['venue'].unique())
    bat_team = st.sidebar.selectbox('Batting Team', options=dataset['bat_team'].unique())
    bowl_team = st.sidebar.selectbox('Bowling Team', options=dataset['bowl_team'].unique())
    
    # Prepare user input data
    user_input = pd.DataFrame({
        'overs': [overs],
        'runs': [runs],
        'wickets': [wickets],
        'runs_last_5': [runs_last_5],
        'wickets_last_5': [wickets_last_5],
        'venue': [venue],
        'bat_team': [bat_team],
        'bowl_team': [bowl_team]
    })
    
    # Display user input data
    st.write("### User Input")
    st.table(user_input)

    # Transform user input data
    user_input_transformed = preprocessor.transform(user_input)
    
    # Predict the score
    if st.button('Predict Score'):
        try:
            prediction = model.predict(user_input_transformed)
            # Update and display metrics with predicted score
            metrics_df.at[3, 'Value'] = f"{prediction[0]:.2f}"
            st.write("### Model Performance Metrics")
            st.table(metrics_df)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Add footer with stylish underline
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #000;
            color: #f1f1f1;
            text-align: center;
            padding: 15px;
            font-size: 16px;
            box-shadow: 0 -1px 5px rgba(0,0,0,0.3);
        }
        .footer a {
            color: #14299c;
            text-decoration: none;
            font-size: 22px;
            font-weight: bold;
            position: relative;
            display: inline-block;
        }
        .footer a::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -5px;
            width: 100%;
            height: 2px;
            background: #14299c;
            transform: scaleX(0);
            transform-origin: bottom right;
            transition: transform 0.3s ease-out;
        }
        .footer a:hover::after {
            transform: scaleX(1);
            transform-origin: bottom left;
        }
        </style>
        <div class="footer">
            Created By <a href="https://dhruvpatelofficial.vercel.app" target="_blank">Dhruv Patel</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
