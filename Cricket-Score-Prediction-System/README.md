# Cricket Match Score Prediction System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [URLs](#urls)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
The Cricket Match Score Prediction System is a machine learning application designed to predict the score of a cricket match based on various inputs such as overs played, current runs, wickets lost, and more. This application uses a Random Forest Regressor model for prediction and provides a user-friendly interface using Streamlit.

## Features
- Predicts the final score of a cricket match based on user inputs.
- Displays model performance metrics including R-squared, Mean Squared Error, Mean Absolute Error, and Root Mean Squared Error.
- Visualizes the actual vs. predicted scores.
- Interactive and user-friendly web interface.
- Footer with a stylish, animated underline effect for the creator's portfolio link.

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/dhruvpatel16120/cricket-score-prediction.git
    cd cricket-score-prediction
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure the dataset file is available:**
    Place your `ipl.csv` file in the appropriate directory.

## Usage
1. **Run the Streamlit app:**
    ```bash
    streamlit run Cricket-Score-Prediction.py
    ```

2. **Open your browser:**
    Navigate to `http://localhost:8501` to interact with the application.

## Model
The model used for prediction is a Random Forest Regressor. The following steps are performed during the model training and prediction:

1. **Data Preprocessing:**
    - One-Hot Encoding for categorical features.
    - Standard Scaling for numerical features.
    - Polynomial Features to capture interactions.

2. **Model Training:**
    - Hyperparameter tuning using Grid Search CV.
    - Model evaluation using metrics such as R-squared, Mean Squared Error, and more.

## URLs
The application is deployed and can be accessed at the following URLs:
- [Website](https://cricketscorepredictionsystem.streamlit.app/)

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. **Fork the repository.**
2. **Create a new branch:**
    ```bash
    git checkout -b feature-branch
    ```
3. **Make your changes and commit them:**
    ```bash
    git commit -m 'Add new feature'
    ```
4. **Push to the branch:**
    ```bash
    git push origin feature-branch
    ```
5. **Create a pull request.**

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Created by Dhruv Patel - [Portfolio](https://dhruvpatelofficial.vercel.app)

If you have any questions or suggestions, feel free to reach out!
