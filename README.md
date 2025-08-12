# 🌦 Weather Predictor App

A **machine learning-powered weather classification application** built with **Streamlit**.  
This app predicts **weather type** (Sunny, Rainy, Cloudy, Snowy, etc.) based on real-world weather parameters, allowing interactive data exploration, visualisation, and model performance comparison.

---

## 📌 Features

✅ **Interactive Dashboard** — Navigate between pages:  
- **Home** — Overview and dataset statistics.  
- **Data Exploration** — View dataset records, filter by ranges, inspect columns.  
- **Visualisations** — Bar charts, line charts, and correlation heatmaps.  
- **Prediction** — Input weather parameters and predict the weather type with confidence score.  
- **Model Performance** — View evaluation metrics, confusion matrix, and compare multiple models.

✅ **Multiple ML Models Compared** — Logistic Regression & Random Forest.  
✅ **Beautiful Dark-Themed UI** with gradient headers & styled elements.  
✅ **End-to-End ML Pipeline** — Data preprocessing, feature engineering, model training, evaluation, and deployment.  
✅ **Cloud Deployment** on [Streamlit Cloud](https://prashoharan-weather-predictor-app-e6opfx.streamlit.app/).  

---

## 📂 Project Structure

Weather-Predictor/
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
├── model.pkl # Best trained model (Random Forest)
├── log_reg_model.pkl # Logistic Regression model
├── rf_model.pkl # Random Forest model
├── train_columns.pkl # Feature columns used for training
├── data/
│ └── weather_classification_data.csv # Dataset
├── notebooks/
│ └── model_training.ipynb # Model training & analysis
└── README.md # Project documentation


###  Clone the Repository
```bash
git clone https://github.com/PrashoHaran/weather-predictor.git

### 📊 Machine Learning Workflow

1) Data Loading & Cleaning
    Missing value handling
    Encoding categorical variables
    Feature scaling & engineering

2) Model Training
    Logistic Regression
    Random Forest
    Hyperparameter tuning

3) Evaluation
    Accuracy, Precision, Recall, F1-score
    Confusion Matrix
    Model comparison table

4) Deployment
    Streamlit Cloud deployment with GitHub integration


### 📈 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 21.81%   |
| Random Forest       | 97.60%   |

📌 Random Forest was selected as the best model for deployment.


### 🌍 Live Demo

Try it here: