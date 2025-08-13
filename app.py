import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime

# Page config
st.set_page_config(page_title="Weather Prediction App", page_icon="🌦", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .highlight-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(0, 114, 255, 0.1);
        border: 1px solid rgba(0, 114, 255, 0.4);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and training columns
model = pickle.load(open("model.pkl", "rb"))
train_columns = pickle.load(open("train_columns.pkl", "rb"))

# Load dataset
df = pd.read_csv("data/weather_classification_data.csv")

# Sidebar menu
menu = ["Home", "Data Exploration", "Visualisations", "Prediction", "Model Performance", "About"]
choice = st.sidebar.selectbox("📂 Menu", menu)

#  HOME PAGE
if choice == "Home":
    st.markdown('<p class="big-font">🌦 Weather Prediction App</p>', unsafe_allow_html=True)
    st.write("Predict weather types like **Sunny**, **Rainy**, **Cloudy**, and more using real meteorological data.")

    st.markdown('<div class="highlight-box">✅ Explore the dataset easily<br>📊 Visualise patterns and insights<br>🤖 Predict weather with AI models<br>📈 View model performance and metrics</div>', unsafe_allow_html=True)

    # Dataset stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📄 Dataset Rows", value=df.shape[0])
    with col2:
        st.metric(label="🔢 Features", value=df.shape[1])
    with col3:
        st.metric(label="📅 Last Updated", value=datetime.now().strftime("%d %B %Y"))

    # Fun facts
    weather_facts = [
        "🌦 The highest recorded temperature on Earth was 56.7°C in Death Valley, USA.",
        "❄️ The largest snowflake ever recorded was 15 inches wide!",
        "🌪 The fastest wind speed ever recorded was 407 km/h during a cyclone in Australia.",
        "🌈 Did you know? Rainbows can occur at night — they are called 'moonbows'!",
        "☁️ Clouds can weigh millions of kilograms but still float."
    ]
    st.info(random.choice(weather_facts))

# DATA EXPLORATION
elif choice == "Data Exploration":
    st.markdown('<h2 style="color:#00c6ff;">🔍 Data Exploration</h2>', unsafe_allow_html=True)
    st.write("Explore the dataset, view sample records, and filter data interactively.")

    # Dataset shape & info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Total Rows", df.shape[0])
    with col2:
        st.metric("🔢 Total Columns", df.shape[1])
    with col3:
        st.metric("❓ Missing Values", int(df.isnull().sum().sum()))

    # Dataset preview
    st.markdown("### Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    # Column selector
    st.markdown("### Select Columns to View")
    columns = st.multiselect("Choose columns", list(df.columns), default=list(df.columns))
    
    if columns: 
        st.dataframe(df[columns].head(20), use_container_width=True)
    else:
        st.warning("⚠ Please select at least one column to display.")

    # Numeric column filter
    numeric_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)
    if numeric_cols:
        selected_col = st.selectbox("📊 Filter by numeric column", numeric_cols)
        min_val, max_val = float(df[selected_col].min()), float(df[selected_col].max())
        filter_range = st.slider(
            f"Select range for {selected_col}",
            min_val, max_val, (min_val, max_val)
        )
        filtered_df = df[(df[selected_col] >= filter_range[0]) & (df[selected_col] <= filter_range[1])]

        st.markdown(f"### 📄 Filtered Data ({len(filtered_df)} rows)")
        st.dataframe(filtered_df, use_container_width=True)

        # Quick visualization
        st.markdown("### Quick Distribution Plot")
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], bins=20, kde=True, color="#00c6ff", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠ No numeric columns found in the dataset.")


# DATA VISUALISATION
elif choice == "Visualisations":
    st.subheader("Weather Data Visualisations")

    #Bar Chart - Select feature to compare by season
    st.markdown("### Average Values by Season")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_bar = st.selectbox("Select a numeric feature", num_cols)
    avg_by_season = df.groupby("Season")[feature_bar].mean().reset_index()

    fig1, ax1 = plt.subplots()
    sns.barplot(data=avg_by_season, x="Season", y=feature_bar, palette="viridis", ax=ax1)
    ax1.set_title(f"Average {feature_bar} by Season", fontsize=14)
    st.pyplot(fig1)

    st.markdown("---")

    #Line Chart - Filterable by Season or Location
    st.markdown("### Line Chart: Temperature vs Atmospheric Pressure")
    season_filter = st.multiselect("Select Season(s)", df["Season"].unique(), default=df["Season"].unique())
    location_filter = st.multiselect("Select Location(s)", df["Location"].unique(), default=df["Location"].unique())

    filtered_df = df[(df["Season"].isin(season_filter)) & (df["Location"].isin(location_filter))]

    fig2, ax2 = plt.subplots()
    sns.lineplot(data=filtered_df, x="Temperature", y="Atmospheric Pressure", hue="Season", palette="coolwarm", ax=ax2)
    ax2.set_title("Temperature vs Atmospheric Pressure", fontsize=14)
    st.pyplot(fig2)

    st.markdown("---")

    #Correlation Heatmap
    st.markdown("### Feature Correlation Heatmap")
    if st.checkbox("Show Correlation Heatmap"):
        corr = df.select_dtypes(include=['float64', 'int64']).corr()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", ax=ax3)
        st.pyplot(fig3)


# PREDICTION
elif choice == "Prediction":
    st.subheader("Weather Prediction")
    st.write("Enter a few details and click **Predict** to see the weather type.")

    # Simple user inputs
    temperature = st.number_input("🌡 Temperature (°C)", -10.0, 50.0, step=0.1)
    humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
    wind_speed = st.number_input("💨 Wind Speed (km/h)", 0.0, 100.0, step=0.5)
    season = st.selectbox("🍂 Season", df["Season"].unique())
    location = st.selectbox("📍 Location", df["Location"].unique())

    if st.button("Predict"):
        try:
            # Fill missing inputs with defaults from dataset
            default_values = df.mode().iloc[0]  
            default_values.update(df.mean(numeric_only=True))  
            input_data = default_values.to_dict()

            # Override with user inputs
            input_data.update({
                "Temperature": temperature,
                "Humidity": humidity,
                "Wind Speed": wind_speed,
                "Season": season,
                "Location": location
            })

            # Create DataFrame
            input_df = pd.DataFrame([input_data])

            # ===== Feature Engineering (match training) =====
            wind_median = df["Wind Speed"].median()
            input_df["High_Precip"] = (input_df["Precipitation (%)"] > 50).astype(int)
            input_df["Windy"] = (input_df["Wind Speed"] > wind_median).astype(int)
            input_df["Temp_Humid"] = input_df["Temperature"] * input_df["Humidity"]

            #One-hot encoding 
            input_encoded = pd.get_dummies(input_df)

            #Align columns with model's training features
            input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

            #Predict
            prediction = model.predict(input_encoded)[0]
            confidence = model.predict_proba(input_encoded).max() * 100

            st.success(f"**Predicted Weather Type:** {prediction}")
            st.info(f"Prediction Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Error during prediction: {e}")


# MODEL PERFORMANCE
elif choice == "Model Performance":
    st.subheader("Model Performance")
    st.markdown("This section shows how well our trained models performed on the test data.")

    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import pandas as pd
    import pickle

    df = pd.read_csv("data/weather_classification_data.csv")
    X = df.drop("Weather Type", axis=1)
    y = df["Weather Type"]

    X_encoded = pd.get_dummies(X)

    train_columns = pickle.load(open("train_columns.pkl", "rb"))
    X_encoded = X_encoded.reindex(columns=train_columns, fill_value=0)

    best_model = pickle.load(open("model.pkl", "rb"))

    try:
        log_reg_model = pickle.load(open("log_reg_model.pkl", "rb"))
        rf_model = pickle.load(open("rf_model.pkl", "rb"))
    except:
        log_reg_model = rf_model = None

    # Evaluation Metrics (Best Model)
    y_pred_best = best_model.predict(X_encoded)
    report = classification_report(y, y_pred_best, output_dict=True)
    st.markdown("### 📊 Evaluation Metrics (Best Model)")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    # Confusion Matrix 
    st.markdown("### 🔍 Confusion Matrix")
    cm = confusion_matrix(y, y_pred_best, labels=best_model.classes_)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    # Model Comparison 
    if log_reg_model and rf_model:
        st.markdown("Model Comparison")
        y_pred_lr = log_reg_model.predict(X_encoded)
        y_pred_rf = rf_model.predict(X_encoded)
        acc_lr = (y_pred_lr == y).mean()
        acc_rf = (y_pred_rf == y).mean()
        st.table(pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [acc_lr, acc_rf]
        }))
    else:
        st.info("⚠ Model comparison is not available because both models were not saved separately.")


# ABOUT Section
elif choice == "About":
    st.markdown('<h2 style="color:#00c6ff;">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    st.write("""
    This **Weather Prediction App** uses **Machine Learning** to classify weather types such as
    *Sunny*, *Rainy*, *Cloudy*, and *Snowy* based on meteorological data.

    ### How it works:
    - **Data Source:** Historical weather data with features like temperature, humidity, wind speed, precipitation, etc.
    - **Feature Engineering:** Extracts additional indicators such as high precipitation flags and combined temperature-humidity metrics.
    - **Machine Learning Models:** Compares Logistic Regression (baseline) with Random Forest (advanced).
    - **Model Selection:** Chooses the highest accuracy model for predictions.
    - **Deployment:** Interactive app built with **Streamlit**.

    ---
    **Developed with using Python, Scikit-learn, Pandas, Matplotlib, and Streamlit.**
    """)


