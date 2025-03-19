# import streamlit as st
# import joblib  # type: ignore
# import pandas as pd
# import os
# import numpy as np
# import seaborn as sns  # type: ignore
# import matplotlib.pyplot as plt  # type: ignore
# from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore

# # ✅ Set Streamlit Page
# st.set_page_config(page_title="Rain in Australia", page_icon="☔", layout="wide")

# # ✅ Define Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
# DATA_PATH = os.path.join(BASE_DIR, "weatherAUS.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "voting5_pipeline.pkl")
# IMAGE_PATH = os.path.join(BASE_DIR, "Screenshot 2025-03-09 201614.png")

# # ✅ Sidebar Navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "🔍 Prediction"])

# # ✅ Load dataset
# @st.cache_data
# def load_data():
#     if not os.path.exists(DATA_PATH):
#         st.error("❌ Dataset file not found! Check the file path.")
#         return None
#     df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    
#     required_columns = ["WindGustDir", "RainTomorrow", "Location", "Rainfall", 
#                         "WindSpeed9am", "Humidity9am", "Temp9am", "Temp3pm"]
    
#     missing_cols = [col for col in required_columns if col not in df.columns]
#     if missing_cols:
#         st.error(f"❌ Missing required columns: {missing_cols}")
#         return None

#     df.dropna(subset=required_columns, inplace=True)
    
#     df["Season"] = df["Date"].dt.month.map(
#         lambda m: "Winter" if m in [12, 1, 2] else 
#                   "Spring" if m in [3, 4, 5] else 
#                   "Summer" if m in [6, 7, 8] else 
#                   "Autumn"
#     )
#     return df

# df = load_data()

# # ✅ Load Model
# model, scaler, le = None, None, None
# if os.path.exists(MODEL_PATH):
#     try:
#         model, scaler, le = joblib.load(MODEL_PATH)
#         if not hasattr(model, "predict"):
#             st.sidebar.error("❌ Model is not properly trained.")
#             model = None
#     except Exception as e:
#         st.sidebar.error(f"❌ Error loading model: {e}")
#         model = None
# else:
#     st.sidebar.warning("⚠️ No valid model file found.")

# # 📌 **Home Page**
# if page == "🏠 Home":
#     st.title("Rain in Australia 🌧️🌦️")

#     if os.path.exists(IMAGE_PATH):
#         st.image(IMAGE_PATH, width=600)
#     else:
#         st.warning("⚠️ Image file not found!")

#     if df is not None:
#         st.subheader("Dataset Preview")
#         st.dataframe(df.head())
#     else:
#         st.error("🚨 Dataset could not be loaded.")
    
#     st.write("Use the sidebar to navigate to Visualization or Prediction.")

# # 📌 **Visualization Page**
# elif page == "📊 Visualization" and df is not None:
#     st.title("📊 Weather Data Visualization")
#     st.sidebar.title("🔍 Choose Visualization")
    
#     visualization_option = st.sidebar.selectbox("Select an option", [
#         "🌧️ Rainfall Distribution", "📊 Rain Probability by Wind Direction", "🔥 Correlation Heatmap",
#         "💨 Wind Speed vs. Rain Probability", "🌦️ Seasonal Rain Effects"
#     ])
    
#     fig, ax = plt.subplots(figsize=(10, 5))
    
#     if visualization_option == "🌧️ Rainfall Distribution":
#         selected_location = st.sidebar.selectbox("Select Location", df["Location"].unique())
#         sns.histplot(df[df["Location"] == selected_location]["Rainfall"], bins=30, kde=True, color="blue", ax=ax)
#         ax.set_title(f"Rainfall Distribution in {selected_location}")

#     elif visualization_option == "📊 Rain Probability by Wind Direction":
#         df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
#         sns.barplot(x=df["WindGustDir"], y=df["RainTomorrow"], ax=ax, palette="coolwarm")
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

#     elif visualization_option == "🔥 Correlation Heatmap":
#         sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")

#     elif visualization_option == "💨 Wind Speed vs. Rain Probability":
#         sns.boxplot(x=df["RainTomorrow"], y=df["WindSpeed9am"], palette="coolwarm", ax=ax)
#         ax.set_title("Wind Speed in Morning vs. Rain Probability")

#     elif visualization_option == "🌦️ Seasonal Rain Effects":
#         sns.countplot(data=df, x="Season", hue="RainTomorrow", palette="coolwarm", ax=ax)
#         ax.set_title("Seasonal Effect on Rain Tomorrow")
    
#     st.pyplot(fig)

# # ✅ Prediction Page
# elif page == "🔍 Prediction" and model and scaler and le:
#     st.title("🔍 Rain Prediction")
#     st.sidebar.title("🌍 Enter Weather Details")
    
#     unique_locations = df["Location"].unique().tolist() if df is not None else []
#     valid_wind_gust_dirs = sorted(set(df["WindGustDir"].dropna().unique().tolist())) if df is not None else []
    
#     location = st.sidebar.selectbox("🌏 Choose Location", unique_locations)
#     wind_gust_dir = st.sidebar.selectbox("🌬️ Wind Gust Direction", valid_wind_gust_dirs)
#     wind_speed_9am = st.sidebar.slider("🌬️ Wind Speed 9AM (km/h)", 0, 100, 30)
#     humidity_9am = st.sidebar.slider("💧 Humidity 9AM (%)", 0, 100, 50)
#     temp_9am = st.sidebar.slider("🌡️ Temperature 9AM (°C)", -10, 50, 20)
#     temp_3pm = st.sidebar.slider("🌡️ Temperature 3PM (°C)", -10, 50, 25)
#     prev_day_rainfall = st.sidebar.number_input("☔️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
#     rain_today = 1 if st.sidebar.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes" else 0
    
#     # Handle unseen labels safely
#     if location in le.classes_:
#         location_encoded = le.transform([location])[0]
#     else:
#         location_encoded = le.transform([le.classes_[0]])[0]  # Default to first known class
    
#     if wind_gust_dir in le.classes_:
#         wind_gust_encoded = le.transform([wind_gust_dir])[0]
#     else:
#         wind_gust_encoded = le.transform([le.classes_[0]])[0]  # Default to first known class
    
#     input_data = np.array([[location_encoded, wind_gust_encoded, wind_speed_9am, humidity_9am, temp_9am, temp_3pm, prev_day_rainfall, rain_today]])
#     input_data_scaled = scaler.transform(input_data)
    
#     if st.sidebar.button("Submit"):
#         prediction = model.predict(input_data_scaled)
#         result = "☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain."
#         st.success(result)
import streamlit as st
import joblib  # type: ignore
import pandas as pd
import os
import numpy as np
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore

# ✅ Set Streamlit Page
st.set_page_config(page_title="Rain in Australia", page_icon="☔", layout="wide")

# ✅ Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "weatherAUS.csv")
MODEL_PATH = os.path.join(BASE_DIR, "voting5_pipeline.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "Screenshot 2025-03-09 201614.png")

# ✅ Load dataset
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ Dataset file not found!")
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    
    required_columns = ["WindGustDir", "RainTomorrow", "Location", "Rainfall", 
                        "WindSpeed9am", "Humidity9am", "Temp9am", "Temp3pm"]
    
    if not set(required_columns).issubset(df.columns):
        st.error("❌ Missing required columns!")
        return None
    
    df.dropna(subset=required_columns, inplace=True)
    df["Season"] = df["Date"].dt.month.map({1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 
                                              5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 
                                              9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"})
    return df

df = load_data()

# ✅ Load Model
model, scaler, le = None, None, None
if os.path.exists(MODEL_PATH):
    try:
        model, scaler, le = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            st.sidebar.error("❌ Model is not properly trained.")
            model = None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        model = None
else:
    st.sidebar.warning("⚠️ No valid model file found.")

# ✅ Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "🔍 Prediction"])

# 📌 **Home Page**
# 📌 **Home Page**
if page == "🏠 Home":
    st.title("Rain in Australia 🌧️🌦️")
    
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, width=600)

    st.subheader("About the Dataset")
    st.write("""
    **Context**  
    Ever wondered if you should carry an umbrella tomorrow? With this dataset, you can predict next-day rain by training classification models on the target variable **RainTomorrow**.

    **Content**  
    - This dataset comprises about **10 years of daily weather observations** from numerous locations across Australia.
    - **RainTomorrow** is the target variable to predict. It answers the crucial question: **Will it rain the next day? (Yes or No)**.
    - This column is marked 'Yes' if the rain for that day was **1mm or more**.
    """)

    st.subheader("Dataset Preview")
    if df is not None:
        st.dataframe(df.head())
    else:
        st.error("🚨 Dataset could not be loaded.")
    
    st.write("Use the sidebar to navigate to Visualization or Prediction.")


# ✅ Prediction Page
elif page == "🔍 Prediction" and model and scaler and le:
    st.title("🔍 Rain Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("🌏 Choose Location", df["Location"].unique())
        wind_gust_dir = st.selectbox("🌬️ Wind Gust Direction", df["WindGustDir"].unique())
        wind_speed_9am = st.slider("🌬️ Wind Speed 9AM (km/h)", 0, 100, 30)
        humidity_9am = st.slider("💧 Humidity 9AM (%)", 0, 100, 50)
    
    with col2:
        temp_9am = st.slider("🌡️ Temperature 9AM (°C)", -10, 50, 20)
        temp_3pm = st.slider("🌡️ Temperature 3PM (°C)", -10, 50, 25)
        prev_day_rainfall = st.number_input("☔️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
        rain_today = 1 if st.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes" else 0
    
    location_encoded = le.transform([location])[0] if location in le.classes_ else -1
    wind_gust_encoded = le.transform([wind_gust_dir])[0] if wind_gust_dir in le.classes_ else -1
    
    input_data = np.array([[location_encoded, wind_gust_encoded, wind_speed_9am, humidity_9am, temp_9am, temp_3pm, prev_day_rainfall, rain_today]])
    input_data_scaled = scaler.transform(input_data)
    
    if st.button("Submit Prediction"):
        prediction = model.predict(input_data_scaled)
        st.success("☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain.")

# 📌 **Visualization Page**
elif page == "📊 Visualization" and df is not None:
    st.title("📊 Weather Data Visualization")
    visualization_option = st.sidebar.selectbox("Select an option", [
        "🌧️ Rainfall Distribution", "📊 Rain Probability by Wind Direction", "🔥 Correlation Heatmap",
        "💨 Wind Speed vs. Rain Probability", "🌦️ Seasonal Rain Effects"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if visualization_option == "🌧️ Rainfall Distribution":
        selected_location = st.sidebar.selectbox("Select Location", df["Location"].unique())
        sns.histplot(df[df["Location"] == selected_location]["Rainfall"], bins=30, kde=True, ax=ax)
        ax.set_title(f"Rainfall Distribution in {selected_location}")
    
    elif visualization_option == "📊 Rain Probability by Wind Direction":
        df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
        sns.barplot(x="WindGustDir", y="RainTomorrow", data=df, ax=ax, palette="coolwarm")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    
    elif visualization_option == "🔥 Correlation Heatmap":
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    
    elif visualization_option == "💨 Wind Speed vs. Rain Probability":
        sns.boxplot(x="RainTomorrow", y="WindSpeed9am", data=df, ax=ax, palette="coolwarm")
        ax.set_title("Wind Speed in Morning vs. Rain Probability")
    
    elif visualization_option == "🌦️ Seasonal Rain Effects":
        sns.countplot(x="Season", hue="RainTomorrow", data=df, ax=ax, palette="coolwarm")
    
    st.pyplot(fig)