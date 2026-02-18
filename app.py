import streamlit as st
import joblib
import numpy as np

# ----- Add Background Color
st.markdown("""
<style>
.stApp {
background-image:
url("https://images.unsplash.com/photo-1544551763-7ef4200f4b8d");
    background-size: cover;
    background-attachement: fixed;
 }
</style>
""", unsafe_allow_html=True)

# ----- Define Lists
fish_list = ["Anchovy", "Catfish", "Dagaa", "Kuhe", "Mackerel", "Perch", "Sangara", "Sardine", "Snapper", "Tilapia", "Tuna"] # 11 items
weather_list = ["Cloudy", "Dry", "Rainy", "Sunny"] # 4 items
season_list = ["Spring", "Summer", "Autumn", "Winter"] # 4 items

# ----- Load model
@st.cache_resource
def get_model():
#----- Ensure you are loading the .joblib file, not the .ipynb
    return joblib.load("ml_project.joblib")

try:
    model =get_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")

st.title(" 🐟 FISH CATCH PREDICTION SYSTEM")
st.write("Fill your input to get prediction.")

col1, col2 = st.columns(2)
with col1:
    selected_fish = st.selectbox("Fish Type:", fish_list)
    selected_weather = st.selectbox("Weather:", weather_list)
with col2:
    selected_season = st.selectbox("Season:", season_list)
    input_fishermen = st.number_input("Number of fishermen:", min_value=1)

if st.button("Predict Now"):
    try:
# 1. Initialize array with 21 zeros (as you requested)
        data_input = np.zeros((1, 20))

# 2. Map Fish Type (Indices 0 to 10)
        fish_idx = fish_list.index(selected_fish)
        data_input[0, fish_idx] = 1
        
# 3. Map Weather (Indices 11 to 14)
        weather_idx = weather_list.index(selected_weather)
        data_input[0, 11 + weather_idx] = 1
        
# 4. Map Season (Indices 15 to 18)
        season_idx = season_list.index(selected_season)
        data_input[0, 15 + season_idx] = 1
        
# 5. Map Fishermen (Put in index 19 or 20 depending on your training)
        data_input[0, 19] = input_fishermen 

#.....  Prediction
        prediction = model.predict(data_input)
#------Extract value safely (Your logic corrected)
        if hasattr(prediction, "__len__"):
            final_result = prediction[0]
        else:
            final_result = prediction
        
#--------Success Output
        st.success(f"Predicted Catch: {float(final_result):.2f} kg")

        
    except Exception as e:
        st.error(f"Prediction Error: {e}")



    



