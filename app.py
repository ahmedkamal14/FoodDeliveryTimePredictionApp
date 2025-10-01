import streamlit as st
import numpy as np
import joblib
import datetime
import calendar
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import Geocoder

# --- Helper Functions ---

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r

def get_date_features(dt):
    """
    Extracts date-based features from a datetime object.
    """
    day = dt.day
    month = dt.month
    year = dt.year
    day_of_week_num = dt.weekday()  # Monday is 0, Sunday is 6
    day_of_week_name = dt.strftime('%A') # Get the full day name e.g., "Monday"
    quarter = (month - 1) // 3 + 1
    
    is_weekend = 1 if day_of_week_num >= 5 else 0
    is_month_start = 1 if day == 1 else 0
    _, num_days_in_month = calendar.monthrange(year, month)
    is_month_end = 1 if day == num_days_in_month else 0
    
    is_quarter_start = 1 if day == 1 and month in [1, 4, 7, 10] else 0
    quarter_end_month_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    q_end_month, q_end_day = quarter_end_month_day[quarter]
    # Adjust for leap years for Q1
    if calendar.isleap(year) and quarter == 1:
        q_end_day = 29
    is_quarter_end = 1 if month == q_end_month and day == q_end_day else 0
    
    is_year_start = 1 if month == 1 and day == 1 else 0
    is_year_end = 1 if month == 12 and day == 31 else 0

    return {
        'Month': month, 'Day_of_week_Name': day_of_week_name, 'day': day, 'quarter': quarter, 'year': year,
        'is_month_start': is_month_start, 'is_month_end': is_month_end, 'is_quarter_start': is_quarter_start,
        'is_quarter_end': is_quarter_end, 'is_year_start': is_year_start, 'is_year_end': is_year_end,
        'is_weekend': is_weekend
    }

# --- Load Model and Scaler ---
# Corrected file paths for consistency
MODEL_DIR = 'Models'
MODEL_PATH = os.path.join(MODEL_DIR, 'bestModel.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'sclaer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error(f"Error: Model or scaler not found. Please make sure 'model.pkl' and 'scaler.pkl' are in the '{MODEL_DIR}' directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model/scaler: {e}")
    st.stop()

# --- Initialize Session State for Coordinates ---
if 'restaurant_coords' not in st.session_state:
    st.session_state.restaurant_coords = None
if 'delivery_coords' not in st.session_state:
    st.session_state.delivery_coords = None

# --- Streamlit App UI ---
st.set_page_config(page_title="Delivery Time Predictor", page_icon="🛵", layout="wide")

st.title("🛵 Food Delivery Time Prediction")
st.markdown("Enter the details below, provide locations, and click 'Predict'.")

# --- Input Mappings (UPDATED TO MATCH YOUR ENCODERS) ---
weather_options = ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy']
traffic_options = ['High', 'Jam', 'Low', 'Medium']
order_type_options = ['Buffet', 'Drinks', 'Meal', 'Snack']
vehicle_type_options = ['bicycle', 'electric_scooter', 'motorcycle', 'scooter']
festival_options = ['No', 'Yes']
city_options = ['Metropolitan', 'Semi-Urban', 'Urban']
day_of_week_options = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']

# --- Location Input Section ---
st.header("Location Details")
input_method = st.radio("How would you like to enter locations?", ("Select on Map", "Enter Manually"), horizontal=True)

# Initialize coordinate variables
restaurant_latitude, restaurant_longitude = None, None
delivery_location_latitude, delivery_location_longitude = None, None

if input_method == "Select on Map":
    st.markdown("Use the search bar to find a location, then click the map to drop a pin.")
    map_col, coords_col = st.columns([5, 1])
    with map_col:
        map_center = [28.6139, 77.2090]
        m = folium.Map(location=map_center, zoom_start=11)
        Geocoder().add_to(m)
        if st.session_state.restaurant_coords:
            folium.Marker(st.session_state.restaurant_coords, popup="Restaurant", icon=folium.Icon(color='blue', icon='cutlery', prefix='fa')).add_to(m)
        if st.session_state.delivery_coords:
            folium.Marker(st.session_state.delivery_coords, popup="Delivery", icon=folium.Icon(color='green', icon='home', prefix='fa')).add_to(m)
        map_data = st_folium(m, width='100%', height=450)

    with coords_col:
        point_to_select = st.radio("Location to select:", ("Restaurant", "Delivery"), key="location_selector")
        st.write("**Restaurant Coordinates:**", st.session_state.restaurant_coords)
        st.write("**Delivery Coordinates:**", st.session_state.delivery_coords)

    if map_data and map_data['last_clicked']:
        lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
        if point_to_select == "Restaurant":
            st.session_state.restaurant_coords = [lat, lon]
        else:
            st.session_state.delivery_coords = [lat, lon]
        st.rerun()
    
    # Set coordinates for prediction from session state
    if st.session_state.restaurant_coords:
        restaurant_latitude, restaurant_longitude = st.session_state.restaurant_coords
    if st.session_state.delivery_coords:
        delivery_location_latitude, delivery_location_longitude = st.session_state.delivery_coords

else: # Manual Entry
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        st.subheader("Restaurant Location")
        restaurant_latitude = st.number_input("Restaurant Latitude", value=28.686273, format="%.6f", key="rest_lat")
        restaurant_longitude = st.number_input("Restaurant Longitude", value=77.221783, format="%.6f", key="rest_lon")
    with loc_col2:
        st.subheader("Delivery Location")
        delivery_location_latitude = st.number_input("Delivery Latitude", value=28.512688, format="%.6f", key="del_lat")
        delivery_location_longitude = st.number_input("Delivery Longitude", value=77.209123, format="%.6f", key="del_lon")


# --- Input Form ---
with st.form("delivery_form"):
    st.header("Delivery Details")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Personnel & Vehicle")
        delivery_person_age = st.slider("Delivery Person's Age", 18, 50, 25)
        delivery_person_ratings = st.slider("Delivery Person's Rating", 0.0, 5.0, 4.5, 0.1)
        vehicle_condition = st.slider("Vehicle Condition", 0, 3, 2, help="0: Worst, 3: Best")
        multiple_deliveries = st.slider("Number of Multiple Deliveries", 0, 5, 1)
        type_of_vehicle = st.selectbox("Type of Vehicle", vehicle_type_options)

    with col2:
        st.subheader("Order & Conditions")
        type_of_order = st.selectbox("Type of Order", order_type_options)
        prep_time_min = st.slider("Order Preparation Time (minutes)", 5, 60, 20)
        weatherconditions = st.selectbox("Weather Conditions", weather_options)
        road_traffic_density = st.selectbox("Road Traffic Density", traffic_options)
        festival = st.selectbox("Is it a Festival?", festival_options)
        city = st.selectbox("City Type", city_options)

    submitted = st.form_submit_button("Predict Delivery Time")

# --- Prediction Logic ---
if submitted:
    # Validate that both locations have been provided
    if not all([restaurant_latitude, restaurant_longitude, delivery_location_latitude, delivery_location_longitude]):
        st.warning("Please provide both a restaurant and a delivery location.")
    else:
        # 2. Calculate derived features
        delivery_distance_km = haversine(restaurant_longitude, restaurant_latitude, delivery_location_longitude, delivery_location_latitude)
        date_features = get_date_features(datetime.datetime.now())

        # 3. Encode categorical inputs
        weather_map = {v: i for i, v in enumerate(weather_options)}
        traffic_map = {v: i for i, v in enumerate(traffic_options)}
        order_type_map = {v: i for i, v in enumerate(order_type_options)}
        vehicle_type_map = {v: i for i, v in enumerate(vehicle_type_options)}
        festival_map = {'No': 0, 'Yes': 1}
        city_map = {v: i for i, v in enumerate(city_options)}
        
        day_of_week_encoded = day_of_week_options.index(date_features['Day_of_week_Name'])

        weather_encoded = weather_map[weatherconditions]
        traffic_encoded = traffic_map[road_traffic_density]
        order_type_encoded = order_type_map[type_of_order]
        vehicle_type_encoded = vehicle_type_map[type_of_vehicle]
        festival_encoded = festival_map[festival]
        city_encoded = city_map[city]

        # 4. Assemble the feature vector
        feature_vector = [
            delivery_person_age, delivery_person_ratings,
            restaurant_latitude, restaurant_longitude,
            delivery_location_latitude, delivery_location_longitude,
            weather_encoded, traffic_encoded, vehicle_condition,
            order_type_encoded, vehicle_type_encoded, multiple_deliveries,
            festival_encoded, city_encoded,
            date_features['Month'], day_of_week_encoded, prep_time_min,
            delivery_distance_km, date_features['day'], date_features['quarter'],
            date_features['year'],day_of_week_encoded, date_features['is_month_start'],
            date_features['is_month_end'], date_features['is_quarter_start'],
            date_features['is_quarter_end'], date_features['is_year_start'],
            date_features['is_year_end'], date_features['is_weekend']
        ]
        
        # 5. Scale and Predict
        try:
            input_data = np.array(feature_vector).reshape(1, -1)
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)
            
            st.balloons()
            st.success(f"**Estimated Delivery Time: {prediction[0]:.2f} minutes**")
            
            with st.expander("Show Prediction Details"):
                st.write("Calculated Distance:", f"{delivery_distance_km:.2f} km")
                st.write("Date Features:", date_features)
                st.write("Raw Input Vector:", feature_vector)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

