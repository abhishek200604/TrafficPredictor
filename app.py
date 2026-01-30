"""
================================================================================
Urban Traffic Congestion Prediction - Streamlit Dashboard
================================================================================
Author: Traffic Prediction Project
Purpose: Interactive web dashboard for traffic prediction with Folium maps

Features:
- Date, Time, Location selection
- "Major Event" checkbox for crowd prediction
- Traffic volume prediction display
- Color-coded Folium map (Green/Yellow/Red)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
from datetime import datetime, date
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Pune Traffic Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR BETTER UI
# =============================================================================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Traffic level badges */
    .traffic-low {
        background-color: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .traffic-medium {
        background-color: #F59E0B;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .traffic-high {
        background-color: #EF4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Info box */
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOCATION DATA (Must match data_generator.py)
# =============================================================================
LOCATIONS = {
    'FC_Road': {
        'name': 'FC Road (Fergusson College Road)',
        'latitude': 18.5236,
        'longitude': 73.8419,
        'description': 'Popular shopping and college area'
    },
    'JM_Road': {
        'name': 'JM Road (Jungli Maharaj Road)', 
        'latitude': 18.5308,
        'longitude': 73.8475,
        'description': 'Commercial hub with restaurants and offices'
    },
    'University_Circle': {
        'name': 'University Circle',
        'latitude': 18.5528,
        'longitude': 73.8251,
        'description': 'Near Savitribai Phule Pune University'
    },
    'Pune_Station': {
        'name': 'Pune Railway Station',
        'latitude': 18.5285,
        'longitude': 73.8742,
        'description': 'Major railway station with heavy commuter traffic'
    }
}

WEATHER_OPTIONS = ['Clear', 'Cloudy', 'Rainy', 'Foggy']


@st.cache_resource
def load_model():
    """
    Load the trained model from pickle file.
    Uses st.cache_resource to avoid reloading on every interaction.
    """
    try:
        model_package = joblib.load('traffic_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please run `python train_model.py` first.")
        return None


def get_traffic_level(volume: int) -> tuple:
    """
    Categorize traffic volume into levels with colors.
    
    Returns:
        Tuple of (level_name, color_code, emoji)
    """
    if volume < 300:
        return ("Low Traffic", "#10B981", "🟢")  # Green
    elif volume < 600:
        return ("Moderate Traffic", "#F59E0B", "🟡")  # Yellow/Orange
    else:
        return ("High Traffic", "#EF4444", "🔴")  # Red


def create_traffic_map(location_id: str, traffic_volume: int) -> folium.Map:
    """
    Create a Folium map with color-coded marker based on traffic level.
    
    WHY FOLIUM (for examiner):
    - Leaflet.js based, works seamlessly with Streamlit
    - Professional-looking interactive maps
    - Easy to add markers, popups, and custom styling
    
    Args:
        location_id: ID of the selected location
        traffic_volume: Predicted traffic volume
    
    Returns:
        Folium Map object
    """
    location = LOCATIONS[location_id]
    level_name, color, emoji = get_traffic_level(traffic_volume)
    
    # Create base map centered on Pune
    m = folium.Map(
        location=[18.5204, 73.8567],  # Pune center
        zoom_start=13,
        tiles='CartoDB positron'  # Clean, modern map style
    )
    
    # Add marker for selected location
    # Color coding based on traffic level
    if traffic_volume < 300:
        icon_color = 'green'
    elif traffic_volume < 600:
        icon_color = 'orange'
    else:
        icon_color = 'red'
    
    # Create custom popup with traffic info
    popup_html = f"""
    <div style="font-family: Arial; width: 200px;">
        <h4 style="margin: 0; color: #1E3A5F;">{location['name']}</h4>
        <hr style="margin: 5px 0;">
        <p style="margin: 5px 0;"><b>Traffic Volume:</b> {traffic_volume} vehicles/hr</p>
        <p style="margin: 5px 0;"><b>Status:</b> {emoji} {level_name}</p>
        <p style="margin: 5px 0; color: #666; font-size: 0.9em;">{location['description']}</p>
    </div>
    """
    
    folium.Marker(
        location=[location['latitude'], location['longitude']],
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{location['name']}: {traffic_volume} vehicles/hr",
        icon=folium.Icon(color=icon_color, icon='car', prefix='fa')
    ).add_to(m)
    
    # Add circle to show traffic intensity
    folium.Circle(
        location=[location['latitude'], location['longitude']],
        radius=traffic_volume * 0.5,  # Radius proportional to traffic
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.3
    ).add_to(m)
    
    # Add all other locations as smaller markers
    for loc_id, loc_info in LOCATIONS.items():
        if loc_id != location_id:
            folium.Marker(
                location=[loc_info['latitude'], loc_info['longitude']],
                tooltip=loc_info['name'],
                icon=folium.Icon(color='gray', icon='map-marker', prefix='fa')
            ).add_to(m)
    
    return m


def predict_traffic(model_package: dict, location_id: str, hour: int, 
                   day_of_week: int, month: int, weather: str,
                   is_holiday: bool, is_major_event: bool) -> int:
    """
    Make traffic prediction using the trained model.
    
    Args:
        model_package: Loaded model package with model and encoders
        location_id: Selected location ID
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        month: Month (1-12)
        weather: Weather condition
        is_holiday: Whether it's a holiday
        is_major_event: Whether there's a major event
    
    Returns:
        Predicted traffic volume
    """
    model = model_package['model']
    encoders = model_package['encoders']
    
    # Encode categorical variables
    location_encoded = encoders['location'].transform([location_id])[0]
    weather_encoded = encoders['weather'].transform([weather])[0]
    
    # Create feature array (must match training order!)
    features = np.array([[
        hour,
        day_of_week,
        month,
        location_encoded,
        weather_encoded,
        int(is_holiday),
        int(is_major_event)
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return max(0, int(prediction))


def main():
    """Main Streamlit application."""
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    st.markdown('<p class="main-header">🚗 Pune Traffic Congestion Predictor</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-powered traffic prediction under sparse sensor data conditions</p>', 
                unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    # ==========================================================================
    # SIDEBAR - Input Controls
    # ==========================================================================
    with st.sidebar:
        st.markdown("## 📍 Select Parameters")
        st.markdown("---")
        
        # Date selection
        selected_date = st.date_input(
            "📅 Select Date",
            value=date.today(),
            help="Choose a date for prediction"
        )
        
        # Time selection
        selected_hour = st.slider(
            "🕐 Select Hour",
            min_value=0,
            max_value=23,
            value=9,
            format="%d:00",
            help="Choose hour of day (24-hour format)"
        )
        
        st.markdown("---")
        
        # Location selection
        location_options = {v['name']: k for k, v in LOCATIONS.items()}
        selected_location_name = st.selectbox(
            "📍 Select Location",
            options=list(location_options.keys()),
            help="Choose a location in Pune"
        )
        selected_location_id = location_options[selected_location_name]
        
        # Weather selection
        selected_weather = st.selectbox(
            "🌤️ Weather Condition",
            options=WEATHER_OPTIONS,
            index=0,
            help="Current weather affects traffic patterns"
        )
        
        st.markdown("---")
        
        # Holiday checkbox
        is_holiday = st.checkbox(
            "🎉 Is it a Holiday?",
            value=False,
            help="Holidays have different traffic patterns"
        )
        
        # MAJOR EVENT CHECKBOX - Key feature for crowd prediction!
        is_major_event = st.checkbox(
            "🏏 Major Event Today?",
            value=False,
            help="Major events (cricket match, festival, concert) cause traffic spikes!"
        )
        
        if is_major_event:
            st.info("🎉 Event mode enabled! Expecting higher traffic due to crowds.")
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🔮 Predict Traffic", type="primary", use_container_width=True)
    
    # ==========================================================================
    # MAIN CONTENT
    # ==========================================================================
    
    # Calculate derived values
    day_of_week = selected_date.weekday()
    month = selected_date.month
    
    # Make prediction (always, for real-time updates)
    predicted_volume = predict_traffic(
        model_package,
        selected_location_id,
        selected_hour,
        day_of_week,
        month,
        selected_weather,
        is_holiday,
        is_major_event
    )
    
    # Get traffic level
    level_name, level_color, level_emoji = get_traffic_level(predicted_volume)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        # Prediction Results Card
        st.markdown("### 📊 Prediction Results")
        
        # Main metric
        st.metric(
            label="Predicted Traffic Volume",
            value=f"{predicted_volume} vehicles/hr",
            delta=f"{level_emoji} {level_name}"
        )
        
        # Traffic level indicator
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {level_color}22 0%, {level_color}44 100%);
            border-left: 4px solid {level_color};
            padding: 1rem;
            border-radius: 0 10px 10px 0;
            margin: 1rem 0;
        ">
            <h3 style="margin: 0; color: {level_color};">{level_emoji} {level_name}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666;">
                {"Light traffic flow, good time to travel!" if predicted_volume < 300 else 
                 "Moderate congestion expected, plan for delays." if predicted_volume < 600 else
                 "Heavy congestion! Consider alternative routes or times."}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input summary
        st.markdown("### 📋 Input Summary")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | 📅 Date | {selected_date.strftime('%B %d, %Y')} |
        | 🕐 Time | {selected_hour:02d}:00 |
        | 📍 Location | {LOCATIONS[selected_location_id]['name']} |
        | 🌤️ Weather | {selected_weather} |
        | 🎉 Holiday | {'Yes' if is_holiday else 'No'} |
        | 🏏 Major Event | {'Yes' if is_major_event else 'No'} |
        """)
        
        # Event impact explanation
        if is_major_event:
            st.warning("""
            **🏏 Major Event Impact:**  
            Traffic is ~80-120% higher than normal due to crowds.
            This simulates real-world scenarios like IPL matches or festivals.
            """)
    
    with col2:
        st.markdown("### 🗺️ Location Map")
        
        # Create and display map
        traffic_map = create_traffic_map(selected_location_id, predicted_volume)
        st_folium(traffic_map, width=700, height=450)
        
        # Map legend
        st.markdown("""
        **Map Legend:**
        - 🟢 Green marker: Low traffic (<300 vehicles/hr)
        - 🟡 Orange marker: Moderate traffic (300-600 vehicles/hr)
        - 🔴 Red marker: High traffic (>600 vehicles/hr)
        - ⚪ Gray markers: Other monitoring locations
        """)
    
    # ==========================================================================
    # FOOTER - Project Info
    # ==========================================================================
    st.markdown("---")
    
    with st.expander("ℹ️ About This Project"):
        st.markdown("""
        ### Urban Traffic Congestion Prediction Under Sparse and Noisy Sensor Data
        
        **Project Highlights:**
        
        1. **Synthetic Data Generation**: Since real sensor data is unavailable, 
           we generated realistic traffic data for 4 Pune locations over 90 days.
        
        2. **Sparsity Handling**: We intentionally corrupted 40% of data to simulate 
           real-world sensor failures, then used KNN Imputation to recover it.
        
        3. **Event-Aware Prediction**: The "Major Event" checkbox demonstrates 
           crowd prediction capability - essential for smart city applications.
        
        4. **Random Forest Model**: Chosen for its ability to handle non-linear 
           patterns (rush hours) and robustness to outliers (event spikes).
        
        **Tech Stack:** Python, Pandas, Scikit-Learn, Streamlit, Folium
        
        ---
        *Internship Project - Intelligent Transportation Systems*
        """)


if __name__ == "__main__":
    main()
