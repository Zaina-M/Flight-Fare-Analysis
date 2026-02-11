"""
Flight Price Prediction - Streamlit Web Application.

This application provides an interactive interface for predicting
flight prices using trained ML models.

Usage:
    streamlit run app.py

Stretch Challenge Implementation:
- Live predictions via web interface
- Interactive feature input
- Real-time model inference
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from datetime import datetime, date

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MODELS_DIR, PLOTS_DIR, data_config, RAW_DATA_DIR

# Dataset path
DATASET_PATH = RAW_DATA_DIR / data_config.dataset_name

# Page configuration
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1565C0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(ttl=60)  # Cache for 60 seconds, then recheck
def load_model():
    """Load the trained model and feature engineer."""
    model_path = MODELS_DIR / "feature_engineer.pkl"
    
    # Find the latest best model
    model_files = list(MODELS_DIR.glob("best_model_*.joblib"))
    
    if not model_files:
        # Try to find any saved model
        model_files = list(MODELS_DIR.glob("*.joblib"))
    
    if not model_files:
        return None, None
    
    # Get the most recent model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    try:
        model_data = joblib.load(latest_model)
        model = model_data.get('model') if isinstance(model_data, dict) else model_data
        
        # Load feature engineer if available
        feature_engineer = None
        if model_path.exists():
            import pickle
            with open(model_path, 'rb') as f:
                feature_engineer = pickle.load(f)
        
        return model, feature_engineer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def load_dataset_options():
    """Load airlines and cities dynamically from the dataset."""
    try:
        df = pd.read_csv(DATASET_PATH)
        airlines = sorted(df['Airline'].dropna().unique().tolist())
        cities = sorted(set(df['Source'].dropna().unique().tolist() + 
                           df['Destination'].dropna().unique().tolist()))
        classes = sorted(df['Class'].dropna().unique().tolist())
        booking_sources = sorted(df['Booking Source'].dropna().unique().tolist())
        seasonalities = sorted(df['Seasonality'].dropna().unique().tolist())
        return {
            'airlines': airlines,
            'cities': cities,
            'classes': classes,
            'booking_sources': booking_sources,
            'seasonalities': seasonalities
        }
    except Exception as e:
        st.warning(f"Could not load dataset: {e}. Using defaults.")
        return {
            'airlines': ['Biman Bangladesh', 'US-Bangla', 'Novoair'],
            'cities': ['Dhaka', 'Chittagong', 'Sylhet'],
            'classes': ['Economy', 'Business', 'First Class'],
            'booking_sources': ['Online Website', 'Direct Booking', 'Travel Agency'],
            'seasonalities': ['Regular', 'Eid', 'Hajj', 'Winter Holidays']
        }


def get_sample_data():
    """Get sample data for reference."""
    options = load_dataset_options()
    return {
        'airlines': options['airlines'],
        'cities': options['cities'],
        'classes': options['classes'],
        'booking_sources': options['booking_sources'],
        'seasonalities': options['seasonalities'],
        'seasons': ['Winter', 'Summer', 'Monsoon', 'Autumn']
    }


def create_input_features(airline, source, destination, travel_date,
                          flight_class, booking_source, seasonality,
                          duration, days_before):
    """Create feature DataFrame from user inputs (without fare components)."""
    # Create DataFrame matching the columns the model was trained on
    # (excluding Base Fare, Tax, and columns_to_drop — those cause data leakage)
    features = pd.DataFrame({
        'Airline': [airline],
        'Source': [source],
        'Destination': [destination],
        'Departure Date & Time': [pd.Timestamp(travel_date)],
        'Duration (hrs)': [duration],
        'Class': [flight_class],
        'Booking Source': [booking_source],
        'Seasonality': [seasonality],
        'Days Before Departure': [days_before],
    })
    
    return features


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header"> Flight Price Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header(" About")
        st.write("""
        This application predicts flight ticket prices based on:
        - Route (Source → Destination)
        - Airline
        - Travel Date & Season
        
        The model was trained on Bangladesh flight data.
        """)
        
        st.header(" Model Info")
        
        # Add refresh button to reload model
        if st.button("🔄 Refresh Model"):
            st.cache_resource.clear()
            st.rerun()
        
        model, feature_engineer = load_model()
        
        if model is not None:
            st.success(" Model loaded successfully!")
            model_type = type(model).__name__
            st.write(f"**Model Type:** {model_type}")
        else:
            st.warning(" No trained model found. Please train a model first using `python main.py`")
    
    # Main content
    sample_data = get_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Flight Details")
        
        # Airline selection
        airline = st.selectbox(
            "Select Airline",
            options=sample_data['airlines'],
            help="Choose the airline for your flight"
        )
        
        # Route selection
        source = st.selectbox(
            "Departure City",
            options=sample_data['cities'],
            index=0,
            help="Select your departure city"
        )
        
        destination = st.selectbox(
            "Arrival City",
            options=sample_data['cities'],
            index=1,
            help="Select your destination city"
        )
        
        if source == destination:
            st.warning(" Source and destination cannot be the same!")
        
        # Class selection
        flight_class = st.selectbox(
            "Ticket Class",
            options=sample_data['classes'],
            index=sample_data['classes'].index('Economy') if 'Economy' in sample_data['classes'] else 0,
            help="Economy, Business, or First Class"
        )
        
        # Booking source
        booking_source = st.selectbox(
            "Booking Source",
            options=sample_data['booking_sources'],
            help="How the ticket is booked"
        )
    
    with col2:
        st.subheader(" Travel Information")
        
        # Date selection
        travel_date = st.date_input(
            "Travel Date",
            value=date.today(),
            min_value=date.today(),
            help="Select your travel date"
        )
        
        # Show travel info
        st.info(f"**Day:** {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][travel_date.weekday()]}")
        if travel_date.weekday() >= 5:
            st.warning("Weekend travel may have higher prices")
        
        # Seasonality
        seasonality = st.selectbox(
            "Season / Occasion",
            options=sample_data['seasonalities'],
            help="Regular, Eid, Hajj, or Winter Holidays"
        )
        
        # Duration
        duration = st.slider(
            "Estimated Flight Duration (hours)",
            min_value=0.5, max_value=16.0, value=2.0, step=0.5,
            help="Approximate flight duration in hours"
        )
        
        # Days before departure
        days_before = st.slider(
            "Days Before Departure",
            min_value=1, max_value=90, value=14,
            help="How many days in advance you are booking"
        )
    
    st.markdown("---")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_btn = st.button(" Predict Price", use_container_width=True, type="primary")
    
    if predict_btn:
        if source == destination:
            st.error("Please select different source and destination cities!")
        elif model is None:
            st.error("No trained model available. Please train a model first.")
        else:
            with st.spinner("Calculating prediction..."):
                try:
                    # Create input features (without fare components)
                    input_df = create_input_features(
                        airline, source, destination, travel_date,
                        flight_class, booking_source, seasonality,
                        duration, days_before
                    )
                    
                    # If we have feature engineer, use it
                    if feature_engineer is not None:
                        try:
                            input_transformed = feature_engineer.transform(input_df, scale_features=True)
                        except Exception as e:
                            st.error(f"Feature transformation error: {e}")
                            input_transformed = None
                    else:
                        input_transformed = None
                    
                    # Make prediction
                    if input_transformed is not None:
                        # Safety net: fill any remaining NaN with 0
                        input_transformed = input_transformed.fillna(0)
                        prediction = model.predict(input_transformed)[0]
                        
                        # Reverse log-transform if model was trained on log target
                        if feature_engineer is not None and getattr(feature_engineer, 'log_transform_target', False):
                            prediction = np.expm1(prediction)
                        
                        # Display prediction
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown("### Predicted Total Fare")
                        st.markdown(f'<p class="prediction-value">৳ {prediction:,.2f}</p>', 
                                   unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show flight details
                        st.subheader(" Flight Details")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Route", f"{source} → {destination}")
                        with col2:
                            st.metric("Airline", airline)
                        with col3:
                            st.metric("Date", travel_date.strftime("%d %b %Y"))
                    
                        # Show input summary
                        with st.expander(" View Input Summary"):
                            st.dataframe(input_df)
                            
                            st.write("**Route:**", f"{source} → {destination}")
                            st.write("**Class:**", flight_class)
                            st.write("**Booking Source:**", booking_source)
                            st.write("**Seasonality:**", seasonality)
                            st.write("**Duration:**", f"{duration} hours")
                            st.write("**Days Before Departure:**", days_before)
                            st.write("**Day of Week:**", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][travel_date.weekday()])
                            st.write("**Weekend Travel:**", "Yes" if travel_date.weekday() >= 5 else "No")
                    else:
                        st.error("Could not process features. Please retrain the model with: `python main.py`")
                
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info("Try running the full pipeline first: `python main.py`")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Flight Price Prediction | Built with Streamlit | 
        <a href='https://github.com'>View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
