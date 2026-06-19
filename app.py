import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

# Set up browser page properties
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI styling configurations
st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #6a0dad; text-align: center; margin-bottom: 2rem; }
.prediction-card { background-color: #f0f8ff; padding: 2rem; border-radius: 10px; border-left: 5px solid #6a0dad; margin: 1rem 0; }
.confidence-bar { height: 20px; background-color: #e0e0e0; border-radius: 10px; margin: 0.5rem 0; overflow: hidden; }
.confidence-fill { height: 100%; background: linear-gradient(90deg, #ff6b6b, #4ecdc4); text-align: center; color: white; font-weight: bold; font-size: 0.85rem; line-height: 20px; }
</style>
""", unsafe_allow_html=True)

# Performance caching helper routines
@st.cache_resource
def load_model(format_type='joblib'):
    try:
        if format_type == 'joblib':
            return joblib.load('models/iris_model.joblib')
        elif format_type == 'pickle':
            with open('models/iris_model.pickle', 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model binary file: {e}")
        return None

@st.cache_resource
def load_metadata():
    try:
        with open('models/model_info.json', 'r') as f:
            info = json.load(f)
        with open('models/feature_ranges.json', 'r') as f:
            ranges = json.load(f)
        return info, ranges
    except Exception as e:
        st.error(f"Error extracting metadata definitions: {e}")
        return None, None

# Import data layers
model_info, feature_ranges = load_metadata()

# Default ranges lookup guard loop
if not feature_ranges:
    feature_ranges = {
        'sepal_length': {'min': 4.0, 'max': 8.0, 'default': 5.8},
        'sepal_width': {'min': 2.0, 'max': 4.5, 'default': 3.0},
        'petal_length': {'min': 1.0, 'max': 7.0, 'default': 4.0},
        'petal_width': {'min': 0.1, 'max': 2.5, 'default': 1.2}
    }

# Control Panel Sidebar Formulations
with st.sidebar:
    st.title("⚙️ Engine Controls")
    model_format = st.radio("Model Format Structure", ["joblib", "pickle"])
    
    st.divider()
    st.subheader("📊 Engine Specifications")
    if model_info:
        st.write(f"**Architecture:** {model_info.get('model_type')}")
        st.write(f"**Baseline Accuracy:** {model_info.get('accuracy', 0.90):.1%}")
    st.divider()

# Instantiate runtime instance memory model structure 
model = load_model(model_format)

# Core dashboard viewing page elements
st.markdown('<h1 class="main-header">🌸 Iris Flower Classification App</h1>', unsafe_allow_html=True)
st.markdown("Adjust the feature dimensions below to estimate iris classifications through live inferencing.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Metric Selection Elements")
    sepal_length = st.slider("Sepal Length (cm)", float(feature_ranges['sepal_length']['min']), float(feature_ranges['sepal_length']['max']), float(feature_ranges['sepal_length']['default']), step=0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  float(feature_ranges['sepal_width']['min']),  float(feature_ranges['sepal_width']['max']),  float(feature_ranges['sepal_width']['default']),  step=0.1)
    petal_length = st.slider("Petal Length (cm)", float(feature_ranges['petal_length']['min']), float(feature_ranges['petal_length']['max']), float(feature_ranges['petal_length']['default']), step=0.1)
    petal_width  = st.slider("Petal Width (cm)",  float(feature_ranges['petal_width']['min']),  float(feature_ranges['petal_width']['max']),  float(feature_ranges['petal_width']['default']),  step=0.1)

with col2:
    st.subheader("📊 Array Vectors Map")
    features_df = pd.DataFrame({
        'Measurement Parameter': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
    })
    st.dataframe(features_df, hide_index=True, use_container_width=True)

st.markdown("---")

# Compute Active Class Evaluation Pipelines
if st.button("🎯 Execute Prediction Task", type="primary", use_container_width=True):
    if model is not None and model_info is not None:
        # Vector shape mapping structure adaptation
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        try:
            prediction = model.predict(input_features)[0]
            probabilities = model.predict_proba(input_features)[0]
            predicted_class_label = model_info['target_names'][prediction]
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"### Target Class Result: **{predicted_class_label}**")
            st.markdown("#### Probability Matrix Confidence Weights:")
            
            for index, target_name in enumerate(model_info['target_names']):
                pct_val = probabilities[index] * 100
                st.write(f"**{target_name.capitalize()}**")
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {pct_val}%;">{pct_val:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as err:
            st.error(f"Failed evaluation pipeline routines execution: {err}")
    else:
        st.error("Target inference components are non-responsive.")
# At the very end of your app.py file
if __name__ == "__main__":
    import uvicorn
    # Make sure port=8000 is different from Streamlit's 8501
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)