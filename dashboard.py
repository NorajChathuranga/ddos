import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import tensorflow as tf

# Architecture Reconstruction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
import plotly.express as px

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DDoS Detection Shield",
    page_icon="🛡️",
    layout="wide"
)

MODEL_FOLDER = 'saved_models'
DATA_FOLDER = 'processed_data'

# ==========================================
# 2. ARCHITECTURE RECONSTRUCTION
# ==========================================
# We manually rebuild the 3 Deep Learning models exactly as they were trained.
# This guarantees they work regardless of the "time_major" version bug.

def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_rnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(64, activation='tanh'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

# ==========================================
# 3. CACHED RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_resources():
    """Loads preprocessors and rebuilds models with saved weights."""
    
    # A. Load Preprocessors
    scaler = joblib.load(os.path.join(DATA_FOLDER, 'scaler.joblib'))
    
    with open(os.path.join(DATA_FOLDER, 'selected_features.json'), 'r') as f:
        features = json.load(f)
        
    # B. Load ML Models (Scikit-Learn is safe)
    rf = joblib.load(os.path.join(MODEL_FOLDER, 'rf_model.joblib'))
    svm = joblib.load(os.path.join(MODEL_FOLDER, 'svm_model.joblib'))
    meta = joblib.load(os.path.join(MODEL_FOLDER, 'meta_model.joblib'))
    
    # C. Load DL Models (Reconstruct + Load Weights)
    input_shape = (len(features), 1)
    
    cnn = build_cnn(input_shape)
    cnn.load_weights(os.path.join(MODEL_FOLDER, 'cnn_model.h5'))
    
    lstm = build_lstm(input_shape)
    lstm.load_weights(os.path.join(MODEL_FOLDER, 'lstm_model.h5'))
    
    rnn = build_rnn(input_shape)
    rnn.load_weights(os.path.join(MODEL_FOLDER, 'rnn_model.h5'))
    
    return scaler, features, rf, svm, meta, cnn, lstm, rnn

# Load everything
try:
    scaler, selected_features, rf, svm, meta, cnn, lstm, rnn = load_resources()
    st.success("System Ready: All 5 Models Loaded & Hybrid Engine Active.")
except Exception as e:
    st.error(f"Critical Error Loading Resources: {e}")
    st.stop()

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.title("🛡️ Hybrid DDoS Detection System")
st.markdown("### Multi-Stage Stacked Ensemble Classifier")

st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Network Traffic (CSV)", type=["csv"])

# ==========================================
# 5. PREDICTION ENGINE
# ==========================================
if uploaded_file is not None:
    st.info("Processing file... please wait.")
    
    # A. Read Data
    input_df = pd.read_csv(uploaded_file)
    original_df = input_df.copy()
    
    # Force clean columns (removes spaces like " Active Std" -> "Active Std")
    input_df.columns = input_df.columns.str.strip()
    
    # B. Preprocessing
    # 1. Select Columns (Handle missing/extra)
    missing_cols = [c for c in selected_features if c not in input_df.columns]
    if missing_cols:
        st.warning(f"Warning: The following features were missing and filled with 0: {missing_cols}")
        for c in missing_cols:
            input_df[c] = 0
            
    input_df = input_df[selected_features] 
    
    # ---------------------------------------------------------
    # CRITICAL FIX: SANITIZE DATA (Handle Infinity & NaN)
    # ---------------------------------------------------------
    # Replace Infinity with NaN, then fill NaN with 0
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df.fillna(0, inplace=True)
    # ---------------------------------------------------------
    
    # 3. Scale
    try:
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=selected_features)
    except Exception as e:
        st.error(f"Scaling Error: {e}")
        st.stop()
    
    # 4. Reshape for DL
    X_ml = input_scaled.values
    X_dl = X_ml.reshape((X_ml.shape[0], X_ml.shape[1], 1))

    # C. Level 1 Predictions (Base Models)
    progress_bar = st.progress(0)
    st.write("Gathering votes from Base Models...")
    
    # Random Forest
    p1 = rf.predict_proba(X_ml)[:, 1]
    progress_bar.progress(20)
    
    # SVM
    p2 = svm.predict_proba(X_ml)[:, 1]
    progress_bar.progress(40)
    
    # CNN
    p3 = cnn.predict(X_dl, verbose=0).flatten()
    progress_bar.progress(60)
    
    # LSTM
    p4 = lstm.predict(X_dl, verbose=0).flatten()
    progress_bar.progress(80)
    
    # RNN
    p5 = rnn.predict(X_dl, verbose=0).flatten()
    progress_bar.progress(100)
    
    # Stack Votes
    stacked_input = np.column_stack((p1, p2, p3, p4, p5))
    
    # D. Level 2 Prediction (Meta-Model)
    final_probs = meta.predict_proba(stacked_input)[:, 1]
    final_preds = (final_probs > 0.5).astype(int)
    
    # ==========================================
    # 6. RESULTS DASHBOARD
    # ==========================================
    st.divider()
    
    total_packets = len(final_preds)
    attack_count = np.sum(final_preds)
    benign_count = total_packets - attack_count
    attack_percentage = (attack_count / total_packets) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Packets", total_packets)
    col2.metric("Malicious Packets", int(attack_count), delta_color="inverse")
    col3.metric("Safe Packets", int(benign_count))
    
    if attack_percentage > 10:
        st.error(f"⚠️ CRITICAL ALERT: High Volume Attack Detected! ({attack_percentage:.2f}% Malicious)")
    elif attack_percentage > 0:
        st.warning(f"⚠️ Warning: Suspicious Traffic Detected ({attack_percentage:.2f}% Malicious)")
    else:
        st.success("✅ System Normal: No Threats Detected.")
        
    st.subheader("Traffic Composition")
    chart_data = pd.DataFrame({
        "Type": ["Benign", "Malicious"],
        "Count": [benign_count, attack_count]
    })
    fig = px.pie(chart_data, values='Count', names='Type', 
                 color='Type', color_discrete_map={'Benign':'green', 'Malicious':'red'})
    st.plotly_chart(fig)
    
    st.subheader("Packet Analysis Details")
    display_df = original_df.copy()
    display_df['RF_Vote'] = p1
    display_df['CNN_Vote'] = p3
    display_df['Final_Probability'] = final_probs
    display_df['Prediction'] = ['ATTACK' if x==1 else 'BENIGN' for x in final_preds]
    
    display_df = display_df.sort_values(by='Final_Probability', ascending=False)
    st.dataframe(display_df.head(100))
    
    st.download_button(
        "Download Full Analysis Report",
        display_df.to_csv(index=False).encode('utf-8'),
        "ddos_analysis_report.csv",
        "text/csv",
        key='download-csv'
    )

else:
    st.info("👈 Upload a CSV file from the sidebar to begin analysis.")