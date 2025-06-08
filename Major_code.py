import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import plotly.figure_factory as ff
import plotly.graph_objs as go

# Set page configuration
st.set_page_config(page_title="Autoencoder + XGBoost Classifier", layout="wide")

# Title and description
st.title("🔍 Autoencoder + XGBoost Classifier")
st.markdown("""
Upload a CSV dataset, configure model parameters, and train an Autoencoder combined with XGBoost classifier.
The app displays model performance metrics and a confusion matrix heatmap.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None

def train_model(df, epochs, learning_rate, n_estimators, max_depth):
    try:
        # Encode target
        target_le = LabelEncoder()
        df["Target"] = target_le.fit_transform(df["Target"])
        y = df["Target"]
        X = df.drop("Target", axis=1)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        # Build autoencoder
        input_dim = X_train.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(24, activation="relu")(input_layer)
        encoded = Dense(12, activation="relu")(encoded)
        bottleneck = Dense(6, activation="relu")(encoded)
        decoded = Dense(12, activation="relu")(bottleneck)
        decoded = Dense(24, activation="relu")(decoded)
        output_layer = Dense(input_dim, activation="linear")(decoded)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        encoder = Model(inputs=input_layer, outputs=bottleneck)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)

        # Transform features using encoder
        X_train_encoded = encoder.predict(X_train, verbose=0)
        X_test_encoded = encoder.predict(X_test, verbose=0)

        # Train XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, 
            use_label_encoder=False, eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_encoded, y_train)

        # Evaluate model
        y_pred = xgb_model.predict(X_test_encoded)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_le.classes_, output_dict=True)

        return accuracy, conf_matrix, report, target_le.classes_

    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None, None, None, None

if uploaded_file is not None:
    # Read and display dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Model parameters
    st.subheader("⚙️ Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Number of Epochs", min_value=10, max_value=100, value=50, step=10)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
    with col2:
        n_estimators = st.slider("XGBoost Estimators", min_value=50, max_value=300, value=200, step=50)
        max_depth = st.slider("XGBoost Max Depth", min_value=3, max_value=10, value=6, step=1)

    # Train button
    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            accuracy, conf_matrix, report, classes = train_model(df, epochs, learning_rate, n_estimators, max_depth)
            
            if accuracy is not None:
                st.session_state.results = {
                    'accuracy': accuracy,
                    'conf_matrix': conf_matrix,
                    'report': report,
                    'classes': classes
                }

    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        st.subheader("📈 Model Results")
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col2:
            st.write("Classification Report")
            report_df = pd.DataFrame(results['report']).transpose()
            st.dataframe(report_df)

        # Confusion Matrix Heatmap
        st.subheader("📊 Confusion Matrix Heatmap")
        fig = ff.create_annotated_heatmap(
            z=results['conf_matrix'],
            x=results['classes'].tolist(),
            y=results['classes'].tolist(),
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=500,
            height=500
        )
        st.plotly_chart(fig)

else:
    st.info("Please upload a CSV file to begin.")
