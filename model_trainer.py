import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import cv2
from pathlib import Path

def train_model():
    print("Training voice model...")
    # URL to the UCI Parkinson's dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    
    try:
        df = pd.read_csv(url)
        print("Voice Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading voice dataset: {e}")
        # Create synthetic data if URL fails
        print("Creating synthetic voice dataset...")
        data_size = 200
        np.random.seed(42)
        
        # Synthetic data based on common ranges
        data = {
            'MDVP:Fo(Hz)': np.random.uniform(100, 250, data_size),
            'MDVP:Fhi(Hz)': np.random.uniform(120, 300, data_size),
            'MDVP:Flo(Hz)': np.random.uniform(80, 200, data_size),
            'MDVP:Jitter(%)': np.random.uniform(0.001, 0.02, data_size),
            'MDVP:Shimmer': np.random.uniform(0.01, 0.1, data_size),
            'NHR': np.random.uniform(0.001, 0.05, data_size),
            'HNR': np.random.uniform(15, 35, data_size),
            'RPDE': np.random.uniform(0.3, 0.7, data_size),
            'DFA': np.random.uniform(0.5, 0.8, data_size),
            'status': np.random.randint(0, 2, data_size)
        }
        df = pd.DataFrame(data)

    # Features we want (9 features + status)
    features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 
        'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 
        'HNR', 'RPDE', 'DFA'
    ]
    
    X_voice = df[features]
    y_voice = df['status']

    # Split and scale
    X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_voice, y_voice, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled_v = scaler.fit_transform(X_train_v)
    
    # Train model
    voice_model = RandomForestClassifier(n_estimators=100, random_state=42)
    voice_model.fit(X_train_scaled_v, y_train_v)
    print("Voice model trained.")

    print("Training vision model...")
    # Extract features from images
    dataset_path = Path("dataset/drawings")
    if not dataset_path.exists():
        dataset_path = Path("dataset") # Fallback
    
    X_vision = []
    y_vision = []
    
    # Helper to extract features
    def extract_vision_features(img_path):
        img = cv2.imread(str(img_path))
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary.flatten()

    # Load healthy
    healthy_count = 0
    for img_path in dataset_path.rglob("training/healthy/*.png"):
        feat = extract_vision_features(img_path)
        if feat is not None:
            X_vision.append(feat)
            y_vision.append(0)
            healthy_count += 1
            
    # Load parkinson
    parkinson_count = 0
    for img_path in dataset_path.rglob("training/parkinson/*.png"):
        feat = extract_vision_features(img_path)
        if feat is not None:
            X_vision.append(feat)
            y_vision.append(1)
            parkinson_count += 1
            
    print(f"Loaded {healthy_count} healthy images and {parkinson_count} parkinson images.")
    
    if len(X_vision) > 0:
        X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_vision, y_vision, test_size=0.2, random_state=42)
        vision_model = RandomForestClassifier(n_estimators=100, random_state=42)
        vision_model.fit(X_train_img, y_train_img)
        print("Vision model trained.")
    else:
        vision_model = None
        print("No images found, vision model not trained.")
        
    # Save both models and scaler
    model_data = {
        'model': voice_model,
        'scaler': scaler,
        'features': features,
        'vision_model': vision_model
    }
    
    with open('parkinsons_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Models trained and saved to parkinsons_model.pkl")

if __name__ == "__main__":
    train_model()
