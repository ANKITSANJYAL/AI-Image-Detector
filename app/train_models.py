import os
import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import canny
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define paths
train_csv_path = "/Users/ankitsanjyal/Desktop/Projects/Portfolio/AI-REAL-IMAGES-DATA/train.csv"  
train_images_path = "/Users/ankitsanjyal/Desktop/Projects/Portfolio/AI-REAL-IMAGES-DATA/"  

# Load dataset
print("Loading dataset...")
df_train = pd.read_csv(train_csv_path)
df_train["image_path"] = df_train["file_name"].apply(lambda x: os.path.join(train_images_path, x))
print(f"Dataset loaded with {len(df_train)} entries.")

# Function to extract statistical and frequency-based features
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Image {image_path} not found.")
        return None  # Skip if image is not found
    
    img = cv2.resize(img, (256, 256))  # Standardize dimensions
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pixel Intensity Stats
    mean_intensity = np.mean(img_gray)
    median_intensity = np.median(img_gray)
    std_intensity = np.std(img_gray)

    # Edge Density (Canny Edge Detection)
    edges = canny(img_gray, sigma=1)
    edge_density = np.sum(edges) / (img_gray.shape[0] * img_gray.shape[1])

    # Fourier Transform Features
    f_transform = np.fft.fft2(img_gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)

    mean_freq = np.mean(magnitude_spectrum)
    std_freq = np.std(magnitude_spectrum)
    high_freq = np.mean(magnitude_spectrum[128:, 128:])  # Focus on high-freq region

    return [mean_intensity, median_intensity, std_intensity, edge_density, mean_freq, std_freq, high_freq]

# Extract features
print("Extracting features...")
features = []
labels = []
for _, row in df_train.iterrows():
    extracted = extract_features(row['image_path'])
    if extracted is not None:
        features.append(extracted)
        labels.append(row['label'])
print(f"Feature extraction completed for {len(features)} images.")

# Convert to NumPy array
X = np.array(features)
y = np.array(labels)

# Split dataset
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models with best hyperparameters
print("Training SVM model...")
svm_model = SVC(probability=True, kernel='rbf', C=10, gamma=1, random_state=42)
svm_model.fit(X_train, y_train)
print("SVM model trained.")

print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=150,  # Reduce number of trees slightly
    max_depth=12,  # Reduce depth to prevent overfitting
    min_samples_split=10,  # Require at least 10 samples to split a node
    min_samples_leaf=5,  # Each leaf node should have at least 5 samples
    max_features='sqrt',  # Consider only a subset of features to add randomness
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Random Forest model trained.")

print("Training Logistic Regression model...")
log_reg_pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, interaction_only=False)),  # Add non-linearity
    ("log_reg", LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, random_state=42))  # Regularized Logistic Regression
])
log_reg_pipeline.fit(X_train, y_train)
print("Logistic Regression model trained.")

# Save models
print("Saving models...")
os.makedirs("models", exist_ok=True)
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(log_reg_pipeline, "models/log_reg_pipeline.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Models saved in 'models/' folder.")

# Evaluate models
print("Evaluating models...")
for model, name in zip([svm_model, rf_model, log_reg_pipeline], ["SVM", "Random Forest", "Logistic Regression"]):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

print("Training completed successfully.")
