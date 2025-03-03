from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import cv2
import joblib
from skimage.feature import canny
from sklearn.preprocessing import StandardScaler
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load trained models safely
models = {}
model_files = {
    "svm": "models/svm_model.pkl",
    "log_reg": "models/log_reg_pipeline.pkl",
    "random_forest": "models/rf_model.pkl",
}

for key, path in model_files.items():
    if os.path.exists(path):
        models[key] = joblib.load(path)
    else:
        print(f"Warning: Model {key} not found at {path}")

# Load scaler
scaler_path = "models/scaler.pkl"
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

def extract_features(image_path):
    """Extract features from an image for prediction."""
    img = cv2.imread(image_path)
    if img is None:
        return None  # Handle missing image case
    
    img = cv2.resize(img, (256, 256))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Feature Extraction
    mean_intensity = np.mean(img_gray)
    median_intensity = np.median(img_gray)
    std_intensity = np.std(img_gray)

    # Edge Density
    edges = canny(img_gray, sigma=1)
    edge_density = np.sum(edges) / (img_gray.shape[0] * img_gray.shape[1])

    # Fourier Transform
    f_transform = np.fft.fft2(img_gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)

    mean_freq = np.mean(magnitude_spectrum)
    std_freq = np.std(magnitude_spectrum)
    high_freq = np.mean(magnitude_spectrum[128:, 128:])

    features = np.array([mean_intensity, median_intensity, std_intensity, edge_density, mean_freq, std_freq, high_freq]).reshape(1, -1)

    return scaler.transform(features) if scaler else None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        model_choice = request.form.get("model")

        if file and model_choice:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            features = extract_features(filepath)
            if features is None:
                return "Error: Could not extract features from image", 400

            if model_choice not in models:
                return "Error: Model not found", 400

            model = models[model_choice]
            raw_prediction = model.predict(features)[0]
            confidence = max(model.predict_proba(features)[0]) * 100

            # Convert prediction to human-readable format
            prediction_label = "Real Image" if raw_prediction == 0 else "AI-Generated Image"

            # Move the image to static folder
            static_filepath = os.path.join(STATIC_FOLDER, file.filename)
            shutil.move(filepath, static_filepath)

            return render_template("result.html", prediction=prediction_label, confidence=confidence, model=model_choice, image_file=file.filename)

    return render_template("index.html")

@app.route("/static/<filename>")
def serve_static_image(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
