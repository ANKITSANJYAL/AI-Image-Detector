from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import cv2
import joblib
from skimage.feature import canny
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained models
models = {
    "svm": joblib.load("models/svm_model.pkl"),
    "log_reg": joblib.load("models/log_reg_pipeline.pkl"),
    "random_forest": joblib.load("models/rf_model.pkl"),
}

# Load scaler (ensure it matches training preprocessing)
scaler = joblib.load("models/scaler.pkl")

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(image_path):
    """Extract features from an image for prediction"""
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
    return scaler.transform(features)

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

            model = models[model_choice]
            raw_prediction = model.predict(features)[0]
            confidence = max(model.predict_proba(features)[0]) * 100

            # Convert prediction to human-readable format
            prediction_label = "Real Image" if raw_prediction == 0 else "AI-Generated Image"

            # Move the image to static for displaying
            static_filepath = os.path.join(STATIC_FOLDER, file.filename)
            os.rename(filepath, static_filepath)

            return render_template("result.html", prediction=prediction_label, confidence=confidence, model=model_choice, image_file=file.filename)

    return render_template("index.html")

@app.route("/static/<filename>")
def serve_static_image(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
