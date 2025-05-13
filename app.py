from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import os
import io
import tempfile
import librosa
import noisereduce as nr
import soundfile as sf
from flask_cors import CORS


app = Flask(__name__)
model = None  # Global model variable
CORS(app)

# Load model from file
def load_audio_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Prediction Function
def predict_health(audio_data, sr, model):
    # Preprocess the audio data (adjust as needed for your model)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=400)  # Example: MFCC features
    mfccs_processed = np.mean(mfccs.T, axis=0)  # Example: Averaging MFCCs
    mfccs_processed = mfccs_processed.reshape(1, -1)  # Reshape for model input

    # Make a prediction
    prediction = model.predict(mfccs_processed)

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    if predicted_class == 1:
        status = "Healthy"
    else:
        status = "Unhealthy"

    return status, confidence

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_path = temp_file.name
        file.save(temp_path)

    try:
        new_audio, sr = librosa.load(temp_path, sr=None)
        result = predict_health(new_audio, sr, model)
        os.remove(temp_path)
        return jsonify(result), 200
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

# Healthy route
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "Healthy"}), 200

# Noise Reduction route
@app.route('/noise-reduction', methods=['POST'])
def noise_reduction():
    # 1) Validate presence of both files
    if 'noise_only' not in request.files or 'heart_noisy' not in request.files:
        return jsonify({"error": "Both 'noise_only' and 'heart_noisy' files are required"}), 400

    noise_file = request.files['noise_only']
    heart_file = request.files['heart_noisy']

    # 2) Validate filenames
    if noise_file.filename == '' or heart_file.filename == '':
        return jsonify({"error": "No file selected for one or both fields"}), 400

    try:
        # 3) Save uploads to temporary .wav files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_noise:
            noise_path = tmp_noise.name
            noise_file.save(noise_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_heart:
            heart_path = tmp_heart.name
            heart_file.save(heart_path)

        # 4) Load with librosa (requires ffmpeg/libsndfile on PATH)
        noisy, sr = librosa.load(heart_path, sr=None)
        noise, _ = librosa.load(noise_path, sr=sr)

        # 5) Perform noise reduction
        clean = nr.reduce_noise(y=noisy, y_noise=noise, sr=sr)

        # 6) Write cleaned audio to an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, clean, sr, format='WAV')
        buffer.seek(0)

        # 7) Clean up temp files
        os.remove(noise_path)
        os.remove(heart_path)

    except Exception as e:
        # Ensure temp files are removed on error
        for path in (locals().get('noise_path'), locals().get('heart_path')):
            if path and os.path.exists(path):
                os.remove(path)
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    # 8) Send the cleaned audio back as a WAV attachment
    return send_file(
        buffer,
        as_attachment=True,
        download_name="cleaned.wav",
        mimetype="audio/wav"
    )


# Main startup
if __name__ == "__main__":
    model_path = "/models/lung_sound_classification_model.keras"
    if os.path.exists(model_path):
        model = load_audio_model(model_path)
    else:
        print(f"Model file not found at {model_path}")
    app.run(host="0.0.0.0", port=5000, debug=True)
