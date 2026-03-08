from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import time
import os
from gtts import gTTS
import uuid
from datetime import datetime, timedelta
import glob
import gdown

app = Flask(__name__)

# Global variables
predicted_sentence = ""
current_sign = ""
prediction_count = 0
threshold_frames = 15
last_prediction = ""
camera_active = False
model_loaded = False  # Track if model successfully loaded

# Paths
model_path = "model/asl_model.joblib"
encoder_path = "model/label_encoder.joblib"

# Google Drive file IDs (only IDs, no full URLs)
model_file_id = "1oZeBgnRUqLYe5IaYG6NIokCEuqz07Ru2"
encoder_file_id = "13oBSsI927KltAI7z0bpz3hgCTrUAQap-"

# Ensure model directory exists before download
if not os.path.exists("model"):
    os.makedirs("model")

# Download model and encoder if missing
if not os.path.exists(model_path):
    print("Downloading ASL model...")
    gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_path, quiet=False)

if not os.path.exists(encoder_path):
    print("Downloading LabelEncoder...")
    gdown.download(f"https://drive.google.com/uc?id={encoder_file_id}", encoder_path, quiet=False)

# Load model and encoder safely
try:
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    model_loaded = True
    print("Model and encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    model_loaded = False

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.95,
    min_tracking_confidence=0.95
)

# Create static directory for audio files if it doesn't exist
AUDIO_DIR = os.path.join('static', 'audio')
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global predicted_sentence, current_sign, prediction_count, last_prediction, model_loaded

        ret, frame = self.video.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and model_loaded:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                x_input = np.array(features).reshape(1, -1)

                y_pred = model.predict(x_input)
                label = le.inverse_transform(y_pred)[0]

                current_sign = label

                if label == last_prediction:
                    prediction_count += 1
                else:
                    prediction_count = 0
                    last_prediction = label

                if prediction_count == threshold_frames:
                    if label == "space":
                        predicted_sentence += " "
                    elif label == "del":
                        predicted_sentence = predicted_sentence[:-1]
                    elif label != "nothing":
                        predicted_sentence += label
                    prediction_count = 0
        else:
            current_sign = "nothing"

        cv2.rectangle(frame, (10, 10), (630, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {current_sign}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {predicted_sentence}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        return frame

def gen_frames():
    camera = VideoCamera()
    while camera_active:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1/45)

def cleanup_old_audio_files():
    try:
        cutoff_time = datetime.now() - timedelta(hours=1)
        audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))

        for file_path in audio_files:
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_time < cutoff_time:
                os.remove(file_path)
                print(f"Removed old audio file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up audio files: {e}")

def generate_audio_file(text):
    try:
        if not text or text.strip() == "":
            return None

        cleanup_old_audio_files()

        filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)

        return filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({'status': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'status': 'Camera stopped'})

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    return jsonify({
        'sentence': predicted_sentence,
        'current_sign': current_sign,
        'prediction_count': prediction_count,
        'threshold_frames': threshold_frames
    })

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global predicted_sentence
    predicted_sentence = ""
    return jsonify({'status': 'Sentence cleared'})

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    global predicted_sentence

    if not predicted_sentence or predicted_sentence.strip() == "":
        return jsonify({
            'status': 'error',
            'message': 'No sentence to speak'
        }), 400

    audio_filename = generate_audio_file(predicted_sentence)

    if audio_filename:
        return jsonify({
            'status': 'success',
            'sentence': predicted_sentence,
            'audio_url': f'/static/audio/{audio_filename}',
            'message': 'Audio generated successfully'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate audio'
        }), 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
