import os
import cv2
from flask import Flask, render_template, request, redirect, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Load the Model
try:
    # Ensure your 'best.pt' is in the 'models' folder
    model = YOLO('models/best.pt')
    print(f"✅ EcoSort AI Loaded. Classes: {model.names}")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    model = None

# --- WEBCAM LOGIC ---
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success: break
        
        # Color fix & Predict
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=img_rgb, conf=0.25, imgsz=640, verbose=False)
        
        # Plot and Convert back for Browser
        annotated = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not model:
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '': return redirect('/')

    # Save original
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run Inference
    results = model.predict(source=img_rgb, conf=0.2, imgsz=640)
    
    # Save Plotted Result
    res_plotted = results[0].plot()
    result_filename = 'res_' + filename
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR))

    # Detect what was found for the UI label
    detections = [model.names[int(c)] for c in results[0].boxes.cls]
    found_text = ", ".join(detections) if detections else "No Plastic Detected"

    return render_template('index.html', 
                           original_image=filename, 
                           result_image=result_filename,
                           analysis=found_text)

if __name__ == '__main__':
    app.run(debug=True, port=5000)