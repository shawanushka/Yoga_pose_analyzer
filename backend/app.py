from flask import Flask, render_template, request, jsonify
from pose_utils import analyze_pose
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    data = request.json
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    filename = secure_filename('webcam_frame.jpg')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(img_bytes)
    result = analyze_pose(filepath)
    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = analyze_pose(filepath)
    # Use annotated image if available
    image_path = result.get('annotated_image', filepath)
    return render_template('index.html', result=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
