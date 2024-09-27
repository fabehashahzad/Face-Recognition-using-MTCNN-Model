from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import face_recognition
import pickle
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

app = Flask(__name__)

# Create a folder to save images if it doesn't exist
IMAGES_FOLDER = 'static/Images'
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

# Function to draw bounding boxes on detected faces
def draw_facebox(image, result_list, recognized_names=None):
    plt.imshow(image)
    ax = plt.gca()

    for idx, result in enumerate(result_list):
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='green')
        ax.add_patch(rect)

        # Add recognized names near the bounding box
        if recognized_names and idx < len(recognized_names):
            plt.text(x, y-10, recognized_names[idx], color='green', fontsize=12)

    plt.axis('off')
    # Save the image with bounding boxes as output
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return Image.open(buffer)

# Function to save face encodings
def save_face_encoding(name, encoding, known_faces_file='known_faces.pkl'):
    try:
        with open(known_faces_file, 'rb') as f:
            known_faces = pickle.load(f)
    except FileNotFoundError:
        known_faces = {}

    known_faces[name] = encoding
    with open(known_faces_file, 'wb') as f:
        pickle.dump(known_faces, f)

# Add image to dataset route
@app.route('/add_to_dataset', methods=['POST'])
def add_to_dataset():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    name = request.form.get('name')
    if not name:
        return jsonify({'error': 'Name not provided'}), 400

    image = request.files['image']
    img = Image.open(image)
    pixels = np.array(img)

    detector = MTCNN()
    faces = detector.detect_faces(pixels)

    if faces:
        # Save uploaded image
        image_path = os.path.join(IMAGES_FOLDER, f'{name}.jpg')
        img.save(image_path)

        # Draw bounding boxes on the image
        output_image = draw_facebox(pixels, faces, [name]*len(faces))
        output_image_path = os.path.join(IMAGES_FOLDER, f'processed_{name}.png')
        output_image.save(output_image_path)

        # Save face encoding
        face_encodings = face_recognition.face_encodings(pixels)
        if face_encodings:
            save_face_encoding(name, face_encodings[0])
            return jsonify({'message': f'Image saved and encoding for {name} stored.', 'output_image': f'/static/Images/processed_{name}.png'})
        else:
            return jsonify({'error': 'No face encodings found'}), 400
    else:
        return jsonify({'error': 'No faces detected'}), 400

# Recognition route to check if prediction is correct
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    img = Image.open(image)
    pixels = np.array(img)

    # Load known faces
    try:
        with open('known_faces.pkl', 'rb') as f:
            known_faces = pickle.load(f)
    except FileNotFoundError:
        return jsonify({'error': 'No known faces found.'}), 400

    known_names = list(known_faces.keys())
    known_encodings = np.array(list(known_faces.values()))

    # Detect faces and compare
    detector = MTCNN()
    faces = detector.detect_faces(pixels)

    if not faces:
        return jsonify({'error': 'No faces detected in the uploaded image.'}), 400

    face_encodings = face_recognition.face_encodings(pixels)

    # Check for matches
    matched_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        for i, match in enumerate(matches):
            if match:
                matched_names.append(known_names[i])

    if matched_names:
        return jsonify({'matched_names': matched_names})
    else:
        return jsonify({'message': 'No matches found.'})

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve processed images
@app.route('/static/Images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
