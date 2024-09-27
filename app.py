from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image, ImageDraw
import numpy as np
import face_recognition
import pickle
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Create necessary folders
if not os.path.exists('Images'):
    os.makedirs('Images')
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

IMAGES_FOLDER = 'Images'
UPLOAD_FOLDER = 'static/uploads'
KNOWN_FACES_FILE = 'known_faces.pkl'

# Function to save uploaded image
def save_uploaded_image(image, name):
    image_path = os.path.join(IMAGES_FOLDER, f'{name}.jpg')
    image.save(image_path)
    return image_path

# Function to save face encodings
def save_face_encodings(image, name):
    pixels = np.array(image)
    face_encodings = face_recognition.face_encodings(pixels)

    if len(face_encodings) > 0:
        try:
            with open(KNOWN_FACES_FILE, 'rb') as f:
                known_faces = pickle.load(f)
        except FileNotFoundError:
            known_faces = {}

        known_faces[name] = face_encodings[0]

        with open(KNOWN_FACES_FILE, 'wb') as f:
            pickle.dump(known_faces, f)
        return True
    return False

# Load known faces from the dataset
def load_known_faces():
    known_faces = {}
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, 'rb') as f:
            known_faces = pickle.load(f)
    return known_faces

# Compare new faces with known encodings
def compare_faces(face_encodings, known_faces):
    recognized_names = []

    for encoding in face_encodings:
        match_found = False
        for name, known_encoding in known_faces.items():
            # Use a tolerance to decide whether the faces match
            results = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.6)
            if results[0]:
                recognized_names.append(name)
                match_found = True
                break
        if not match_found:
            recognized_names.append("Match not found")
    return recognized_names

# Route for the home page with buttons
@app.route('/')
def index():
    return render_template('index.html')

# Handle the upload and dataset functionalities
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    image = Image.open(file.stream)

    # Save the image temporarily to display it later
    temp_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    image.save(temp_image_path)

    if request.form.get('action') == 'dataset':
        # For dataset button, save with the name entered
        name = request.form.get('name')
        if name:
            save_uploaded_image(image, name)
            save_face_encodings(image, name)
            return f"Image saved and face encoded for {name}."
        else:
            return "Name required for dataset!"

    elif request.form.get('action') == 'upload':
        # For upload button, process for recognition
        detector = MTCNN()
        pixels = np.array(image)
        faces = detector.detect_faces(pixels)

        if faces:
            known_faces = load_known_faces()

            # Extract the face locations for recognition
            face_locations = [(face['box'][1], face['box'][0] + face['box'][2],
                               face['box'][1] + face['box'][3], face['box'][0]) for face in faces]

            # Get the face encodings for all detected faces
            face_encodings = face_recognition.face_encodings(pixels, face_locations)

            # Compare each face encoding with known faces
            recognized_names = compare_faces(face_encodings, known_faces)

            # Draw rectangles and names on the image around detected faces
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()  # Load default PIL font

            for face, name in zip(faces, recognized_names):
                x, y, width, height = face['box']
                draw.rectangle(((x, y), (x + width, y + height)), outline="red", width=3)

                # Draw name above the rectangle
                draw.text((x, y - 10), name, fill="green", font=font)

            # Save the image with rectangles and names in the 'static/uploads' folder
            image_with_boxes_path = os.path.join(UPLOAD_FOLDER, 'boxed_' + file.filename)
            image.save(image_with_boxes_path)

            # Only pass the filename to the template
            return render_template('result.html', image_path='boxed_' + file.filename)
        else:
            return "No faces detected."

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
