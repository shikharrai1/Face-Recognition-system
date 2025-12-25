import os
import cv2
import pickle
import face_recognition

IMAGES_DIR = "images"
ENCODINGS_DIR = "encodings"
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "face_encodings.pkl")

os.makedirs(ENCODINGS_DIR, exist_ok=True)

known_encodings = []
known_names = []

print("[INFO] Encoding faces...")

for person_name in os.listdir(IMAGES_DIR):
    person_path = os.path.join(IMAGES_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing {person_name}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = face_recognition.load_image_file(img_path)

        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) == 0:
            print(f"[WARNING] No face found in {img_path}")
            continue

        known_encodings.append(face_encodings[0])
        known_names.append(person_name)

# Save encodings
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encoding completed and saved.")
