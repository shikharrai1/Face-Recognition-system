import cv2
import os


PERSON_NAME = input("Enter person's name: ").strip()
BASE_DIR = "images"
SAVE_DIR = os.path.join(BASE_DIR, PERSON_NAME)
MAX_IMAGES = 30
# ------------------------

if not PERSON_NAME:
    raise ValueError("Person name cannot be empty!")


os.makedirs(SAVE_DIR, exist_ok=True)

# Load Haar Cascade
CASCADE_PATH = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise IOError("Haar cascade file not found or failed to load!")

# Start webcam-- by default it will use system's webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam")

count = 0

print("Instructions:")
print("Press 'c' to capture face")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    #  face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"{PERSON_NAME}: {count}/{MAX_IMAGES}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow(f"Face Capture - {PERSON_NAME}", frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture only if exactly one face 
    if key == ord('c') and len(faces) == 1:
        x, y, w, h = faces[0]

        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        face_img = gray[y1:y2, x1:x2]

        # Safety check before resize
        if face_img.size != 0:
            face_img = cv2.resize(face_img, (200, 200))

            img_path = os.path.join(SAVE_DIR, f"img_{count + 1}.jpg")
            cv2.imwrite(img_path, face_img)

            count += 1
            print(f"Captured: {img_path}")

    if key == ord('q') or count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
