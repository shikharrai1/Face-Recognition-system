# Face Recognition System (OpenCV + face_recognition)

This project is a **simple end-to-end face recognition system** built using **Python**, **OpenCV**, and the **face_recognition** library.

The system works in three clear stages:
1. Capture face images of a person
2. Encode all captured faces
3. Recognize faces in real time using a webcam

---

## Project Structure

project-root/
├── images/
│ └── .gitkeep
├── encodings/
│ └── .gitkeep
├── capture_faces.py
├── encode_faces.py
├── recognize_faces.py
├── .gitignore
└── README.md

> **Important**  
> The `images/` folder is intentionally kept empty in the repository.  
> Person-specific folders (e.g. `images/Shikhar/`) are **created automatically at runtime** and are **not pushed to GitHub** for privacy reasons.

---

## How the System Works (High-Level Flow)

User Input (Name)
↓
capture_faces.py
↓
images/<person_name>/
↓
encode_faces.py
↓
encodings/face_encodings.pkl
↓
recognize_faces.py (Live Recognition)

---

## Step-by-Step Usage Guide

### 1. Install Dependencies

Make sure Python is installed, then run:

```bash
pip install opencv-python face-recognition 
```

### 2️. Capture Face Images

Run:

`python capture_faces.py`

 Here:

User(You) will be asked to enter a person’s name

A folder with this name is created automatically inside images/

Example:

images/
└── Shikhar/

Controls:

Press c → Capture face image

Press q → Quit capture

 Only one face is captured per image to keep the dataset clean.
 Each face is padded and resized to ensure better recognition accuracy.

 ### 3️. Encode All Faces

After capturing images, run:

`python encode_faces.py`

What this does:

Reads all folders inside images/

Treats each folder name as a person label

Converts face images into numerical encodings

Saves them to:

encodings/face_encodings.pkl


 Can safely run this file multiple times
 Old encodings are overwritten with fresh ones

 ### 4️. Recognize Faces (Live)

Run:

`python recognize_faces.py`

Then :

Webcam starts

Faces are detected in real time

Live faces are matched against stored encodings

The recognized person’s name is displayed on screen

### Privacy 

Face images are not committed to GitHub

The images/ folder exists only as a runtime container

Each user generates their own dataset locally

This keeps the repository clean, lightweight, and privacy-safe

### Technologies Used

Python

OpenCV

face_recognition (dlib-based)

NumPy


### Key Design Decisions

Face detection and face recognition are handled separately

Folder names act as labels (supervised learning)

Encodings are regenerated instead of incrementally updated

Dataset creation is manual and controlled for accuracy

