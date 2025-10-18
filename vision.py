import cv2
import numpy as np
import os

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_face_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        raise ValueError("No se detectó rostro.")
    return faces[0]  # (x, y, w, h)

def apply_transformations(img, params):
    wrinkle, brightness, saturation = params
    img2 = img.copy()

    # Detectar rostro
    x, y, w, h = detect_face_landmarks(img2)
    face = img2[y:y+h, x:x+w].astype(np.float32)

    # --- 1️⃣ Ajustar brillo suavemente ---
    brillo_val = np.interp(brightness, [0, 1], [-30, 30])
    face = cv2.convertScaleAbs(face, alpha=1.0, beta=brillo_val)

    # --- 2️⃣ Ajustar saturación ---
    hsv = cv2.cvtColor(face.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= np.interp(saturation, [0, 1], [0.7, 1.2])
    hsv = np.clip(hsv, 0, 255)
    face = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- 3️⃣ Añadir arrugas suaves (ruido con blur) ---
    noise = np.random.normal(0, wrinkle * 20, face.shape).astype(np.float32)
    noisy_face = np.clip(face + noise, 0, 255)
    face = cv2.addWeighted(face, 0.7, noisy_face.astype(np.uint8), 0.3, 0)

    # --- 4️⃣ Resaltar bordes sutilmente ---
    gray = cv2.cvtColor(face.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    face[edges > 100] = [80, 80, 80]  # arrugas más naturales

    # --- 5️⃣ Mezclar textura opcional ---
    texture_path = "data/base/wrinkles_overlay.jpg"
    if os.path.exists(texture_path):
        texture = cv2.imread(texture_path)
        texture = cv2.resize(texture, (w, h))
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        face = cv2.addWeighted(face, 1, texture, wrinkle * 0.3, 0)

    img2[y:y+h, x:x+w] = face.astype(np.uint8)
    return img2
