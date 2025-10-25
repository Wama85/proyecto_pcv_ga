import cv2
import numpy as np

def load_image(path):
    """Carga imagen y convierte a RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_face_landmarks(img):
    """Detecta rostro usando Haar Cascade"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        # ✅ Devolver tupla en lugar de array
        return (0, 0, img.shape[1], img.shape[0])

    # ✅ Convertir el primer rostro a tupla
    x, y, w, h = faces[0]
    return (int(x), int(y), int(w), int(h))


def apply_transformations(img, params):
    """
    Aplica transformaciones de envejecimiento al rostro

    Args:
        img: Imagen RGB
        params: [wrinkle, desaturation, fade_color, edge_strength]

    Returns:
        Imagen envejecida
    """
    wrinkle, desaturation, fade_color, edge_strength = params
    img2 = img.copy()

    x, y, w, h = detect_face_landmarks(img2)
    face = img2[y:y+h, x:x+w]

    # Validar que la región del rostro no esté vacía
    if face.size == 0:
        return img2

    # --- 1️⃣ Colores envejecidos (más cálidos y apagados) ---
    b, g, r = cv2.split(face)
    r = cv2.add(r, int(fade_color * 40))
    g = cv2.add(g, int(fade_color * 25))
    b = cv2.subtract(b, int(fade_color * 30))
    face = cv2.merge([b, g, r])
    face = np.clip(face, 0, 255).astype(np.uint8)

    # --- 2️⃣ Desaturación y brillo reducido ---
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= (1 - desaturation * 1.2)  # Reducir saturación
    hsv[..., 2] *= (1 - fade_color * 0.7)    # Reducir brillo
    hsv = np.clip(hsv, 0, 255)
    face = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2RGB)

    # --- 3️⃣ Textura rugosa tipo papel ---
    noise = np.random.normal(0, wrinkle * 70, face.shape).astype(np.int16)
    face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # --- 4️⃣ Bordes oscuros (grietas o líneas) ---
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    iterations = max(1, int(edge_strength * 3) + 1)
    edges = cv2.dilate(edges, None, iterations=iterations)
    cracks = np.zeros_like(face)
    cracks[edges > 0] = [40, 40, 40]
    face = cv2.addWeighted(face, 1, cracks, 0.4, 0)

    # --- 5️⃣ Viñeta suave sin marcos negros ---
    rows, cols = face.shape[:2]
    if rows > 0 and cols > 0:  # Validación adicional
        mask_x = cv2.getGaussianKernel(cols, cols/4)
        mask_y = cv2.getGaussianKernel(rows, rows/4)
        mask = mask_y * mask_x.T
        mask = mask / mask.max()
        vignette = np.dstack([mask]*3)
        face = (face * vignette + 30 * (1 - vignette)).astype(np.uint8)

    # --- 6️⃣ ✅ NUEVO: Manchas de edad (lentigos) ---
    if wrinkle > 0.5:  # Solo si hay suficiente envejecimiento
        num_spots = int(wrinkle * 15)
        for _ in range(num_spots):
            # Posición aleatoria
            spot_y = np.random.randint(rows // 4, 3 * rows // 4)
            spot_x = np.random.randint(cols // 4, 3 * cols // 4)
            spot_size = np.random.randint(2, 5)

            # Color más oscuro
            face[max(0, spot_y-spot_size):min(rows, spot_y+spot_size),
            max(0, spot_x-spot_size):min(cols, spot_x+spot_size)] = \
                face[max(0, spot_y-spot_size):min(rows, spot_y+spot_size),
                max(0, spot_x-spot_size):min(cols, spot_x+spot_size)] * 0.7

    # --- Fusionar suavemente con la imagen original ---
    overlay = img2.copy()
    overlay[y:y+h, x:x+w] = face
    img2 = cv2.addWeighted(img2, 0.7, overlay, 0.3, 0)

    return img2