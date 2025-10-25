import cv2
import numpy as np

def load_image(path):
    """Carga imagen y convierte a RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_face_landmarks(img):
    """Detecta rostro con Haar Cascade - MEJORADO"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)  # Más preciso

    if len(faces) == 0:
        return (0, 0, img.shape[1], img.shape[0])

    x, y, w, h = faces[0]

    # Ampliar MENOS para evitar incluir fondo
    expand_x = int(w * 0.05)  # Reducido de 0.10
    expand_y = int(h * 0.10)  # Reducido de 0.15
    x = max(0, x - expand_x)
    y = max(0, y - expand_y)
    w = min(img.shape[1] - x, w + 2 * expand_x)
    h = min(img.shape[0] - y, h + 2 * expand_y)

    return (int(x), int(y), int(w), int(h))


def create_smooth_mask(width, height, margin=50):
    """Crea máscara más suave - MEJORADO"""
    mask = np.ones((height, width), dtype=np.float32)

    # Gradiente más amplio
    for i in range(margin):
        alpha = np.sin((i / margin) * np.pi / 2)  # Transición sinusoidal suave

        if i < height:
            mask[i, :] = np.minimum(mask[i, :], alpha)
        if height - i - 1 >= 0:
            mask[height - i - 1, :] = np.minimum(mask[height - i - 1, :], alpha)
        if i < width:
            mask[:, i] = np.minimum(mask[:, i], alpha)
        if width - i - 1 >= 0:
            mask[:, width - i - 1] = np.minimum(mask[:, width - i - 1], alpha)

    # Blur más intenso
    mask = cv2.GaussianBlur(mask, (51, 51), 15)

    return mask


def detect_skin_tone(face):
    """Detecta si la piel es clara, media u oscura"""
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    v_mean = np.mean(hsv[..., 2])  # Valor promedio

    if v_mean > 180:
        return "clara"
    elif v_mean > 120:
        return "media"
    else:
        return "oscura"


def apply_under_eye_bags(face, w, h, intensity, skin_tone):
    """Ojeras adaptadas al tono de piel"""
    if intensity < 0.3:
        return face

    bags_mask = np.zeros((h, w), dtype=np.float32)
    bag_y = int(h * 0.48)

    cv2.ellipse(bags_mask, (int(w * 0.35), bag_y), (int(w * 0.12), int(h * 0.08)), 0, 0, 360, 1.0, -1)
    cv2.ellipse(bags_mask, (int(w * 0.65), bag_y), (int(w * 0.12), int(h * 0.08)), 0, 0, 360, 1.0, -1)

    bags_mask = cv2.GaussianBlur(bags_mask, (31, 31), 0)

    # Color de sombra según tono de piel
    if skin_tone == "clara":
        shadow_color = np.array([130, 115, 105], dtype=np.uint8)  # Más claro
        alpha = min(intensity * 0.25, 0.35)  # Menos intenso
    elif skin_tone == "media":
        shadow_color = np.array([100, 85, 75], dtype=np.uint8)
        alpha = min(intensity * 0.30, 0.40)
    else:
        shadow_color = np.array([70, 60, 50], dtype=np.uint8)
        alpha = min(intensity * 0.35, 0.50)

    shadow_layer = np.ones_like(face) * shadow_color
    bags_mask_3d = np.dstack([bags_mask] * 3)

    face = cv2.addWeighted(face, 1.0, shadow_layer, alpha * bags_mask_3d.mean(), 0)
    face = (face * (1 - bags_mask_3d * alpha * 0.2)).astype(np.uint8)

    return face


def thin_lips(face, w, h, intensity):
    """Adelgaza labios - MEJORADO"""
    if intensity < 0.4:
        return face

    lip_y = int(h * 0.72)
    lip_height = int(h * 0.08)
    lip_x_start = int(w * 0.35)
    lip_x_end = int(w * 0.65)

    lip_region = face[max(0, lip_y - lip_height):min(h, lip_y + lip_height),
                 max(0, lip_x_start):min(w, lip_x_end)]

    if lip_region.size == 0:
        return face

    # Desaturar MENOS
    hsv_lips = cv2.cvtColor(lip_region, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_lips[..., 1] *= (1 - intensity * 0.2)  # Reducido de 0.3
    hsv_lips[..., 2] *= (1 - intensity * 0.15)  # Reducido de 0.2
    hsv_lips = np.clip(hsv_lips, 0, 255)
    lip_region = cv2.cvtColor(np.uint8(hsv_lips), cv2.COLOR_HSV2RGB)

    shrink_factor = 1 - (intensity * 0.12)  # Reducido de 0.15
    new_height = max(1, int(lip_region.shape[0] * shrink_factor))

    thinned_lips = cv2.resize(lip_region, (lip_region.shape[1], new_height))

    blend_mask = np.linspace(0, 1, new_height).reshape(-1, 1)
    blend_mask = np.tile(blend_mask, (1, thinned_lips.shape[1]))
    blend_mask = cv2.GaussianBlur(blend_mask.astype(np.float32), (7, 7), 0)
    blend_mask = np.dstack([blend_mask] * 3)

    y_offset = (lip_region.shape[0] - new_height) // 2
    temp_region = face[max(0, lip_y - lip_height):min(h, lip_y + lip_height),
                  max(0, lip_x_start):min(w, lip_x_end)].copy()

    y_start = y_offset
    y_end = y_start + new_height

    if y_end <= temp_region.shape[0]:
        temp_region[y_start:y_end] = (
                thinned_lips * blend_mask +
                temp_region[y_start:y_end] * (1 - blend_mask)
        ).astype(np.uint8)

        face[max(0, lip_y - lip_height):min(h, lip_y + lip_height),
        max(0, lip_x_start):min(w, lip_x_end)] = temp_region

    return face


def add_wrinkles_around_eyes(face, w, h, intensity):
    """Arrugas alrededor de ojos - MEJORADO"""
    if intensity < 0.5:
        return face

    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 130)  # Umbrales ajustados

    eye_mask_left = np.zeros((h, w), dtype=np.uint8)
    eye_mask_right = np.zeros((h, w), dtype=np.uint8)

    cv2.ellipse(eye_mask_left, (int(w * 0.35), int(h * 0.42)),
                (int(w * 0.15), int(h * 0.12)), 0, 0, 360, 255, -1)
    cv2.ellipse(eye_mask_right, (int(w * 0.65), int(h * 0.42)),
                (int(w * 0.15), int(h * 0.12)), 0, 0, 360, 255, -1)

    eye_mask = cv2.bitwise_or(eye_mask_left, eye_mask_right)
    wrinkles = cv2.bitwise_and(edges, eye_mask)

    kernel = np.ones((2, 2), np.uint8)
    wrinkles = cv2.dilate(wrinkles, kernel, iterations=1)

    wrinkle_layer = np.zeros_like(face)
    wrinkle_layer[wrinkles > 0] = [90, 80, 75]  # Más claro

    alpha = min(intensity * 0.25, 0.35)  # Reducido
    face = cv2.addWeighted(face, 1.0, wrinkle_layer, alpha, 0)

    return face


def apply_transformations(img, params):
    """
    Transformaciones OPTIMIZADAS para piel clara
    """
    wrinkle, desaturation, fade_color, edge_strength = params
    img2 = img.copy()

    x, y, w, h = detect_face_landmarks(img2)
    face = img2[y:y+h, x:x+w].copy()

    if face.size == 0 or w < 10 or h < 10:
        return img2

    # Detectar tono de piel
    skin_tone = detect_skin_tone(face)
    print(f"  🎨 Tono de piel detectado: {skin_tone}")

    # 1. Tonos cálidos - AJUSTADO para piel clara
    if skin_tone == "clara":
        r_add = int(fade_color * 15)  # Reducido
        g_add = int(fade_color * 8)
        b_sub = int(fade_color * 12)
    else:
        r_add = int(fade_color * 20)
        g_add = int(fade_color * 10)
        b_sub = int(fade_color * 15)

    b, g, r = cv2.split(face)
    r = cv2.add(r, r_add)
    g = cv2.add(g, g_add)
    b = cv2.subtract(b, b_sub)
    face = cv2.merge([b, g, r])
    face = np.clip(face, 0, 255).astype(np.uint8)

    # 2. Desaturación y brillo - MUCHO MÁS SUAVE para piel clara
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV).astype(np.float32)

    if skin_tone == "clara":
        hsv[..., 1] *= (1 - desaturation * 0.25)  # MUY reducido (era 0.4)
        hsv[..., 2] *= (1 - fade_color * 0.15)    # MUY reducido (era 0.25)
    elif skin_tone == "media":
        hsv[..., 1] *= (1 - desaturation * 0.35)
        hsv[..., 2] *= (1 - fade_color * 0.20)
    else:
        hsv[..., 1] *= (1 - desaturation * 0.45)
        hsv[..., 2] *= (1 - fade_color * 0.30)

    hsv = np.clip(hsv, 0, 255)
    face = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2RGB)

    # 3. Textura - MUY REDUCIDA para piel clara
    if wrinkle > 0.4:  # Umbral más alto
        if skin_tone == "clara":
            noise_intensity = min(wrinkle * 8, 15)  # MUY reducido
        else:
            noise_intensity = min(wrinkle * 12, 20)

        noise = np.random.normal(0, noise_intensity, face.shape).astype(np.int16)

        noise_mask = np.random.rand(face.shape[0], face.shape[1]) > 0.75  # Menos píxeles
        noise_mask = cv2.GaussianBlur(noise_mask.astype(np.float32), (7, 7), 0)
        noise_mask = np.dstack([noise_mask] * 3)

        face_with_noise = np.clip(face.astype(np.int16) + noise * noise_mask, 0, 255)
        face = face_with_noise.astype(np.uint8)

    # 4. Ojeras adaptadas
    face = apply_under_eye_bags(face, w, h, wrinkle, skin_tone)

    # 5. Labios delgados
    face = thin_lips(face, w, h, wrinkle)

    # 6. Arrugas ojos
    face = add_wrinkles_around_eyes(face, w, h, edge_strength)

    # 7. Manchas de edad - MÁS SUTILES
    if wrinkle > 0.6:  # Umbral más alto
        num_spots = int(wrinkle * 4)  # Menos manchas
        for _ in range(num_spots):
            ry = np.random.randint(int(h * 0.3), int(h * 0.7))
            rx = np.random.randint(int(w * 0.3), int(w * 0.7))
            rs = np.random.randint(2, 4)

            overlay = face.copy()

            if skin_tone == "clara":
                spot_color = (160, 140, 120)  # Más claro
                alpha_spot = 0.06  # Muy sutil
            else:
                spot_color = (140, 120, 100)
                alpha_spot = 0.08

            cv2.circle(overlay, (rx, ry), rs, spot_color, -1)
            face = cv2.addWeighted(face, 1 - alpha_spot, overlay, alpha_spot, 0)

    # 8. Líneas de expresión - MÁS SUTILES
    if edge_strength > 0.5:  # Umbral más alto
        gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_face, 90, 160)  # Umbrales más altos

        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        wrinkles = np.zeros_like(face)

        if skin_tone == "clara":
            wrinkles[edges > 0] = [110, 100, 95]  # Más claro
            alpha = min(edge_strength * 0.12, 0.20)  # Muy reducido
        else:
            wrinkles[edges > 0] = [80, 70, 65]
            alpha = min(edge_strength * 0.15, 0.25)

        face = cv2.addWeighted(face, 1.0, wrinkles, alpha, 0)

    # 9. Cabello gris - SOLO si es muy alto
    if wrinkle > 0.75:  # Umbral más alto
        hair_height = int(h * 0.18)  # Menos área
        if hair_height > 0:
            hair_area = face[:hair_height, :].copy()

            if skin_tone == "clara":
                gray_overlay = np.ones_like(hair_area) * 190  # Más claro
                mix_alpha = 0.15  # Menos intenso
            else:
                gray_overlay = np.ones_like(hair_area) * 180
                mix_alpha = 0.20

            hair_area = cv2.addWeighted(hair_area, 1 - mix_alpha, gray_overlay, mix_alpha, 0)
            face[:hair_height, :] = hair_area

    # 10. Fusión EXTRA SUAVE
    blend_mask = create_smooth_mask(w, h, margin=60)  # Margen más grande
    blend_mask_3d = np.dstack([blend_mask] * 3)

    original_region = img2[y:y+h, x:x+w].copy()

    # Fusión gradual más suave
    blended = (face * blend_mask_3d + original_region * (1 - blend_mask_3d)).astype(np.uint8)

    img2[y:y+h, x:x+w] = blended

    return img2