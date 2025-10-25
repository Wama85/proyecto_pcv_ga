import cv2
import numpy as np
import os

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se encontró: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
    print(f"✅ Guardado: {path}")

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    if len(faces) == 0:
        print("⚠️ No se detectó rostro, usando toda la imagen")
        return (0, 0, img.shape[1], img.shape[0])

    x, y, w, h = faces[0]
    print(f"✅ Rostro detectado: x={x}, y={y}, w={w}, h={h}")
    return (int(x), int(y), int(w), int(h))

def envejecer_FUERTE(img):
    """
    Envejecimiento MUY VISIBLE
    """
    print("\n🎨 Aplicando transformaciones FUERTES...")
    img2 = img.copy()

    x, y, w, h = detect_face(img2)
    face = img2[y:y+h, x:x+w].copy()

    if face.size == 0:
        print("❌ Error: región facial vacía")
        return img2

    print(f"   📐 Tamaño del rostro: {face.shape}")

    # ===============================
    # 1. DESATURACIÓN FUERTE (pérdida de color)
    # ===============================
    print("   1️⃣ Aplicando desaturación...")
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 0.4  # Reducir saturación a 40%
    hsv[..., 2] *= 0.75  # Reducir brillo a 75%
    hsv = np.clip(hsv, 0, 255)
    face = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2RGB)

    # ===============================
    # 2. TONOS AMARILLENTOS/MARRONES
    # ===============================
    print("   2️⃣ Aplicando tonos envejecidos...")
    b, g, r = cv2.split(face)
    r = cv2.add(r, 40)  # Más rojo
    g = cv2.add(g, 20)  # Más verde
    b = cv2.subtract(b, 30)  # Menos azul
    face = cv2.merge([b, g, r])
    face = np.clip(face, 0, 255).astype(np.uint8)

    # ===============================
    # 3. ARRUGAS PROFUNDAS
    # ===============================
    print("   3️⃣ Añadiendo arrugas profundas...")
    gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

    # Detectar bordes
    edges = cv2.Canny(gray_face, 50, 120)

    # Dilatar para hacer líneas más gruesas
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Crear capa de arrugas oscuras
    wrinkles = np.zeros_like(face)
    wrinkles[edges > 0] = [60, 50, 45]  # Marrón oscuro

    # Aplicar arrugas con 50% de opacidad
    face = cv2.addWeighted(face, 1.0, wrinkles, 0.5, 0)

    # ===============================
    # 4. OJERAS PRONUNCIADAS
    # ===============================
    print("   4️⃣ Añadiendo ojeras...")
    h_f, w_f = face.shape[:2]

    # Máscara para ojeras
    bags_mask = np.zeros((h_f, w_f), dtype=np.float32)

    # Ojera izquierda
    cv2.ellipse(bags_mask,
                (int(w_f * 0.35), int(h_f * 0.50)),
                (int(w_f * 0.12), int(h_f * 0.10)),
                0, 0, 360, 1.0, -1)

    # Ojera derecha
    cv2.ellipse(bags_mask,
                (int(w_f * 0.65), int(h_f * 0.50)),
                (int(w_f * 0.12), int(h_f * 0.10)),
                0, 0, 360, 1.0, -1)

    # Difuminar
    bags_mask = cv2.GaussianBlur(bags_mask, (31, 31), 0)

    # Aplicar oscurecimiento
    bags_mask_3d = np.dstack([bags_mask] * 3)
    face = (face * (1 - bags_mask_3d * 0.4)).astype(np.uint8)

    # ===============================
    # 5. MANCHAS DE EDAD
    # ===============================
    print("   5️⃣ Añadiendo manchas de edad...")
    num_spots = 15
    for _ in range(num_spots):
        spot_y = np.random.randint(int(h_f * 0.2), int(h_f * 0.8))
        spot_x = np.random.randint(int(w_f * 0.2), int(w_f * 0.8))
        spot_size = np.random.randint(3, 7)

        overlay = face.copy()
        cv2.circle(overlay, (spot_x, spot_y), spot_size, (120, 100, 80), -1)
        face = cv2.addWeighted(face, 0.85, overlay, 0.15, 0)

    # ===============================
    # 6. PATAS DE GALLO (arrugas ojos)
    # ===============================
    print("   6️⃣ Añadiendo patas de gallo...")

    # Máscara para zona de ojos
    eye_mask = np.zeros((h_f, w_f), dtype=np.uint8)

    # Ojo izquierdo
    cv2.ellipse(eye_mask, (int(w_f * 0.35), int(h_f * 0.42)),
                (int(w_f * 0.18), int(h_f * 0.15)), 0, 0, 360, 255, -1)

    # Ojo derecho
    cv2.ellipse(eye_mask, (int(w_f * 0.65), int(h_f * 0.42)),
                (int(w_f * 0.18), int(h_f * 0.15)), 0, 0, 360, 255, -1)

    # Aplicar arrugas en zona de ojos
    eye_wrinkles = cv2.bitwise_and(edges, eye_mask)
    eye_wrinkles = cv2.dilate(eye_wrinkles, kernel, iterations=1)

    wrinkle_layer = np.zeros_like(face)
    wrinkle_layer[eye_wrinkles > 0] = [50, 45, 40]
    face = cv2.addWeighted(face, 1.0, wrinkle_layer, 0.6, 0)

    # ===============================
    # 7. LÍNEAS DE EXPRESIÓN (frente)
    # ===============================
    print("   7️⃣ Añadiendo líneas de expresión...")

    # Líneas horizontales en la frente
    forehead_y_start = int(h_f * 0.15)
    forehead_y_end = int(h_f * 0.35)

    for i in range(3):
        line_y = forehead_y_start + i * (forehead_y_end - forehead_y_start) // 3
        x_start = int(w_f * 0.25)
        x_end = int(w_f * 0.75)

        # Línea ondulada
        pts = []
        for x in range(x_start, x_end, 5):
            wave = int(3 * np.sin(x / 20))
            pts.append([x, line_y + wave])

        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(face, [pts], False, (70, 60, 55), 2, cv2.LINE_AA)

    # ===============================
    # 8. LABIOS MÁS DELGADOS Y PÁLIDOS
    # ===============================
    print("   8️⃣ Adelgazando labios...")

    lip_y = int(h_f * 0.72)
    lip_height = int(h_f * 0.10)
    lip_x_start = int(w_f * 0.30)
    lip_x_end = int(w_f * 0.70)

    lip_region = face[max(0, lip_y-lip_height):min(h_f, lip_y+lip_height),
                 max(0, lip_x_start):min(w_f, lip_x_end)]

    if lip_region.size > 0:
        # Desaturar labios
        hsv_lips = cv2.cvtColor(lip_region, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_lips[..., 1] *= 0.5  # Reducir saturación
        hsv_lips[..., 2] *= 0.85  # Oscurecer ligeramente
        hsv_lips = np.clip(hsv_lips, 0, 255)
        lip_region = cv2.cvtColor(np.uint8(hsv_lips), cv2.COLOR_HSV2RGB)

        face[max(0, lip_y-lip_height):min(h_f, lip_y+lip_height),
        max(0, lip_x_start):min(w_f, lip_x_end)] = lip_region

    # ===============================
    # 9. TEXTURA RUGOSA
    # ===============================
    print("   9️⃣ Añadiendo textura rugosa...")
    noise = np.random.normal(0, 25, face.shape).astype(np.int16)
    face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ===============================
    # 10. VIÑETA (oscurecer bordes)
    # ===============================
    print("   🔟 Aplicando viñeta...")
    rows, cols = face.shape[:2]

    mask_x = cv2.getGaussianKernel(cols, cols/2.5)
    mask_y = cv2.getGaussianKernel(rows, rows/2.5)
    mask = mask_y * mask_x.T
    mask = mask / mask.max()
    vignette = np.dstack([mask] * 3)

    face = (face * (0.7 + vignette * 0.3)).astype(np.uint8)

    # ===============================
    # FUSIÓN CON IMAGEN ORIGINAL
    # ===============================
    print("   🔄 Fusionando con imagen original...")

    # Crear máscara de fusión suave
    margin = 30
    mask = np.ones((h, w), dtype=np.float32)

    for i in range(margin):
        alpha = (i + 1) / margin
        if i < h:
            mask[i, :] = np.minimum(mask[i, :], alpha)
        if h - i - 1 >= 0:
            mask[h - i - 1, :] = np.minimum(mask[h - i - 1, :], alpha)
        if i < w:
            mask[:, i] = np.minimum(mask[:, i], alpha)
        if w - i - 1 >= 0:
            mask[:, w - i - 1] = np.minimum(mask[:, w - i - 1], alpha)

    mask = cv2.GaussianBlur(mask, (51, 51), 15)
    mask_3d = np.dstack([mask] * 3)

    original_region = img2[y:y+h, x:x+w].copy()
    blended = (face * mask_3d + original_region * (1 - mask_3d)).astype(np.uint8)

    img2[y:y+h, x:x+w] = blended

    print("✅ Transformaciones completadas")
    return img2


def main():
    print("=" * 70)
    print("👵 ENVEJECIMIENTO FACIAL FUERTE - RESULTADOS VISIBLES")
    print("=" * 70)

    # Cargar imagen
    img_path = "data/base/rostro3.jpg"
    print(f"\n📸 Cargando imagen: {img_path}")

    try:
        img = load_image(img_path)
        print(f"✅ Imagen cargada: {img.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Aplicar envejecimiento
    resultado = envejecer_FUERTE(img)

    # Guardar resultado
    print("\n💾 Guardando resultado...")
    save_image(resultado, "data/results/envejecido_VISIBLE.jpg")

    # Crear comparación lado a lado
    print("📊 Creando comparación...")
    comparison = np.hstack([img, resultado])

    # Agregar texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (20, 50), font, 1.5, (255, 255, 255), 3)
    cv2.putText(comparison, "Envejecido", (img.shape[1] + 20, 50), font, 1.5, (255, 255, 255), 3)

    save_image(comparison, "data/results/comparacion_VISIBLE.jpg")

    # Mostrar
    print("\n🖼️ Mostrando resultados...")
    cv2.imshow("Original", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Envejecido FUERTE", cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR))
    cv2.imshow("Comparacion", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO")
    print("📁 Archivos guardados en: data/results/")
    print("   • envejecido_VISIBLE.jpg")
    print("   • comparacion_VISIBLE.jpg")
    print("\n👉 Presiona cualquier tecla para cerrar...")
    print("=" * 70)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()