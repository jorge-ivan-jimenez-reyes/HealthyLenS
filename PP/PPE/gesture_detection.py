import cv2
import mediapipe as mp
import numpy as np

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Filtros disponibles
filters = [
    "Gaussian Blur", "Sobel", "Laplacian", "Smooth", "Negative",
    "Sepia", "Color Shift", "Canny", "Pixelation", "Solarization"
]

filter_colors = {
    "Gaussian Blur": (255, 0, 0),   # Azul
    "Sobel": (0, 255, 0),          # Verde
    "Laplacian": (0, 0, 255),      # Rojo
    "Smooth": (255, 255, 0),       # Amarillo
    "Negative": (0, 255, 255),     # Cyan
    "Sepia": (112, 66, 20),        # Marrón
    "Color Shift": (255, 0, 255),  # Magenta
    "Canny": (255, 255, 255),      # Blanco
    "Pixelation": (0, 128, 255),   # Naranja
    "Solarization": (128, 0, 128)  # Púrpura
}

# Mapear gestos a filtros
gesture_to_filter = {
    "Open Hand": "Gaussian Blur",
    "Fist": "Sobel",
    "Pointing": "Laplacian",
    "Victory": "Smooth",
    "Thumbs Up": "Negative",
    "Thumbs Down": "Sepia",
    "Palm Sideways": "Color Shift",
    "OK Sign": "Canny",
    "Five Fingers Spread": "Pixelation",
    "Two Fingers Crossed": "Solarization"
}

def detect_and_apply_filters():
    """
    Detecta gestos de la mano y aplica un filtro global a toda la pantalla basado en el gesto.
    """
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        print("Iniciando detección de gestos y aplicación de filtros. Presiona 'q' para salir.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el video.")
                break

            # Convertir frame a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar frame con Mediapipe
            results = hands.process(rgb_frame)

            gesture = "No Gesture"
            filter_type = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar puntos de referencia en la mano
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Detectar gesto basado en posiciones clave
                    gesture = detect_gesture(hand_landmarks)
                    filter_type = gesture_to_filter.get(gesture)

            # Aplicar el filtro global si se detectó un gesto
            if filter_type:
                frame = apply_filter(frame, filter_type)

            # Mostrar el gesto detectado
            cv2.putText(frame, f"Gesto: {gesture}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Mostrar el filtro aplicado
            if filter_type:
                cv2.putText(frame, f"Filtro Aplicado: {filter_type}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, filter_colors.get(filter_type, (255, 255, 255)), 2)

            cv2.imshow("Detección de Gestos y Filtros", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def detect_gesture(hand_landmarks):
    """
    Detecta el gesto realizado basado en las posiciones clave de los dedos.
    """
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    if index_tip.y < wrist.y and pinky_tip.y < wrist.y:
        return "Open Hand"
    elif index_tip.y > wrist.y and pinky_tip.y > wrist.y:
        return "Fist"
    elif index_tip.y < wrist.y and pinky_tip.y > wrist.y:
        return "Pointing"
    elif index_tip.y < wrist.y and thumb_tip.y < wrist.y:
        return "Victory"
    elif thumb_tip.y < wrist.y and index_tip.y > wrist.y:
        return "Thumbs Up"
    elif thumb_tip.y > wrist.y and index_tip.y > wrist.y:
        return "Thumbs Down"
    elif pinky_tip.x > wrist.x and thumb_tip.x < wrist.x:
        return "Palm Sideways"
    elif abs(index_tip.x - thumb_tip.x) < 0.05:
        return "OK Sign"
    elif all(landmark.y < wrist.y for landmark in [index_tip, pinky_tip]):
        return "Five Fingers Spread"
    elif index_tip.x < thumb_tip.x and pinky_tip.x > thumb_tip.x:
        return "Two Fingers Crossed"
    return "No Gesture"

def apply_filter(frame, filter_type):
    """
    Aplica un filtro a toda la pantalla basado en el tipo de filtro.
    """
    if filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif filter_type == "Sobel":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        return cv2.merge((abs_sobel_x, abs_sobel_y, gray))
    elif filter_type == "Laplacian":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        abs_laplacian = cv2.convertScaleAbs(laplacian)
        return cv2.merge((abs_laplacian, abs_laplacian, abs_laplacian))
    elif filter_type == "Smooth":
        return cv2.blur(frame, (15, 15))
    elif filter_type == "Negative":
        return cv2.bitwise_not(frame)
    elif filter_type == "Sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(frame, sepia_filter)
    elif filter_type == "Color Shift":
        b, g, r = cv2.split(frame)
        return cv2.merge((r, g, b))
    elif filter_type == "Canny":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.merge((edges, edges, edges))
    elif filter_type == "Pixelation":
        h, w = frame.shape[:2]
        temp = cv2.resize(frame, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    elif filter_type == "Solarization":
        return cv2.bitwise_not(cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1])
    return frame
