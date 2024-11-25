import cv2
import mediapipe as mp

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Filtros disponibles
filters = ["Gaussian Blur", "Sobel", "Laplacian", "Smooth"]
filter_colors = {
    "Gaussian Blur": (255, 0, 0),   # Azul
    "Sobel": (0, 255, 0),          # Verde
    "Laplacian": (0, 0, 255),      # Rojo
    "Smooth": (255, 255, 0)        # Amarillo
}

def detect_and_filter_objects():
    """
    Detecta gestos de la mano y aplica filtros a cada objeto detectado.
    """
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        print("Iniciando detección de objetos y aplicación de filtros. Presiona 'q' para salir.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el video.")
                break

            # Convertir frame a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar frame
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar puntos de referencia en la mano
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Obtener ROI de la mano
                    x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, frame.shape)
                    roi = frame[y_min:y_max, x_min:x_max]

                    # Aplicar filtros
                    if roi.size > 0:
                        filter_type = get_filter_by_region(x_min, x_max)
                        roi_filtered = apply_filter(roi, filter_type)
                        frame[y_min:y_max, x_min:x_max] = roi_filtered

                        # Dibujar el bounding box y el filtro aplicado
                        color = filter_colors.get(filter_type, (255, 255, 255))
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        cv2.putText(frame, filter_type, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Detección y Filtros", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def get_bounding_box(hand_landmarks, shape):
    """
    Calcula el bounding box de una mano dada por los puntos de referencia.
    """
    h, w, _ = shape
    x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
    y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
    x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
    y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
    return x_min, y_min, x_max, y_max

def get_filter_by_region(x_min, x_max):
    """
    Selecciona un filtro basado en la posición del objeto.
    """
    width = x_max - x_min
    if width < 150:
        return "Gaussian Blur"
    elif 150 <= width < 300:
        return "Sobel"
    elif 300 <= width < 450:
        return "Laplacian"
    else:
        return "Smooth"

def apply_filter(roi, filter_type):
    """
    Aplica un filtro a la región de interés.
    """
    if filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(roi, (15, 15), 0)
    elif filter_type == "Sobel":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        return cv2.merge((abs_sobel_x, abs_sobel_y, gray))
    elif filter_type == "Laplacian":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        abs_laplacian = cv2.convertScaleAbs(laplacian)
        return cv2.merge((abs_laplacian, abs_laplacian, abs_laplacian))
    elif filter_type == "Smooth":
        return cv2.blur(roi, (15, 15))
    return roi
