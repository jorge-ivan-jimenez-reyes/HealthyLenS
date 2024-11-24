import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def detect_hands(frame):
    """
    Detecta y dibuja manos en el cuadro de video.
    :param frame: Frame de video (BGR).
    :return: Frame con manos detectadas y sus conexiones.
    """
    # Convertir el frame a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el frame con MediaPipe Hands
    results = hands.process(frame_rgb)

    # Dibujar las manos detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame, results
