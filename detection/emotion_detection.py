import cv2
import mediapipe as mp


def detect_emotions(frame):
    """
    Detecta emociones en el frame dado utilizando MediaPipe.
    :param frame: Frame de video en formato BGR.
    :return: Resultados de detección de emociones.
    """
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Convertir el frame a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar emociones/rostros
    results = face_detection.process(frame_rgb)

    return results
