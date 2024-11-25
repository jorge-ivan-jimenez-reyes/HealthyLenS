import cv2
from deepface import DeepFace


def detect_emotion():
    """
    Detecta emociones faciales y aplica un filtro dependiendo del estado de ánimo.
    """
    haar_cascade = cv2.CascadeClassifier("assets/haarcascades/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("Iniciando detección de emociones. Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in objects:
            roi = frame[y:y + h, x:x + w]
            emotion = "unknown"
            applied_filter = "none"
            if roi.size > 0:
                try:
                    # Usar DeepFace para detectar emociones
                    result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)

                    # Manejar si el resultado es una lista
                    if isinstance(result, list):
                        result = result[0]

                    emotion = result.get('dominant_emotion', 'unknown')
                    print(f"Emoción detectada: {emotion}")

                    # Aplicar filtro según la emoción
                    if emotion == "happy":
                        roi = apply_happy_filter(roi)
                        applied_filter = "Brillo"
                    elif emotion == "sad":
                        roi = apply_sad_filter(roi)
                        applied_filter = "Desenfoque"
                    elif emotion == "fear":
                        roi = apply_fear_filter(roi)
                        applied_filter = "Tonalidad"

                    frame[y:y + h, x:x + w] = roi
                except Exception as e:
                    print(f"Error al detectar emoción: {e}")

            # Dibujar bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Agregar etiqueta con emoción y filtro aplicado
            label = f"Emoción: {emotion}, Filtro: {applied_filter}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Detección de Emociones", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def apply_happy_filter(roi):
    """
    Aplica un filtro de brillo a la región de interés.
    """
    return cv2.convertScaleAbs(roi, alpha=1.5, beta=50)


def apply_sad_filter(roi):
    """
    Aplica un filtro de desenfoque a la región de interés.
    """
    return cv2.GaussianBlur(roi, (15, 15), 0)


def apply_fear_filter(roi):
    """
    Cambia la tonalidad de la región de interés.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 50) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
