import cv2

def run_haar_detection():
    """
    Detección de objetos con clasificadores Haar y aplicación de filtros en tiempo real.
    """
    brightness, blur, hue = 50, 1, 0
    haar_cascade = cv2.CascadeClassifier("assets/haarcascades/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Crear ventana para sliders
    cv2.namedWindow("Controles")
    cv2.createTrackbar("Brillo", "Controles", brightness, 100, lambda x: None)
    cv2.createTrackbar("Desenfoque", "Controles", blur, 20, lambda x: None)
    cv2.createTrackbar("Tonalidad", "Controles", hue, 180, lambda x: None)

    print("Iniciando detección con clasificadores Haar. Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break

        # Obtener valores de sliders
        brightness = cv2.getTrackbarPos("Brillo", "Controles")
        blur = cv2.getTrackbarPos("Desenfoque", "Controles")
        hue = cv2.getTrackbarPos("Tonalidad", "Controles")

        # Detección Haar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in objects:
            roi = frame[y:y + h, x:x + w]
            if roi.size > 0:
                roi = apply_filters(roi, brightness, blur, hue)
                frame[y:y + h, x:x + w] = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Detección y Filtros", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def apply_filters(roi, brightness, blur, hue):
    """
    Aplica filtros de brillo, desenfoque y tonalidad a una región de interés.
    """
    roi = cv2.convertScaleAbs(roi, alpha=brightness / 50, beta=0)
    if blur > 0:
        roi = cv2.GaussianBlur(roi, (blur * 2 + 1, blur * 2 + 1), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
    roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return roi
