from ultralytics import YOLO
import cv2
import cvzone
import math

# Definir las clases y sus colores
classNames = ['apple', 'instant_noodle', 'juice', 'orange', 'sandwich']
classColors = {
    'apple': (0, 255, 0),         # Verde
    'instant_noodle': (0, 255, 255),  # Amarillo
    'juice': (255, 165, 0),       # Naranja
    'orange': (255, 0, 0),        # Rojo
    'sandwich': (128, 0, 128)     # Morado
}

def load_model(model_path="best.pt"):
    """
    Carga el modelo YOLO desde el archivo especificado.
    :param model_path: Ruta al modelo YOLO.
    :return: Modelo YOLO cargado.
    """
    print(f"Cargando modelo YOLO desde {model_path}...")
    model = YOLO(model_path)
    print("Modelo YOLO cargado correctamente.")
    return model

def apply_filters(roi, brightness, blur, hue):
    """
    Aplica filtros de brillo, desenfoque y tonalidad a la región de interés.
    :param roi: Región de interés.
    :param brightness: Nivel de brillo.
    :param blur: Nivel de desenfoque.
    :param hue: Valor de tonalidad.
    :return: Región de interés filtrada.
    """
    # Aplicar brillo
    roi = cv2.convertScaleAbs(roi, alpha=brightness / 50, beta=0)

    # Aplicar desenfoque
    if blur > 0:
        roi = cv2.GaussianBlur(roi, (blur * 2 + 1, blur * 2 + 1), 0)

    # Cambiar tonalidad (espacio HSV)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
    roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return roi

def process_frame(frame, model, brightness, blur, hue):
    """
    Procesa el frame con el modelo YOLO y aplica filtros a las regiones detectadas.
    :param frame: Frame de video.
    :param model: Modelo YOLO.
    :param brightness: Nivel de brillo.
    :param blur: Nivel de desenfoque.
    :param hue: Valor de tonalidad.
    :return: Frame procesado.
    """
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordenadas del bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Extraer la región de interés (ROI)
            roi = frame[y1:y2, x1:x2]

            # Aplicar filtros a la ROI
            if roi.size > 0:  # Evitar errores si el ROI está vacío
                roi_filtered = apply_filters(roi, brightness, blur, hue)
                frame[y1:y2, x1:x2] = roi_filtered  # Reemplazar en la imagen original

            # Obtener clase y confianza
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Seleccionar el color basado en la clase detectada
            myColor = classColors.get(currentClass, (0, 0, 255))  # Rojo por defecto si no se encuentra la clase

            # Dibujar bounding box y texto
            cvzone.putTextRect(frame, f'{currentClass} {conf}',
                               (max(0, x1), max(35, y1)),
                               scale=0.5, thickness=1, colorB=myColor,
                               colorT=(255, 255, 255), offset=5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)

    return frame
