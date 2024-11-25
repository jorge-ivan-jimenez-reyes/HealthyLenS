from ultralytics import YOLO
import cv2
import cvzone
import math

# Definir las clases y sus colores
classNames = ['apple', 'instant_noodle', 'juice', 'orange', 'sandwich']
classColors = {
    'apple': (0, 255, 0),
    'instant_noodle': (0, 255, 255),
    'juice': (255, 165, 0),
    'orange': (255, 0, 0),
    'sandwich': (128, 0, 128),
}

def load_model(model_path="PP/PPE/best.pt"):
    """
    Carga el modelo YOLO desde el archivo especificado.
    """
    print(f"Cargando modelo YOLO desde {model_path}...")
    model = YOLO(model_path)
    print("Modelo YOLO cargado correctamente.")
    return model

def apply_filters(roi, brightness, blur, hue):
    """
    Aplica filtros de brillo, desenfoque y tonalidad a la región de interés.
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

def process_frame(frame, model, brightness=50, blur=0, hue=0):
    """
    Procesa el frame con el modelo YOLO y aplica filtros opcionales a las regiones detectadas.
    """
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordenadas del bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Extraer la región de interés (ROI)
            roi = frame[y1:y2, x1:x2]

            # Aplicar filtros solo si los valores de brillo, desenfoque o tonalidad son diferentes al predeterminado
            if roi.size > 0 and (brightness != 50 or blur != 0 or hue != 0):
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
