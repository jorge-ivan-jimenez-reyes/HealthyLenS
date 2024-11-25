from ultralytics import YOLO
import cv2
import cvzone
import math

# Definir las clases y sus colores
classNames = ['apple', 'instant_noodle', 'juice', 'orange', 'sandwich']
classColors = {
    'apple': (0, 255, 0),          # Verde
    'instant_noodle': (0, 255, 255), # Amarillo
    'juice': (255, 165, 0),        # Naranja
    'orange': (255, 0, 0),         # Rojo
    'sandwich': (128, 0, 128),     # Morado
}

# Saludabilidad de cada clase (0 a 100)
healthiness_score = {
    'apple': 90,               # Muy saludable
    'instant_noodle': 30,      # Poco saludable
    'juice': 60,               # Moderadamente saludable
    'orange': 85,              # Muy saludable
    'sandwich': 50             # Moderado
}

# Filtros según saludabilidad
def get_filter_by_healthiness(score):
    """
    Devuelve el filtro en base al puntaje de saludabilidad.
    """
    if score >= 80:
        return "Gaussian Blur"   # Representa suavidad
    elif 50 <= score < 80:
        return "Smooth"          # Suavizado ligero
    elif 30 <= score < 50:
        return "Sobel"           # Más definido
    else:
        return "Laplacian"       # Más marcado, para productos no saludables

def load_model(model_path="PP/PPE/best.pt"):
    """
    Carga el modelo YOLO desde el archivo especificado.
    """
    print(f"Cargando modelo YOLO desde {model_path}...")
    model = YOLO(model_path)
    print("Modelo YOLO cargado correctamente.")
    return model

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

            # Obtener clase y puntaje de saludabilidad
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = classNames[cls]
            health_score = healthiness_score.get(current_class, 0)
            filter_type = get_filter_by_healthiness(health_score)

            # Aplicar filtro según la saludabilidad
            if roi.size > 0:
                roi_filtered = apply_filter(roi, filter_type)
                frame[y1:y2, x1:x2] = roi_filtered  # Reemplazar en la imagen original

            # Dibujar bounding box y texto
            cvzone.putTextRect(frame, f'{current_class} {conf}%',
                               (max(0, x1), max(35, y1)),
                               scale=0.5, thickness=1, colorB=(0, 255, 0),
                               colorT=(255, 255, 255), offset=5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Mostrar filtro aplicado
            cv2.putText(frame, f"Filtro: {filter_type}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame
