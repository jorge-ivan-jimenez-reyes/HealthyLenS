from ultralytics import YOLO
import cv2
import os  # Importar el módulo para verificar rutas de archivos

# Lista de clases del dataset COCO (puedes reducirla según lo que necesites)
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


def load_yolo_model():
    """
    Carga el modelo YOLO desde un archivo local.
    """
    model_path = "PP/PPE/weights/yolov8n.pt"  # Ruta al archivo descargado
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo YOLOv8 no se encuentra en la ruta especificada: {model_path}")

    print(f"Cargando modelo YOLO desde {model_path}...")
    model = YOLO(model_path)  # Cargar el modelo desde la ruta local
    print("Modelo YOLO cargado correctamente.")
    return model
def process_frame(frame, model):
    """
    Procesa un frame para detectar objetos usando YOLO y dibuja los resultados.
    """
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = classNames[cls]
            conf = round(float(box.conf[0]) * 100, 2)

            # Dibujar bounding box y etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def run_yolo_detection():
    """
    Ejecuta la detección de objetos usando YOLO en tiempo real.
    """
    model = load_yolo_model()
    cap = cv2.VideoCapture(1)  # Cambia a 0 si usas una cámara externa
    cap.set(3, 1280)  # Ancho de la ventana
    cap.set(4, 720)   # Alto de la ventana

    print("Iniciando detección de objetos. Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break

        # Procesar el frame
        frame = process_frame(frame, model)
        cv2.imshow("Detección YOLO", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
