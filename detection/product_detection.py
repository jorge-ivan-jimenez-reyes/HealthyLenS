import cv2
import torch
from yolov5.utils.general import non_max_suppression, scale_boxes
import ssl

# Deshabilitar la verificación SSL para evitar errores con la descarga de modelos
ssl._create_default_https_context = ssl._create_unverified_context

# Ruta al modelo YOLOv5 (puedes ajustar para que sea relativa a la carpeta `assets`)
MODEL_PATH = "../PP/PPE/best.pt"

def load_model():
    """
    Carga el modelo YOLOv5.
    :return: Modelo YOLOv5 y nombres de las clases.
    """
    print("Cargando modelo YOLOv5...")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True, trust_repo=True)
        model.eval()  # Modo evaluación
        CLASSES = model.names  # Nombres de las clases detectadas
        print("Modelo YOLOv5 cargado correctamente.")
        print("Clases detectadas:", CLASSES)
        return model, CLASSES
    except Exception as e:
        print(f"Error al cargar el modelo YOLOv5: {e}")
        raise

def detect_products(frame, model, CLASSES):
    """
    Detecta productos en un frame utilizando YOLOv5.
    :param frame: Frame de video (BGR).
    :param model: Modelo YOLOv5.
    :param CLASSES: Lista de nombres de las clases.
    :return: Frame con productos detectados.
    """
    # Preprocesar el frame para YOLO
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))  # Redimensionar al tamaño esperado por YOLO
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    # Realizar la detección
    with torch.no_grad():
        results = model(img_tensor)[0]

    # Verificar que los resultados sean válidos
    if results is None or results.numel() == 0:
        print("El modelo no detectó objetos.")
        return frame

    # Aplicar NMS para filtrar detecciones
    detections = non_max_suppression(results, conf_thres=0.4, iou_thres=0.5)[0]
    if detections is None or len(detections) == 0:
        print("No se encontraron detecciones después de NMS.")
        return frame

    # Dibujar detecciones en el frame
    for *xyxy, conf, cls in detections:
        coords = scale_boxes(img_tensor.shape[2:], torch.tensor(xyxy), frame.shape[:2]).round().int()
        x1, y1, x2, y2 = coords.tolist()
        label = f"{CLASSES[int(cls)]} {conf:.2f}"

        # Dibujar las detecciones en el frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
