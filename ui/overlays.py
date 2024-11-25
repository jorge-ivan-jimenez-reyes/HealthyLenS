import cv2

def draw_overlays(frame, products):
    """
    Dibuja información de productos detectados en el frame.
    :param frame: Frame de video (BGR).
    :param products: Lista de productos detectados [(x1, y1, x2, y2, label)].
    """
    for (x1, y1, x2, y2, label) in products:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
