import cv2

def apply_filters(frame, products, emotions):
    # Filtro: Suavizado si se detectan emociones negativas
    if emotions:
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
    return frame
