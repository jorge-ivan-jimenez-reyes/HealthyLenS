import cv2
import numpy as np
from config import COLOR_RANGES

def detect_products(frame):
    # Segmentación por color en espacio HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = []
    for color, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        masks.append((color, mask))

    # Dibujar contornos para cada color
    for color, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
