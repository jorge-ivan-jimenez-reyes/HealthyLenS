import cv2

def apply_filters(roi, brightness, blur, hue):
    roi = cv2.convertScaleAbs(roi, alpha=brightness / 50, beta=0)
    if blur > 0:
        roi = cv2.GaussianBlur(roi, (blur * 2 + 1, blur * 2 + 1), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
    roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return roi
