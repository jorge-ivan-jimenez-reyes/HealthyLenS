import cv2

def setup_sliders(window_name):
    cv2.createTrackbar('Brillo', window_name, 50, 100, lambda x: None)
    cv2.createTrackbar('Saturación', window_name, 50, 100, lambda x: None)
