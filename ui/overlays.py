def draw_overlays(frame, products, emotions):
    for (x, y, w, h) in products:
        cv2.putText(frame, 'Producto', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame
