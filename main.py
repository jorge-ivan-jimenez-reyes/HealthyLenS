import cv2
from detection.hand_detection import detect_hands
from detection.product_detection import detect_products
from detection.emotion_detection import detect_emotions
from processing.filters import apply_filters
from processing.recommendations import generate_recommendations
from ui.sliders import setup_sliders
from ui.overlays import draw_overlays

def main():
    # Inicializar captura de video desde la cámara web
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    # Configuración inicial de sliders
    window_name = "HealthyLens AI"
    cv2.namedWindow(window_name)
    setup_sliders(window_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break

        # Detección de manos
        frame, hand_results = detect_hands(frame)

        # Detección de productos
        products = detect_products(frame)

        # Detección de emociones del usuario
        emotions = detect_emotions(frame)

        # Aplicar filtros visuales basados en productos y emociones
        filtered_frame = apply_filters(frame, products, emotions)

        # Dibujar etiquetas y overlays en el video
        draw_overlays(filtered_frame, products, emotions)

        # Mostrar el video procesado en la ventana
        cv2.imshow(window_name, filtered_frame)

        # Tecla 'q' para salir
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Liberar recursos al salir
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
