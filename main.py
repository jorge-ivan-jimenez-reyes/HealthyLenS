import tkinter as tk
from tkinter import messagebox
from PP.PPE.pdetection import load_model, process_frame  # YOLO detection
from PP.PPE.detection_haar import run_yolo_detection  # YOLO preentrenado en COCO
from PP.PPE.emotion_detection import detect_emotion  # Detección de emociones
from PP.PPE.gesture_detection import detect_and_apply_filters  # Detección de gestos
import cv2


def run_object_classification():
    """
    Ejecuta la clasificación de objetos con YOLO personalizado.
    """
    model = load_model()
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("Iniciando clasificación de objetos. Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break

        # Procesar frame con YOLO
        frame = process_frame(frame, model, 50, 0, 0)  # Sin filtros adicionales
        cv2.imshow("Clasificación de Objetos", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_general_object_detection():
    """
    Ejecuta la detección de objetos generales con YOLO (modelo COCO preentrenado).
    """
    run_yolo_detection()


def run_filter_application():
    """
    Ejecuta la aplicación de filtros con clasificadores Haar.
    """
    messagebox.showinfo("Opción No Disponible", "Próximamente se integrarán más opciones de filtrado avanzado.")


def run_emotion_detection():
    """
    Ejecuta la detección de emociones faciales.
    """
    detect_emotion()


def run_gesture_detection():
    """
    Ejecuta la detección de gestos de manos con aplicación de filtros globales.
    """
    detect_and_apply_filters()


def show_about():
    """
    Muestra información sobre la aplicación.
    """
    messagebox.showinfo(
        "Acerca de",
        "Healthy Lens\n\nSistema integral de detección y clasificación.\n\n"
        "Incluye detección de objetos (YOLO), emociones y gestos.\n\n"
        "Desarrollado con Python, OpenCV, Mediapipe y YOLOv8."
    )


def main():
    """
    Interfaz gráfica para el menú principal.
    """
    root = tk.Tk()
    root.title("Healthy Lens")
    root.geometry("600x600")

    # Título
    tk.Label(root, text="Healthy Lens", font=("Helvetica", 28, "bold")).pack(pady=20)

    # Botones del menú
    tk.Button(root, text="Clasificación de Objetos Personalizados", font=("Helvetica", 16),
              command=run_object_classification).pack(pady=10)
    tk.Button(root, text="Detección de Objetos Generales (COCO)", font=("Helvetica", 16),
              command=run_general_object_detection).pack(pady=10)
    tk.Button(root, text="Detección y Filtros (Clasificadores Haar)", font=("Helvetica", 16),
              command=run_filter_application).pack(pady=10)
    tk.Button(root, text="Detección de Emociones", font=("Helvetica", 16),
              command=run_emotion_detection).pack(pady=10)
    tk.Button(root, text="Detección de Gestos (Manos)", font=("Helvetica", 16),
              command=run_gesture_detection).pack(pady=10)
    tk.Button(root, text="Acerca de", font=("Helvetica", 16), command=show_about).pack(pady=10)
    tk.Button(root, text="Salir", font=("Helvetica", 16), command=root.destroy).pack(pady=20)

    # Iniciar la interfaz gráfica
    root.mainloop()


if __name__ == "__main__":
    main()
