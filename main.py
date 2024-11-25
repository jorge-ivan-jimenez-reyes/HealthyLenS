import tkinter as tk
from tkinter import messagebox
from PP.PPE.pdetection import load_model, process_frame  # Archivo YOLO corregido
from PP.PPE.detection_haar import run_haar_detection
from PP.PPE.emotion_detection import detect_emotion
from PP.PPE.gesture_detection import detect_and_apply_filters
import cv2  # Asegúrate de importar OpenCV aquí

def run_object_classification():
    """
    Ejecuta la clasificación de objetos con YOLO.
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

        # Procesar frame con detección YOLO
        frame = process_frame(frame, model, 50, 0, 0)  # Sin filtros aplicados
        cv2.imshow("Clasificación de Objetos", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_filter_application():
    """
    Ejecuta la aplicación de filtros y detección con clasificadores Haar.
    """
    run_haar_detection()


def run_emotion_detection():
    """
    Ejecuta la detección de emociones faciales.
    """
    detect_emotion()


def run_gesture_detection():
    """
    Ejecuta la detección de gestos de manos con aplicación de filtros.
    """
    detect_and_apply_filters()


def show_about():
    """
    Muestra información sobre la aplicación.
    """
    messagebox.showinfo(
        "Acerca de",
        "Healthy Lens\n\nSistema de clasificación de objetos, detección de emociones y gestos con filtros.\n\nDesarrollado con Python, OpenCV y Mediapipe."
    )


def main():
    """
    Interfaz gráfica para el menú principal.
    """
    root = tk.Tk()
    root.title("Healthy Lens")
    root.geometry("500x500")

    # Título
    tk.Label(root, text="Healthy Lens", font=("Helvetica", 24, "bold")).pack(pady=20)

    # Botones del menú
    tk.Button(root, text="Clasificación de Objetos (YOLO)", font=("Helvetica", 16), command=run_object_classification).pack(pady=10)
    tk.Button(root, text="Detección y Filtros (Clasificadores Haar)", font=("Helvetica", 16), command=run_filter_application).pack(pady=10)
    tk.Button(root, text="Detección de Emociones", font=("Helvetica", 16), command=run_emotion_detection).pack(pady=10)
    tk.Button(root, text="Detección de Gestos (Manos)", font=("Helvetica", 16), command=run_gesture_detection).pack(pady=10)
    tk.Button(root, text="Acerca de", font=("Helvetica", 16), command=show_about).pack(pady=10)
    tk.Button(root, text="Salir", font=("Helvetica", 16), command=root.destroy).pack(pady=10)

    # Iniciar la interfaz gráfica
    root.mainloop()


if __name__ == "__main__":
    main()
