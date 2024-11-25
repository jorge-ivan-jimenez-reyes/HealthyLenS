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
        "Sistema Integral de Detección y Clasificación\n\n"
        "Incluye detección de objetos (YOLO), emociones y gestos.\n\n"
        "Desarrollado con Python, OpenCV, Mediapipe y YOLOv8."
    )


def create_custom_button(root, text, command):
    """
    Crea un botón personalizado con estilos modernos.
    """
    def on_enter(e):
        btn.config(bg="#45a049")  # Cambiar color al pasar el mouse

    def on_leave(e):
        btn.config(bg="#4CAF50")  # Volver al color original

    btn = tk.Button(
        root,
        text=text,
        font=("Helvetica", 16),
        bg="#4CAF50",
        fg="white",
        activebackground="#3e8e41",
        activeforeground="white",
        command=command,
        relief="flat",
        bd=0,
        highlightthickness=0,
        width=30,
        height=2,
    )
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


def main():
    """
    Interfaz gráfica para el menú principal.
    """
    root = tk.Tk()
    root.title("Sistema Integral de Detección y Clasificación")
    root.geometry("600x700")  # Tamaño más grande

    # Personalización del fondo
    root.config(bg="#F0F0F0")

    # Título
    title_label = tk.Label(
        root,
        text="Sistema Integral de Detección y Clasificación",
        font=("Helvetica", 24, "bold"),
        bg="#F0F0F0",
        fg="#333",
    )
    title_label.pack(pady=30)

    # Crear botones personalizados
    create_custom_button(root, "Clasificación de Objetos Personalizados", run_object_classification).pack(pady=15)
    create_custom_button(root, "Detección de Objetos Generales (COCO)", run_general_object_detection).pack(pady=15)
    create_custom_button(root, "Detección de Emociones", run_emotion_detection).pack(pady=15)
    create_custom_button(root, "Detección de Gestos (Manos)", run_gesture_detection).pack(pady=15)
    create_custom_button(root, "Acerca de", show_about).pack(pady=15)
    create_custom_button(root, "Salir", root.destroy).pack(pady=30)

    # Iniciar la interfaz gráfica
    root.mainloop()


if __name__ == "__main__":
    main()
