def show_menu():
    """
    Muestra un menú para elegir entre clasificación y filtros.
    """
    print("\n--- Healthy Lens ---")
    print("1. Clasificación de Objetos (YOLO)")
    print("2. Filtros y Detección (Clasificadores Haar/HSV)")
    print("q. Salir")
    choice = input("Seleccione una opción: ").strip()
    return choice
