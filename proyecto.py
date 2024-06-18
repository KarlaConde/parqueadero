import cv2
import numpy as np

# Cargar la imagen
image_path = "C:/Users/conde/Documents/Documentos/UIDE/InteligenciaA/visionc/proyectosencillo/fotos.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("La imagen no se pudo cargar. Verifica la ruta del archivo.")

# Definir las coordenadas de los espacios de estacionamiento ajustadas para la nueva imagen
parking_spaces = [
    ((100, 120), (175, 220), 1),
    ((180, 120), (245, 220), 2),
    ((250, 120), (310, 220), 3),
    ((315, 120), (375, 220), 4),
    ((425, 120), (485, 220), 5),
    ((490, 120), (550, 220), 6),
    ((555, 120), (615, 220), 7),
    ((620, 120), (680, 220), 8),
]

def is_occupied(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    non_zero = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    return non_zero / total_pixels > 0.2  # Ajustar el umbral según sea necesario

empty_spaces = 0

for space in parking_spaces:
    (x1, y1), (x2, y2), space_num = space
    if is_occupied(image, x1, y1, x2, y2):
        color = (0, 0, 255)  # Rojo para espacios ocupados
    else:
        color = (0, 255, 0)  # Verde para espacios vacíos
        empty_spaces += 1
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f'{space_num}', (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.putText(image, f'Espacios vacíos: {empty_spaces}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imshow('Parking Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
