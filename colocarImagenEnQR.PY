import qrcode
import numpy as np
import cv2
from PIL import Image
import random

# Parámetros generales
rango_min, rango_max = 40, 400
ejeAX_rand = [(40, 160), (250, 400)]
num_generaciones = 10  # Número de iteraciones del algoritmo genético
poblacion_tamano = 15  # Número de imágenes por generación

# Función que verifica si una coordenada está en los rangos prohibidos


def verificarRango(numero, rangos):
    for inicio, fin in rangos:
        if inicio <= numero <= fin:
            return True
    return False

# Función para insertar imagen sobre QR


def insertarImagen(x, y, escala, nombre_archivo):
    qr = cv2.imread("QrPrueb.png")
    img = cv2.imread("LOGO.png")

    h, w = img.shape[:2]
    img_redim = cv2.resize(img, (int(w * escala), int(h * escala)))

    if y + img_redim.shape[0] <= qr.shape[0] and x + img_redim.shape[1] <= qr.shape[1]:
        qr[y:y+img_redim.shape[0], x:x+img_redim.shape[1]] = img_redim
        cv2.imwrite(nombre_archivo, qr)

# Función para evaluar la legibilidad del QR


def evaluar_qr(nombre_archivo):
    img = cv2.imread(nombre_archivo)
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img)
    return len(data)  # Cuanto más texto recupere, mejor es el QR

# Función para generar población inicial


def generar_poblacion():
    poblacion = []
    for i in range(poblacion_tamano):
        while True:
            x = random.randint(rango_min, rango_max)
            y = random.randint(rango_min, rango_max)
            escala = random.uniform(0.2, 0.9)
            if not verificarRango(x, ejeAX_rand) and not verificarRango(y, ejeAX_rand):
                break
        poblacion.append((x, y, escala))
    return poblacion

# Función de selección de los mejores individuos


def seleccion(poblacion, aptitudes):
    seleccionados = sorted(zip(poblacion, aptitudes),
                           key=lambda x: x[1], reverse=True)
    return [individuo for individuo, _ in seleccionados[:5]]  # Los 5 mejores

# Cruce de los mejores individuos


def cruce(padres):
    nueva_poblacion = []
    for _ in range(poblacion_tamano - len(padres)):
        p1, p2 = random.sample(padres, 2)
        hijo = (
            (p1[0] + p2[0]) // 2,
            (p1[1] + p2[1]) // 2,
            (p1[2] + p2[2]) / 2
        )
        nueva_poblacion.append(hijo)
    return padres + nueva_poblacion  # Retornar padres + descendencia

# Mutación de algunos individuos


def mutacion(poblacion):
    for i in range(len(poblacion)):
        if random.random() < 0.2:  # Probabilidad de mutación del 20%
            x, y, escala = poblacion[i]
            x += random.randint(-20, 20)
            y += random.randint(-20, 20)
            escala += random.uniform(-0.1, 0.1)
            poblacion[i] = (max(rango_min, min(x, rango_max)), max(
                rango_min, min(y, rango_max)), max(0.2, min(escala, 0.9)))
    return poblacion

# Algoritmo genético principal


def algoritmo_genetico():
    poblacion = generar_poblacion()

    for generacion in range(num_generaciones):
        aptitudes = []
        for i, (x, y, escala) in enumerate(poblacion):
            nombre_archivo = f"QR_{generacion}_{i}.png"
            insertarImagen(x, y, escala, nombre_archivo)
            aptitudes.append(evaluar_qr(nombre_archivo))

        print(f"Generación {generacion} - Mejor legibilidad: {max(aptitudes)}")

        padres = seleccion(poblacion, aptitudes)
        poblacion = cruce(padres)
        poblacion = mutacion(poblacion)

    # Evaluar la mejor solución final
    mejor_x, mejor_y, mejor_escala = max(
        zip(poblacion, aptitudes), key=lambda x: x[1])[0]
    print(
        f"Mejor configuración encontrada: x={mejor_x}, y={mejor_y}, escala={mejor_escala}")
    insertarImagen(mejor_x, mejor_y, mejor_escala, "Mejor_QR.png")
    print("Imagen final guardada como 'Mejor_QR.png'")


# Ejecutar el algoritmo
if __name__ == "__main__":
    algoritmo_genetico()
