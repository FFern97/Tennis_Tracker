# 🎾 Tennis Tracking System

Sistema modular de visión computacional para el seguimiento de jugadores y pelota en partidos de tenis. Utiliza YOLOv8 para detección y estimación de pose, con un motor de tracking robusto basado en homografía de cancha.

---

## 🚀 Guía de Instalación y Configuración

Seguí estos pasos para dejar el entorno listo para el procesamiento.

### Paso 1: Obtención del código y entorno virtual

Para evitar conflictos con otras librerías de Python, usá un **entorno virtual**.

**Clonar el repositorio**

```bash
git clone <url-del-repo>
cd Tennis
```

**Crear el entorno virtual (VENV)**  
Crea un entorno aislado donde se instalan las dependencias del proyecto.

```bash
python -m venv .venv
```

**Activar el entorno**

En **Windows**:

```bash
.\.venv\Scripts\activate
```

En **macOS/Linux**:

```bash
source .venv/bin/activate
```

---

### Paso 2: Instalación de dependencias

Se instala el stack necesario (PyTorch, OpenCV, Ultralytics/YOLO).

**Instalar vía requirements**

```bash
pip install -r requirements.txt
```

`pip` es el gestor de paquetes de Python; `-r` indica que lea la lista desde el archivo.

---

## 📂 Gestión de archivos (Data & Models)

Los modelos y videos están excluidos de Git por su tamaño. El sistema **crea las carpetas automáticamente** la primera vez que se ejecuta.

### Modelos

Colocá los archivos **.pt** en la carpeta **`models/`**:

| Archivo | Descripción |
|--------|-------------|
| **`best.pt`** | Detección de pelota |
| **`model_tennis_court_det.pt`** | Geometría de cancha |
| **`yolov8n-pose.pt`** | Pose de jugadores (YOLOv8 descarga si falta) |

### Video de entrada

Colocá el video a procesar en **`data/videos/`**.

---

## ⚙️ Ejecución y personalización

### Configurar el script

Editá **`config.py`** y ajustá según tu caso:

- **`VIDEO_IN_PATH`**: Nombre del archivo de video dentro de **`data/videos/`**.
- **`BALL_CONFIDENCE`**: Umbral de detección de pelota (0–1).

### Ejecutar el pipeline

```bash
python main.py
```

El video resultante se guarda en **`output_videos/`**.

---

## 🔧 Detalles técnicos del pipeline

- **Inferencia estructurada**: Los resultados de YOLO se mapean a esquemas (**`schema.py`**) antes de ser procesados.
- **Geometría de cancha**: Homografía para transformar coordenadas píxel → metros de cancha.
- **Tracking robusto**: Suavizado y manejo de oclusiones para no perder jugadores al cruzarse.
