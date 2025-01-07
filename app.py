import cv2
import torch
from ultralytics import YOLO
import numpy as np
import gradio as gr


# Cargar el modelo YOLO
model = YOLO('best.pt')  # Asegúrate de que 'best.pt' es el camino correcto a tu modelo

def plot_bboxes(image, results, labels=None, colors=None, conf=0.2, score=True):
    image_copy = image.copy()

    if len(results) > 0 and hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())

            if conf_val < conf:
                continue

            color = colors[cls % len(colors)] if colors else (0, 255, 0)
            label = labels.get(cls, f"Clase {cls}") if labels else f"Clase {cls}"

            if score:
                label += f" ({conf_val:.2f})"

            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_copy

def process_frame(frame):
    # Convertir imagen a formato adecuado si es necesario
    if frame is None:
        return None

    # Convertir de RGB (Gradio) a BGR (OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Realizar la predicción con YOLO
    results = model.predict(frame_bgr, conf=0.5)

    # Anotar la imagen
    annotated_frame = plot_bboxes(
        frame_bgr,
        results,
        labels=model.names,
        conf=0.3,
        score=True
    )

    # Convertir BGR a RGB para mostrar correctamente
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    return annotated_frame_rgb

iface = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(source='webcam', tool=None, type='numpy'),
    outputs=gr.Image(type='numpy'),
    live=True,
    title="Trashformers: Clasificación de Residuos en Tiempo Real",
    description="Usa tu webcam para clasificar residuos con YOLO."
)

if __name__ == "__main__":
    iface.launch()
