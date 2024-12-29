import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import base64
import os
import streamlit as st



if __name__ == "__main__":
    st.title("Trashformers Streamlit App")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:  # Ouvre le fichier en mode binaire.
        data = f.read()  # Lit les données du fichier.
    return base64.b64encode(data).decode()  # Encode les données en base64 et les retourne sous forme de chaîne.


# Fonction pour configurer l'image de fond de l'application.
def set_bg(png_file):
    try:
        bin_str = get_base64(png_file)  # Encode le fichier d'image en base64.
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: scroll;
        }
        </style>
        ''' % bin_str  # Définit le style CSS pour utiliser l'image comme fond.
        st.markdown(page_bg_img, unsafe_allow_html=True)  # Applique le style à la page Streamlit.
    except FileNotFoundError:
        st.error(f"El archivo de fondo {png_file} no se encontró.")  # Affiche une erreur si le fichier n'est pas trouvé.



# Configure l'image de fond de l'application.
set_bg('images/trashformer.jpg')  # Utilise l'image "trashformer.jpg" comme fond.


# Lit et encode une image pour l'utiliser dans les styles CSS.
with open("images/trashformer.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()  # Encode l'image en base64.



#Définir les styles pour le fond, les titres et le GIF
background_style = f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap'); /* Importar Montserrat */

    /* Fondo de la aplicación */
    .stApp {{
        background: url(data:image/png;base64,{encoded_string}) no-repeat center center;
        background-size: cover;

    }}

    /* Configuración del titre a droite */
    h1 {{
        font-size: 4.7em; /* Tamaño de la fuente */
        color: orange; /* Color de la letra */
        text-shadow: -4px -4px 0 blue, 4px -4px 0 yellow, -4px 4px 0 blue, 4px 3px 0 yellow; /* Grosor del borde */
        font-family: 'Montserrat', sans-serif;
        text-align: right; /* Alinea el texto a la derecha */
        position: fixed; /* Fijar posición */
        top: 20px; /* Espaciado desde la parte superior */
        right: 20px; /* Espaciado desde la derecha */
        z-index: 1000; /* Asegurar que el título esté sobre otros elementos */
        background: rgba(0, 0, 0, 0); /* Fondo con opacidad para resaltar */
        padding: 60px; /* Espaciado interno */
    }}

    /* Fondo del sidebar */
    section[data-testid="stSidebar"] {{
        background: rgba(0, 0, 0, 0.7); /* Fondo negro con opacidad */
        color: yellow; /* Color del texto */
        font-family: Arial, sans-serif;
    }}

    </style>
"""

# Applique les styles CSS à l'application.
st.markdown(background_style, unsafe_allow_html=True)





st.markdown(
    """
    <style>
    /* Ajustar la altura y el espaciado del contenido de la barra lateral */
    section[data-testid="stSidebar"] {
        padding: 0px; /* Espaciado interno de la barra lateral */
        height: 100vh; /* Altura completa de la barra lateral */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Distribuir espacio entre los elementos */
    }

    .sidebar-title {
        margin-bottom: 1    0px; /* Espaciado debajo del título */
    }

    .icon-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end; /* Alinear al borde inferior */
        align-items: center;
        height: 50px; /* Ajustar la altura de cada contenedor */
        margin-bottom: 0px; /* Espacio entre filas */
    }

    .icon-container img {
        margin-bottom: 0px; /* Espacio entre imagen y texto */
    }
    </style>
    """,
    unsafe_allow_html=True,
)





# Styles pour la SideBar
st.markdown(
    """
    <style>
    .icon-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        align-items: center;
        height: 145px;
        margin-bottom: 20px;
    }
    .icon-container img {
        width: 60px;
        height: auto;
        margin-bottom: 0px;
        border: 5px solid yellow;
    }
    .sidebar-gif {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 0px;
        padding: 0px;
    }
    .sidebar-gif img {
        width: 150px;
        height: auto;
        border: 3px solid yellow;
        border-radius: 5px;
    }
    .sidebar-gif p {
        text-align: center;
        font-size: 20px;
        color: yellow;
        margin-bottom: 0px; /* Espacio entre el texto y el GIF */
    }
    </style>
    """,
    unsafe_allow_html=True,
)





def plot_bboxes(image, results, labels=None, colors=None, conf=0.2, score=True):
    """
    Annoter une image avec des bounding boxes depuis les résultats YOLO.
    """
    image_copy = image.copy()

    # Vérifier si les résultats ont des boîtes de délimitation
    if len(results) > 0 and hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            # Obtenir la boîte de délimitation
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Obtenir la classe et le score de confiance
            cls = int(box.cls[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())

            # Filtrer par score de confiance
            if conf_val < conf:
                continue

            # Sélectionner la couleur et le label
            color = colors[cls % len(colors)] if colors else (0, 255, 0)
            label = labels.get(cls, f"Classe {cls}") if labels else f"Classe {cls}"

            # Ajouter le score si demandé
            if score:
                label += f" ({conf_val:.2f})"

            # Dessiner le rectangle
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

            # Ajouter le texte
            cv2.putText(image_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_copy

def process_webcam_yolo(model_path):
    """
    Capture et annotation des frames de la webcam en temps réel avec YOLO.
    """
    # Charger le modèle YOLO
    model = YOLO(model_path)

    st.title("")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Impossible d'accéder à la webcam.")
        return

    frame_placeholder = st.empty()
    stop_button = st.button("Arrêter la webcam")
    frame_count = 0  # Compteur pour le frame skipping

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Traiter une frame sur 3 pour réduire le lag
        if frame_count % 3 != 0:
            continue

        # Redimensionner la frame pour améliorer les performances
        frame = cv2.resize(frame, (640, 480))

        # Faire la prédiction avec YOLO
        try:
            start_time = time.time()
            results = model.predict(frame, conf=0.5)
            end_time = time.time()

            # Annoter la frame avec les résultats
            annotated_frame = plot_bboxes(
                frame,
                results,
                labels=results[0].names,
                conf=0.3,
                score=True
            )

            # Convertir de BGR à RGB pour Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Afficher la frame annotée
            frame_placeholder.image(annotated_frame_rgb, caption="Webcam Annotée", use_container_width=True)



        except Exception as e:
            st.error(f"Erreur lors du traitement : {str(e)}")

        if stop_button:
            break

    cap.release()
    st.write("Capture webcam terminée.")

def main():
    st.sidebar.title("")
    model_path = st.sidebar.text_input("Chemin du modèle YOLO", value="best.pt")

    if st.sidebar.button("Démarrer la webcam"):
        # Vérifier si le modèle existe
        if torch.cuda.is_available():
            st.sidebar.success("GPU disponible")

        if model_path and torch.os.path.exists(model_path):
            process_webcam_yolo(model_path)
        else:
            st.error("Veuillez spécifier un chemin de modèle valide.")

if __name__ == "__main__":
    main()
