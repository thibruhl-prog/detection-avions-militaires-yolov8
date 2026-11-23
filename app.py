import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter

# Configuration de la page
st.set_page_config(page_title="Détection d'Avions Militaires", layout="wide")
st.title("Détection et Classification d'Avions Militaires (YOLOv8)")

# Chargement du modèle (mis en cache pour ne pas le recharger à chaque clic)
@st.cache_resource
def charger_modele():
    return YOLO("best.pt")

modele = charger_modele()

# Barre latérale pour le réglage du seuil
st.sidebar.header("Paramètres du modèle")
seuil_confiance = st.sidebar.slider(
    "Seuil de confiance (Gold = 0.35)", 
    min_value=0.05, 
    max_value=1.0, 
    value=0.35, 
    step=0.05
)

# Zone d'upload d'image
fichier_image = st.file_uploader("Importez une image satellite (.jpg, .png)", type=["jpg", "jpeg", "png"])

if fichier_image is not None:
    image = Image.open(fichier_image)
    
    # Prédiction YOLO
    resultats = modele.predict(source=image, conf=seuil_confiance, verbose=False)
    resultat = resultats[0]
    
    # Image annotée (conversion BGR -> RGB pour l'affichage)
    image_annotee = resultat.plot()[..., ::-1]
    
    # Affichage côte à côte
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Détections YOLOv8")
        st.image(image_annotee, use_container_width=True)
        
    # Bilan des détections
    classes_detectees = [modele.names[int(boite.cls[0])] for boite in resultat.boxes]
    st.markdown("---")
    st.subheader(f"Bilan : {len(classes_detectees)} avion(s) détecté(s)")
    
    if classes_detectees:
        compteur = Counter(classes_detectees)
        for classe, nombre in sorted(compteur.items()):
            st.write(f"- **{classe}** : {nombre} appareil(s)")
    else:
        st.info("Aucun avion détecté avec ce seuil de confiance.")