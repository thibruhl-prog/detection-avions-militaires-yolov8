import os
import json
from ultralytics import YOLO

# --- CONFIGURATION GOLD ---
CHEMIN_MODELE = r"C:\Users\thibr\Desktop\IA\runs\detect\train\weights\best.pt" 
DOSSIER_EVAL = r"C:\Users\thibr\Desktop\IA\eval-dataset\images"
FICHIER_SORTIE_JSON = "predictions_thibaut_yolo_gold.json"

# On resserre la vis juste ce qu'il faut (de 0.25 à 0.35)
SEUIL_CONFIANCE = 0.35 

print("Chargement du modèle YOLO (Configuration Gold)...")
modele = YOLO(CHEMIN_MODELE)

resultats_json = {}
images_a_traiter = [f for f in os.listdir(DOSSIER_EVAL) if f.endswith(('.jpg', '.png'))]
total = len(images_a_traiter)

print(f"Début de l'analyse ciblée des {total} images...")

for i, nom_fichier in enumerate(images_a_traiter):
    chemin_image = os.path.join(DOSSIER_EVAL, nom_fichier)
    
    # Inférence classique, sans augmentation, juste avec le nouveau seuil
    resultats = modele.predict(source=chemin_image, conf=SEUIL_CONFIANCE, verbose=False)
    
    liste_detections = []
    resultat = resultats[0]
    
    for boite in resultat.boxes:
        x1, y1, x2, y2 = boite.xyxy[0].tolist()
        id_classe = int(boite.cls[0].item())
        nom_classe = modele.names[id_classe] 
        
        liste_detections.append({
            "class": nom_classe,
            "coordinates": {
                "xmin": int(x1),
                "ymin": int(y1),
                "xmax": int(x2),
                "ymax": int(y2)
            }
        })
        
    resultats_json[nom_fichier] = liste_detections
    
    if (i + 1) % 100 == 0 or (i + 1) == total:
        print(f"Progression : {i + 1}/{total} images traitées...")

with open(FICHIER_SORTIE_JSON, 'w') as f:
    json.dump(resultats_json, f, indent=4)

print(f"\nTerminé ! Le fichier '{FICHIER_SORTIE_JSON}' a été généré.")