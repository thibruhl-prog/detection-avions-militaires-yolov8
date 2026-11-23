import os
import xml.etree.ElementTree as ET
import shutil

# --- CHEMINS D'ORIGINE ---
dossier_images_brutes = r"C:\Users\thibr\Desktop\IA\student-dataset 2\images"
dossier_xml_bruts = r"C:\Users\thibr\Desktop\IA\student-dataset 2\annotations"

# --- NOUVEAUX CHEMINS YOLO ---
dossier_yolo = r"C:\Users\thibr\Desktop\IA\dataset_yolo"
os.makedirs(os.path.join(dossier_yolo, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(dossier_yolo, "labels", "train"), exist_ok=True)

# Définition stricte des 20 classes
CLASSES = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", 
           "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20"]

def convertir_coordonnees(taille_img, boite):
    dw = 1. / taille_img[0]
    dh = 1. / taille_img[1]
    x_centre = (boite[0] + boite[1]) / 2.0 - 1
    y_centre = (boite[2] + boite[3]) / 2.0 - 1
    largeur = boite[1] - boite[0]
    hauteur = boite[3] - boite[2]
    return (x_centre * dw, y_centre * dh, largeur * dw, hauteur * dh)

print("Conversion des données au format YOLO...")
fichiers_xml = [f for f in os.listdir(dossier_xml_bruts) if f.endswith('.xml')]

for xml_file in fichiers_xml:
    arbre = ET.parse(os.path.join(dossier_xml_bruts, xml_file))
    racine = arbre.getroot()
    
    # Récupérer la taille de l'image
    taille = racine.find('size')
    w_img = int(taille.find('width').text)
    h_img = int(taille.find('height').text)
    
    nom_image = xml_file.replace('.xml', '.jpg')
    chemin_img_source = os.path.join(dossier_images_brutes, nom_image)
    
    if not os.path.exists(chemin_img_source):
        continue

    # Copier l'image vers le dossier YOLO
    shutil.copy(chemin_img_source, os.path.join(dossier_yolo, "images", "train", nom_image))
    
    # Créer le fichier texte correspondant
    nom_txt = xml_file.replace('.xml', '.txt')
    chemin_txt_dest = os.path.join(dossier_yolo, "labels", "train", nom_txt)
    
    with open(chemin_txt_dest, 'w') as f_out:
        for objet in racine.findall('object'):
            nom_classe = objet.find('name').text
            if nom_classe not in CLASSES: continue
            
            id_classe = CLASSES.index(nom_classe)
            bndbox = objet.find('bndbox')
            boite = (float(bndbox.find('xmin').text), float(bndbox.find('xmax').text),
                     float(bndbox.find('ymin').text), float(bndbox.find('ymax').text))
            
            yolo_coords = convertir_coordonnees((w_img, h_img), boite)
            f_out.write(f"{id_classe} {' '.join(map(str, yolo_coords))}\n")

# Création du fichier de configuration YAML obligatoire pour YOLO
yaml_content = f"""
path: {dossier_yolo}
train: images/train
val: images/train # On utilise le même pour simplifier

names:
  0: A1
  1: A2
  2: A3
  3: A4
  4: A5
  5: A6
  6: A7
  7: A8
  8: A9
  9: A10
  10: A11
  11: A12
  12: A13
  13: A14
  14: A15
  15: A16
  16: A17
  17: A18
  18: A19
  19: A20
"""
with open(os.path.join(dossier_yolo, "dataset.yaml"), 'w') as f:
    f.write(yaml_content)

print(f"Terminé ! Le dataset YOLO est prêt dans : {dossier_yolo}")
print("Le fichier dataset.yaml a été créé.")
