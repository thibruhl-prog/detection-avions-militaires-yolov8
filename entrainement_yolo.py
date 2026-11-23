from ultralytics import YOLO

# 1. Chargement du modèle de base (version "nano" yolov8n.pt, très légère et rapide pour ton CPU)
print("Chargement de YOLOv8...")
modele = YOLO("yolov8n.pt") 

# 2. Lancement de l'entraînement sur tes données militaires
print("Début de l'apprentissage...")
modele.train(
    data=r"C:\Users\thibr\Desktop\IA\dataset_yolo\dataset.yaml", 
    epochs=25,        # 25 itérations suffisent généralement pour que YOLO apprenne
    imgsz=640,        # YOLO va redimensionner intelligemment les grandes images
    batch=16, 
    device="cpu"      # On précise que l'on calcule sur le processeur
)

print("Entraînement terminé ! Le modèle est sauvegardé.")
