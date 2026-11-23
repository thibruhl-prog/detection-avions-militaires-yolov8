# Détection et Classification d'Avions Militaires (YOLOv8)

Application Python développée dans le cadre du module d'Introduction à l'IA pour localiser et classifier 20 modèles d'avions militaires sur images satellites.

## Architecture
- **Modèle :** YOLOv8n (Transfer Learning, 25 epochs, seuil de confiance optimisé à 0.35).
- **Performances :** Overall F1-Score de **0.6453** (Detection F1 : 0.9117 / Classification F1 : 0.7078).
- **Interface :** Streamlit (`app.py`).
- **Déploiement :** Conteneurisation Docker (`Dockerfile`).

## Lancer l'application

### En local (Python)
```bash
pip install -r requirements.txt
streamlit run app.py
