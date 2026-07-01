"""
Télécharge et met en cache localement le modèle tarteel-ai/whisper-base-ar-quran
(fine-tune officiel de Whisper Base spécialisé sur la récitation coranique).

Usage : python download_model.py
Le modèle (~290 Mo) est stocké dans ./models/whisper-base-ar-quran et n'est
PAS versionné dans git (voir .gitignore) — chaque environnement le télécharge
une fois, puis whisper-service le charge depuis le disque local (100% offline
ensuite, aucun appel réseau au runtime).
"""

from huggingface_hub import snapshot_download

MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
TARGET_DIR = "models/whisper-base-ar-quran"

if __name__ == "__main__":
    path = snapshot_download(repo_id=MODEL_ID, local_dir=TARGET_DIR)
    print(f"Modèle '{MODEL_ID}' téléchargé dans : {path}")
    print("Vous pouvez maintenant démarrer le service avec : uvicorn app:app --host 0.0.0.0 --port 9000")
