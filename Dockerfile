FROM python:3.11-slim

WORKDIR /app

# libsndfile : nécessaire pour le décodage WAV/PCM rapide via `soundfile`.
# (Le décodage des formats compressés du navigateur — webm/opus, ogg, mp4… —
# passe par PyAV, dont les wheels embarquent déjà leurs propres bibliothèques
# ffmpeg : pas besoin d'installer `ffmpeg` au niveau système, ni de spawn de
# sous-processus à chaque requête — voir `_decode_with_av` dans app.py.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
# Télécharge le modèle pendant le build : l'image est ensuite 100% autonome,
# aucun appel réseau au runtime (cohérent avec le principe d'indépendance de l'API).
RUN python download_model.py

COPY app.py .

ENV WHISPER_MODEL_PATH=/app/models/whisper-base-ar-quran

# Filet de securite : meme valeur plafonnee que torch.set_num_threads()
# (app.py) — evite qu'OpenMP/MKL n'utilisent plus de coeurs que prevu,
# pour laisser de la place aux autres containers du serveur.
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

EXPOSE 9000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
