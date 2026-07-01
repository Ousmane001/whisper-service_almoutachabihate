"""
whisper-service — microservice de transcription pour le suivi vocal "Tarteel".

Expose un seul endpoint HTTP simple, conforme au contrat "moteur distant" déjà
attendu par l'API Fastify (`runRemoteTranscription` dans api/src/services/tarteel.ts) :

    POST /transcribe
    -> { audioBase64, mimeType, language?, prompt? }
    <- { text, confidence }

Utilise tarteel-ai/whisper-base-ar-quran (fine-tune officiel de Whisper Base
spécialisé sur la récitation coranique, WER ≈ 5,75 %) en local — aucune
conversion ONNX, aucun appel à un service tiers : 100 % auto-hébergé.

Le modèle est chargé UNE SEULE FOIS au démarrage (warm start), et reste en
mémoire pour des inférences rapides à la volée — c'est l'architecture qui rend
le suivi en temps réel "ultra rapide" : pas de rechargement par requête.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("whisper-service")

MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", "models/whisper-base-ar-quran")
SAMPLE_RATE = 16_000

# Sur CPU (pas de CUDA/MPS), PyTorch ne detecte pas toujours correctement le
# nombre de coeurs disponibles dans un conteneur Docker et peut se limiter a 1
# thread — un facteur de ralentissement pour l'inference. On force donc un
# nombre de threads explicite, mais VOLONTAIREMENT plafonne (pas tous les
# coeurs de la machine) pour laisser de la place aux autres containers qui
# tournent sur le meme serveur (api, monitoring, etc.) — voir aussi la limite
# Docker "cpus:" cote docker-compose.yml, qui plafonne en plus au niveau OS.
# Ajustable via WHISPER_CPU_THREADS si besoin, sans toucher au code.
if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    _default_threads = min(os.cpu_count() or 2, 2)
    _cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", _default_threads))
    torch.set_num_threads(_cpu_threads)
    logger.info("CPU detecte : torch configure sur %d threads (plafonne, voir WHISPER_CPU_THREADS)", _cpu_threads)

# Whisper a une fenêtre de décodage limitée (`max_target_positions` = 448 tokens,
# prompt + tokens spéciaux + sortie générée compris). Un prompt trop long déclenche
# une ValueError côté `generate` plutôt qu'une troncature silencieuse — on tronque
# donc nous-mêmes aux N derniers tokens (les plus proches du passage récité, en
# général en fin de page) en gardant toujours le token spécial `<|startofprev|>`
# en tête. 192 tokens laisse une marge confortable pour les tokens spéciaux du
# décodeur et jusqu'à ~250 tokens de sortie générée.
MAX_PROMPT_TOKENS = 192

# État partagé : le pipeline est chargé une fois au démarrage (lifespan) et réutilisé.
_state: dict[str, object] = {}


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps:0"
    return "cpu"


@asynccontextmanager
async def lifespan(_: FastAPI):
    device = _pick_device()
    logger.info("Chargement du modèle %s sur %s…", MODEL_PATH, device)
    t0 = time.time()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_PATH,
        device=device,
    )
    _state["pipe"] = pipe

    # Warm-up : la toute première inférence déclenche la compilation/JIT du graphe
    # (≈ 15s sur MPS/CPU) — on l'exécute ici, au démarrage, sur un buffer silencieux,
    # pour que la première vraie requête utilisateur soit déjà rapide (~0,5s).
    try:
        silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        pipe(silence)
        logger.info("Warm-up effectué — première inférence utilisateur sera rapide")
    except Exception:  # noqa: BLE001
        logger.warning("Échec du warm-up (non bloquant)", exc_info=True)

    logger.info("Modèle chargé et préchauffé en %.1fs — prêt pour l'inférence en temps réel", time.time() - t0)
    yield
    _state.clear()


app = FastAPI(title="whisper-service — Tarteel follow-along", lifespan=lifespan)


class TranscribeRequest(BaseModel):
    audioBase64: str = Field(..., description="Audio encodé en base64 (webm/opus, wav, mp3…)")
    mimeType: str | None = Field(default="audio/webm")
    language: str | None = Field(default="ar")
    prompt: str | None = Field(
        default=None,
        description=(
            "Contexte textuel optionnel (ex. les versets de la page affichée) "
            "utilisé pour guider/biaiser la transcription vers le vocabulaire attendu."
        ),
    )


class TranscribeResponse(BaseModel):
    text: str
    confidence: float


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if "pipe" in _state else "loading",
        "model": MODEL_PATH,
        "device": str(getattr(_state.get("pipe"), "device", "n/a")),
    }


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(payload: TranscribeRequest) -> TranscribeResponse:
    pipe = _state.get("pipe")
    if pipe is None:
        raise HTTPException(status_code=503, detail="Le modèle est encore en cours de chargement")

    try:
        audio_bytes = base64.b64decode(payload.audioBase64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="audioBase64 invalide") from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audioBase64 est vide")

    waveform = _decode_audio(audio_bytes)

    # NB : whisper-base-ar-quran est un fine-tune mono-tâche (transcription arabe
    # uniquement). Sa generation_config publiée est antérieure aux exigences
    # récentes de `transformers` pour l'argument `language=` (voir issue #25084) ;
    # on ne force donc PAS la langue/tâche ici — le modèle transcrit nativement en
    # arabe coranique sans ambiguïté.
    generate_kwargs: dict[str, object] = {}

    # Optimisation contextuelle : si l'app transmet le texte attendu (versets de la
    # page affichée), on l'utilise comme "prompt" pour orienter le décodage vers ce
    # vocabulaire — ça améliore la précision sans ralentir l'inférence.
    if payload.prompt:
        try:
            prompt_ids = pipe.tokenizer.get_prompt_ids(payload.prompt, return_tensors="pt")
            if prompt_ids.shape[-1] > MAX_PROMPT_TOKENS:
                # Garde le token spécial de tête (`<|startofprev|>`) + les derniers
                # tokens du prompt — c'est généralement la fin de la page affichée,
                # donc le vocabulaire le plus pertinent pour la suite de la récitation.
                prompt_ids = torch.cat([prompt_ids[:1], prompt_ids[-(MAX_PROMPT_TOKENS - 1):]])
            generate_kwargs["prompt_ids"] = prompt_ids.to(pipe.model.device)
        except Exception:  # noqa: BLE001
            logger.warning("Impossible d'appliquer le prompt fourni — transcription sans contexte")

    t0 = time.time()
    result = pipe(waveform, generate_kwargs=generate_kwargs)
    elapsed = time.time() - t0

    text = (result.get("text") or "").strip()
    logger.info("Transcription en %.0f ms : %r", elapsed * 1000, text)

    # whisper-base-ar-quran ne renvoie pas de score de confiance natif — on dérive
    # une estimation simple à partir de la longueur du texte détecté (heuristique
    # raisonnable : un texte vide/très court = faible confiance).
    confidence = 0.85 if len(text) >= 4 else 0.4

    return TranscribeResponse(text=text, confidence=confidence)


def _decode_audio(audio_bytes: bytes) -> "np.ndarray":
    """Décode un buffer audio (webm/opus, wav, mp3…) en signal mono 16kHz.

    Le navigateur (`MediaRecorder`) envoie de l'audio compressé — typiquement
    `audio/webm;codecs=opus` — que `libsndfile` (donc `soundfile`) ne sait PAS
    décoder nativement (conteneur WebM non supporté). La première implémentation
    retombait alors sur `librosa`/`audioread`, qui **relance un sous-processus
    ffmpeg à chaque appel** : un coût de démarrage de processus énorme répété
    toutes les ~2 secondes, invisible dans les logs de "temps de transcription"
    (le décodage a lieu avant) mais responsable de réponses à 20-30 secondes en
    usage réel — bien au-delà du temps réel attendu.
    →  On utilise `PyAV` (bindings natifs vers les bibliothèques ffmpeg,
    décodage/ré-échantillonnage en mémoire, AUCUN sous-processus) comme moteur
    principal : rapide et universel (gère webm/opus, ogg, mp4, wav, mp3…).
    `soundfile` reste un essai rapide pour le WAV/PCM brut (chemin le plus
    direct quand le format le permet, ex. clients mobiles).
    """
    try:
        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            data = _resample(data, sr, SAMPLE_RATE)
        return np.asarray(data, dtype=np.float32)
    except Exception:
        return _decode_with_av(audio_bytes)


def _decode_with_av(audio_bytes: bytes) -> "np.ndarray":
    """Décode et ré-échantillonne en une passe via PyAV (libavformat/libavcodec)."""
    import av

    container = av.open(io.BytesIO(audio_bytes))
    try:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())

        # Vide le buffer interne du ré-échantillonneur (dernières frames en attente).
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32)


def _resample(data: "np.ndarray", orig_sr: int, target_sr: int) -> "np.ndarray":
    """Ré-échantillonnage simple par interpolation linéaire (suffisant pour de
    la voix ; évite une dépendance supplémentaire pour ce cas, déjà rare — les
    navigateurs envoient en 16 kHz, voir `getUserMedia({ sampleRate: 16000 })`)."""
    if len(data) == 0 or orig_sr == target_sr:
        return data
    duration = len(data) / orig_sr
    target_length = max(1, round(duration * target_sr))
    original_x = np.linspace(0, duration, num=len(data), endpoint=False)
    target_x = np.linspace(0, duration, num=target_length, endpoint=False)
    return np.interp(target_x, original_x, data)
