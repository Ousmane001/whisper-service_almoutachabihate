# whisper-service

Microservice de transcription pour le suivi vocal **"Tarteel"** d'Al-Moutachabihate.

Remplace l'ancienne inférence locale `@xenova/transformers` (modèle générique
`Xenova/whisper-tiny`, format ONNX) par **`tarteel-ai/whisper-base-ar-quran`**,
le modèle officiellement fine-tuné par Tarteel AI sur de la récitation
coranique (WER ≈ 5,75 %), utilisé **tel que publié** (PyTorch natif via
`transformers`), sans conversion de format.

## Pourquoi un microservice séparé ?

- **Aucune conversion ONNX nécessaire** — le modèle Tarteel n'existe qu'au
  format PyTorch ; `@xenova/transformers` (Node/ONNX) l'aurait obligé à une
  conversion fragile. Python + `transformers` le charge nativement.
- **Découplage** — l'inférence ML lourde ne tourne plus dans le process Node
  de l'API (qui reste léger et réactif pour le catalogue/recherche/etc.).
  Si ce service est indisponible, le reste de l'API continue de fonctionner.
- **Écosystème ML mature** — accélération GPU native (CUDA/MPS), meilleure
  maîtrise de la mémoire et de la latence qu'en JS/ONNX.
- **100 % local et auto-hébergé** — aucun appel à OpenAI, à Tarteel.ai (le
  service commercial) ni à un cloud tiers. Le modèle est téléchargé une fois
  puis tourne entièrement sur votre machine.

## Contrat HTTP

Respecte exactement le contrat "moteur distant" déjà prévu côté API
(`runRemoteTranscription` dans `api/src/services/tarteel.ts` —
`TARTEEL_WHISPER_ENDPOINT`) :

```
POST /transcribe
{
  "audioBase64": "...",     // audio encodé en base64 (webm/opus, wav, mp3…)
  "mimeType": "audio/webm",
  "language": "ar",
  "prompt": "..."           // optionnel : contexte (versets de la page affichée)
}
→ { "text": "...", "confidence": 0.85 }
```

`GET /health` renvoie l'état du modèle (`ok`/`loading`) et le device utilisé
(`cuda:0` / `mps:0` / `cpu`).

## Démarrage en local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_model.py        # télécharge tarteel-ai/whisper-base-ar-quran (~290 Mo, une fois)
uvicorn app:app --host 0.0.0.0 --port 9000
```

Puis, côté `api/.env` :
```
TARTEEL_WHISPER_ENDPOINT=http://localhost:9000/transcribe
```

## Performances observées (Mac 8 Go, GPU Apple MPS)

- Chargement + warm-up : ~1-2 s au démarrage
- Transcription d'un segment de ~2 s : **~0,5 s** une fois le modèle préchauffé
- → largement temps réel pour un suivi vocal fluide, et de la marge pour
  plusieurs utilisateurs simultanés (avec une file d'attente côté API si besoin)

## Optimisation contextuelle ("prompt")

Le endpoint accepte un champ `prompt` optionnel : l'API peut y transmettre le
texte des versets de la page actuellement affichée. Whisper utilise ce contexte
pour orienter son décodage vers le vocabulaire attendu — ça améliore la
précision sans coût de performance significatif (architecture Whisper standard,
voir `pipe.tokenizer.get_prompt_ids` dans `app.py`).

## Déploiement

`Dockerfile` fourni : build une image autonome (modèle inclus au build, aucun
téléchargement au runtime). Voir `docker-compose.yml` à la racine du workspace
pour démarrer l'API et ce service ensemble.
# whisper-service_almoutachabihate
