# Telegram Voice Bot

Telegram voice assistant with voice cloning.

Whisper ASR → Groq/Llama brain → XTTS v2 TTS

## Quick Start

```bash
cp .env.example .env
# edit .env with your tokens
docker compose up
```

## Voice

The bot uses `default_speaker.pt` (Annmarie Nele, built-in XTTS voice).

To use your own voice, place a `.speaker.pt` file in the project root. The bot will automatically prefer it over the default.

### How to create `.speaker.pt`

Record ~10 seconds of clean speech (no background noise). Then:

```python
import os, torch
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf, numpy as np

config = XttsConfig()
config.load_json("path/to/xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="path/to/xtts_v2/")
model.eval()

gpt_cond, speaker_emb = model.get_conditioning_latents(
    audio_path="your_recording.wav",
    max_ref_length=10, gpt_cond_len=10,
    gpt_cond_chunk_len=4, librosa_trim_db=25,
    sound_norm_refs=True,
)
y, _ = sf.read("your_recording.wav")

torch.save({
    "gpt_cond_latent": gpt_cond,
    "speaker_embedding": speaker_emb,
    "ref_rms": float(np.sqrt(np.mean(y**2))),
}, ".speaker.pt")
```

## Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `TELEGRAM_BOT_TOKEN` | yes | — |
| `GROQ_API_KEY` | yes | — |
| `SYSTEM_PROMPT` | no | Generic Italian assistant |
| `SPEAKER_FILE` | no | auto (`.speaker.pt` → `default_speaker.pt`) |
| `FFMPEG_BIN` | no | `ffmpeg` |
| `WHISPER_MODEL` | no | `small` |
| `LANGUAGE` | no | `it` |
| `GROQ_MODEL` | no | `groq/llama-3.3-70b-versatile` |

## Files

| File | Committed | Purpose |
|------|-----------|---------|
| `bot.py` | yes | The bot |
| `default_speaker.pt` | yes | Default voice (Annmarie Nele) |
| `.speaker.pt` | no | Your custom voice (gitignored) |
| `.env` | no | Credentials (gitignored) |
| `.env.example` | yes | Credentials template |
| `docker-compose.yml` | yes | Build + run |
