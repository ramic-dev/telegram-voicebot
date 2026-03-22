#!/usr/bin/env python3
"""
bot.py — Telegram voice assistant with cloned voice

Whisper ASR -> Groq/Llama brain (via aider) -> XTTS v2 TTS

Usage:
    docker compose up
    # or: python bot.py (with env vars set)

Env vars:
    TELEGRAM_BOT_TOKEN  — Bot token from BotFather (required)
    GROQ_API_KEY        — Groq API key (required)
    SPEAKER_FILE        — Path to speaker.pt (default: ./speaker.pt)
    FFMPEG_BIN          — Path to ffmpeg (default: ffmpeg)
    WHISPER_MODEL       — Whisper model size (default: small)
    LANGUAGE            — Language code (default: it)
    GROQ_MODEL          — Groq model (default: groq/llama-3.3-70b-versatile)
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

# ── Environment setup (must be before any TTS/torch import) ───────────
os.environ["COQUI_TOS_AGREED"] = "1"

BOT_DIR = Path(__file__).resolve().parent
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_BIN)

_host_speaker = Path("/host/.speaker.pt")
_local_speaker = BOT_DIR / ".speaker.pt"
_default_speaker = BOT_DIR / "default_speaker.pt"
_auto_speaker = _host_speaker if _host_speaker.exists() else (_local_speaker if _local_speaker.exists() else _default_speaker)
SPEAKER_FILE = os.environ.get("SPEAKER_FILE", "") or str(_auto_speaker)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
LANGUAGE = os.environ.get("LANGUAGE", "it")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq/llama-3.3-70b-versatile")

# Persona from env
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT",
    "Rispondi sempre in italiano. Sei un'assistente vocale. "
    "Rispondi in modo naturale e conversazionale. "
    "Sii concisa: risposte brevi e dirette, massimo 2-3 frasi per i messaggi semplici."
)


# ── Validate ──────────────────────────────────────────────────────────
if not os.path.isfile(SPEAKER_FILE):
    raise FileNotFoundError(f"Speaker embedding not found: {SPEAKER_FILE}")
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable not set.")

# ── torchaudio monkey-patch (Windows) ─────────────────────────────────
import soundfile as sf
import torch

def _sf_load(filepath, **kwargs):
    y, sr = sf.read(str(filepath), dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    return torch.FloatTensor(y).unsqueeze(0), sr

import torchaudio
torchaudio.load = _sf_load
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Load models (once at startup) ─────────────────────────────────────
log.info("Loading Whisper model '%s'...", WHISPER_MODEL_SIZE)
import whisper
WHISPER = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu")
log.info("Whisper loaded.")

log.info("Loading XTTS v2 (low-level for quality tuning)...")
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np

# Auto-download model if not cached, then load
from TTS.utils.manage import ModelManager
_mm = ModelManager()
XTTS_MODEL_DIR, _config_path, _ = _mm.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_MODEL_DIR = str(XTTS_MODEL_DIR)

_xtts_config = XttsConfig()
_xtts_config.load_json(str(_config_path))
XTTS_MODEL = Xtts.init_from_config(_xtts_config)
XTTS_MODEL.load_checkpoint(_xtts_config, checkpoint_dir=XTTS_MODEL_DIR)
XTTS_MODEL.eval()

# Load pre-computed speaker conditioning from .pt file (no WAV needed)
log.info("Loading speaker embedding from %s...", SPEAKER_FILE)
_speaker_data = torch.load(SPEAKER_FILE, weights_only=True)
GPT_COND_LATENT = _speaker_data["gpt_cond_latent"]
SPEAKER_EMBEDDING = _speaker_data["speaker_embedding"]
REF_RMS = float(_speaker_data["ref_rms"])
log.info("XTTS v2 loaded. Speaker: %s", os.path.basename(SPEAKER_FILE))

# ── Telegram imports ──────────────────────────────────────────────────
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


# ── Aider/Groq brain ─────────────────────────────────────────────────

async def ask_groq(message: str) -> str:
    """Send a message to Groq via aider CLI in an isolated temp directory."""
    import shutil

    # Create isolated temp workdir — aider runs here, can't touch real files
    workdir = tempfile.mkdtemp(prefix="tgvoicebot-")
    prompt_path = os.path.join(workdir, "prompt.txt")

    try:
        # Write aider config with system prompt + user message
        conf_path = os.path.join(workdir, ".aider.conf.yml")
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write(f"chat-language: it\n")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(f"[ISTRUZIONI DI SISTEMA - SEGUI SEMPRE]: {SYSTEM_PROMPT}\n\n[MESSAGGIO UTENTE]: {message}")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [
                    "aider",
                    "--no-git",
                    "--no-auto-commits",
                    "--no-stream",
                    "--no-pretty",
                    "--yes-always",
                    "--analytics-disable",
                    "--no-show-release-notes",
                    "--no-check-update",
                    "--exit",
                    "--no-auto-lint",
                    "--no-auto-test",
                    "--edit-format", "ask",
                    "--chat-language", "it",
                    "--model", GROQ_MODEL,
                    "--message-file", prompt_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=workdir,  # isolated: aider can only write here
                env={**os.environ, "GROQ_API_KEY": os.environ["GROQ_API_KEY"]},
            ),
        )

        if result.returncode != 0 and not result.stdout.strip():
            log.error("Aider failed (exit %d): %s", result.returncode, result.stderr[:200])
            return None

        # Parse aider output: filter all non-content lines
        NOISE = [
            "Aider v", "Model:", "Git repo:", "Repo-map:",
            "Warning:", "Can't initialize", "can't open",
            "prompt toolkit", "winpty", "cmd.exe",
            "Cygwin", "Windows console", "Reading prompt",
            "diff edit format", "whole edit format", "ask edit format",
            "architect edit format", "Analytics", "analytics",
            "Release notes", "Release history", "release-notes",
            "HISTORY.html", "aider.chat", "https://aider.chat",
            "Tokens:", "Cost:", "> ",
        ]
        lines = result.stdout.strip().split("\n")
        response_lines = []
        for line in lines:
            if any(x in line for x in NOISE):
                continue
            if line.strip() == "" and not response_lines:
                continue  # skip leading blank lines
            response_lines.append(line)

        raw = "\n".join(response_lines).strip()
        if not raw:
            log.error("Aider returned empty response. stderr: %s", result.stderr[:200])
            return None

        return _clean_response(raw)

    except subprocess.TimeoutExpired:
        log.error("Aider timed out after 120s")
        return None
    except Exception as e:
        log.error("Aider error: %s", e)
        return None
    finally:
        # Remove entire temp workdir — all aider residuals gone
        shutil.rmtree(workdir, ignore_errors=True)


def _clean_response(raw: str) -> str:
    """Remove tool_call blocks, JSON blobs, markdown artifacts."""
    text = raw
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json|python|bash|text)?\s*\n.*?\n```", "", text, flags=re.DOTALL)
    text = re.sub(r"\{[^{}]*\"tool\"[^{}]*\}", "", text, flags=re.DOTALL)
    text = re.sub(r"<[a-z_]+>.*?</[a-z_]+>", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r":[a-z_]+:", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text if text else raw.strip()


def _summarize(text: str, max_chars: int = 250) -> str:
    """Extract a short summary for TTS."""
    if len(text) <= max_chars:
        return text
    for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
        idx = text.rfind(sep, 0, max_chars + 20)
        if idx > 80:
            return text[:idx + 1].strip()
    idx = text.rfind(" ", 0, max_chars)
    return text[:idx].strip() + "..." if idx > 80 else text[:max_chars].strip() + "..."


def _split_into_chunks(text: str, max_chars: int = 180) -> list:
    """Split text into sentence-based chunks under max_chars."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(current)
            if len(sent) > max_chars:
                parts = re.split(r'(?<=[,;:])\s+', sent)
                sub = ""
                for p in parts:
                    if len(sub) + len(p) + 1 <= max_chars:
                        sub = (sub + " " + p).strip() if sub else p
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = p
                current = sub if sub else ""
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


async def _send_response(message, response: str):
    """Send text + voice. Strategy by length:
    - Short (<= 180): one voice note
    - Medium (181-450): 2-3 sequential voice notes, first sent ASAP
    - Long (> 450): summary voice + full text expandable
    """
    text_len = len(response)

    if text_len <= 180:
        await message.reply_text(response)
        await synthesize_and_send_msg(message, response)

    elif text_len <= 450:
        await message.reply_text(response)
        chunks = _split_into_chunks(response, max_chars=180)
        for chunk in chunks[:3]:
            await synthesize_and_send_msg(message, chunk)

    else:
        summary = _summarize(response)
        await message.reply_text(summary)
        await synthesize_and_send_msg(message, summary)
        try:
            await message.reply_text(
                f"<blockquote expandable>{response}</blockquote>",
                parse_mode="HTML",
            )
        except Exception:
            await message.reply_text(response)


# ── Handlers ──────────────────────────────────────────────────────────

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ciao! Mandami un vocale e ti rispondo con la mia voce, "
        "oppure scrivimi un messaggio di testo."
    )




async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    log.info("Text received: %s", text[:100])
    response = await ask_groq(text)
    if not response:
        context.user_data["pending_retry"] = text
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Riprova", callback_data="retry")]
        ])
        await update.message.reply_text(
            "Non ho ricevuto risposta in tempo.",
            reply_markup=keyboard,
        )
        return
    await _send_response(update.message, response)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tmp_in_ogg = None
    tmp_in_wav = None

    try:
        tmp_in_ogg = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
        tmp_in_ogg.close()
        tmp_in_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_in_wav.close()

        voice_file = await update.message.voice.get_file()
        await voice_file.download_to_drive(tmp_in_ogg.name)
        log.info("Voice received, %d bytes", os.path.getsize(tmp_in_ogg.name))

        result = subprocess.run(
            [FFMPEG_BIN, "-y", "-i", tmp_in_ogg.name, "-ar", "22050", "-ac", "1", tmp_in_wav.name],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            await update.message.reply_text("Errore nella conversione audio.")
            return

        log.info("Transcribing...")
        transcript = WHISPER.transcribe(tmp_in_wav.name, language=LANGUAGE)["text"].strip()
        log.info("Transcript: %s", transcript[:100])

        if not transcript:
            await update.message.reply_text("Non ho capito, riprova.")
            return

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Procedi", callback_data="proceed")]
        ])
        context.user_data["pending_transcript"] = transcript
        await update.message.reply_text(
            f'Ho capito: "{transcript}"\n\nProcedo?',
            reply_markup=keyboard,
        )

    except Exception as e:
        log.error("Unexpected error in handle_voice: %s", e, exc_info=True)
        try:
            await update.message.reply_text("Errore interno, riprova.")
        except Exception:
            pass

    finally:
        for tmp in [tmp_in_ogg, tmp_in_wav]:
            if tmp is not None:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass


async def handle_proceed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    transcript = context.user_data.pop("pending_transcript", None)
    if not transcript:
        await query.edit_message_text("Sessione scaduta, manda un nuovo vocale.")
        return

    await query.edit_message_text(f'Ho capito: "{transcript}"\n\nElaborazione in corso...')

    log.info("Asking Groq (confirmed): %s", transcript[:100])
    response = await ask_groq(transcript)

    if not response:
        context.user_data["pending_retry"] = transcript
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Riprova", callback_data="retry")]
        ])
        await query.message.reply_text(
            "Non ho ricevuto risposta in tempo.",
            reply_markup=keyboard,
        )
        return

    log.info("Groq response: %s", response[:100])
    await _send_response(query.message, response)


async def handle_retry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    message_text = context.user_data.pop("pending_retry", None)
    if not message_text:
        await query.edit_message_text("Sessione scaduta, manda un nuovo messaggio.")
        return

    await query.edit_message_text("Riprovo...")

    log.info("Retrying Groq: %s", message_text[:100])
    response = await ask_groq(message_text)

    if not response:
        context.user_data["pending_retry"] = message_text
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Riprova", callback_data="retry")]
        ])
        await query.message.reply_text(
            "Non ho ricevuto risposta in tempo.",
            reply_markup=keyboard,
        )
        return

    log.info("Groq response (retry): %s", response[:100])
    await _send_response(query.message, response)


async def synthesize_and_send_msg(message, text: str):
    """Generate TTS audio from text and send as voice message."""
    tmp_wav = None
    tmp_ogg = None
    try:
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        tmp_ogg = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
        tmp_ogg.close()

        # Use low-level inference with optimized parameters
        out = XTTS_MODEL.inference(
            text=text,
            language=LANGUAGE,
            gpt_cond_latent=GPT_COND_LATENT,
            speaker_embedding=SPEAKER_EMBEDDING,
            temperature=0.88,
            top_p=0.90,
            top_k=50,
            repetition_penalty=2.0,
            speed=1.0,
            enable_text_splitting=False,
        )
        wav = out["wav"]
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        # Loudness normalize to match original reference
        wav_rms = float(np.sqrt(np.mean(wav**2)))
        if wav_rms > 0:
            wav = wav * (REF_RMS / wav_rms)
        sf.write(tmp_wav.name, wav, 24000)

        result = subprocess.run(
            [FFMPEG_BIN, "-y", "-i", tmp_wav.name, "-c:a", "libopus", tmp_ogg.name],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.error("FFmpeg encode failed: %s", result.stderr.decode()[:200])
            return

        with open(tmp_ogg.name, "rb") as f:
            voice_bytes = f.read()
        await message.reply_voice(voice=voice_bytes)
        log.info("Voice reply sent, %d bytes", len(voice_bytes))

    except Exception as e:
        log.error("TTS failed: %s", e)
    finally:
        for tmp in [tmp_wav, tmp_ogg]:
            if tmp is not None:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass


# ── Main ──────────────────────────────────────────────────────────────

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN environment variable not set. "
            "Get a token from @BotFather on Telegram."
        )

    log.info("Brain: aider + Groq (%s)", GROQ_MODEL)

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_proceed, pattern="^proceed$"))
    app.add_handler(CallbackQueryHandler(handle_retry, pattern="^retry$"))

    log.info("Bot starting... (polling)")
    app.run_polling()


if __name__ == "__main__":
    main()
