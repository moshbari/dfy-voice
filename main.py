import io
import os
import time
import uuid
import logging
import tempfile
import threading
from contextlib import asynccontextmanager

import torch
import torchaudio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dfy-voice")

# Global model references
tts_model = None
multilingual_model = None

# Voices directory
VOICES_DIR = Path(__file__).parent / "voices"

# Job tracking for async generation
jobs: dict = {}
JOB_TTL = 300  # auto-cleanup after 5 minutes

# Languages that require the multilingual model
MULTILINGUAL_LANGS = {
    "ar", "cs", "de", "es", "fr", "hi", "hu", "it", "ja", "ko",
    "nl", "pl", "pt", "ro", "ru", "sv", "tr", "zh",
}


def load_models():
    """Load TTS models at startup."""
    global tts_model, multilingual_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models on device: {device}")

    start = time.time()
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    tts_model = ChatterboxTurboTTS.from_pretrained(device=device)
    logger.info(f"ChatterboxTurbo loaded in {time.time() - start:.1f}s")

    start = time.time()
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    # Workaround: multilingual model doesn't pass map_location to torch.load,
    # which crashes on CPU-only machines. Patch torch.load temporarily.
    # See: https://github.com/resemble-ai/chatterbox/issues/351
    if device == "cpu":
        _original_torch_load = torch.load
        def _cpu_torch_load(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device("cpu"))
            return _original_torch_load(*args, **kwargs)
        torch.load = _cpu_torch_load

    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    if device == "cpu":
        torch.load = _original_torch_load

    logger.info(f"ChatterboxMultilingual loaded in {time.time() - start:.1f}s")


def cleanup_old_jobs():
    """Remove jobs older than JOB_TTL seconds."""
    now = time.time()
    expired = [jid for jid, j in jobs.items() if now - j["created"] > JOB_TTL]
    for jid in expired:
        jobs.pop(jid, None)


def run_tts_job(job_id: str, text: str, language: str, voice_path: str | None, clone_path: str | None):
    """Run TTS generation in a background thread."""
    job = jobs[job_id]
    try:
        job["status"] = "generating"
        job["started"] = time.time()

        if clone_path:
            wav = tts_model.generate(text, audio_prompt_path=clone_path)
            sr = tts_model.sr
        elif language == "en":
            wav = tts_model.generate(text, audio_prompt_path=voice_path)
            sr = tts_model.sr
        else:
            wav = multilingual_model.generate(text, language_id=language)
            sr = multilingual_model.sr

        job["status"] = "converting"

        mp3_buf = io.BytesIO()
        torchaudio.save(mp3_buf, wav, sr, format="mp3")
        mp3_buf.seek(0)

        job["result"] = mp3_buf.getvalue()
        job["status"] = "done"

    except Exception as e:
        logger.exception("TTS job failed")
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        job["finished"] = time.time()
        # Clean up temp clone file
        if clone_path and os.path.exists(clone_path):
            os.unlink(clone_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    logger.info("=" * 50)
    logger.info("DFY Voice API is ready!")
    logger.info("Docs: http://0.0.0.0:8000/docs")
    logger.info("=" * 50)
    yield


app = FastAPI(
    title="DFY Voice",
    description="Self-hosted Text-to-Speech API powered by Chatterbox TTS",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


def wav_to_mp3(wav_tensor: torch.Tensor, sample_rate: int) -> io.BytesIO:
    """Convert a WAV tensor to MP3 bytes."""
    buf = io.BytesIO()
    torchaudio.save(buf, wav_tensor, sample_rate, format="mp3")
    buf.seek(0)
    return buf


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "turbo": tts_model is not None,
            "multilingual": multilingual_model is not None,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/voices")
async def list_voices():
    """List available pre-loaded voices."""
    voices = []
    if VOICES_DIR.exists():
        for f in sorted(VOICES_DIR.glob("*.wav")):
            voices.append(f.stem)
    return {"voices": voices}


@app.get("/voices/{name}/preview")
async def voice_preview(name: str):
    """Serve the raw WAV file for a voice preview."""
    vfile = VOICES_DIR / f"{name}.wav"
    if not vfile.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return FileResponse(vfile, media_type="audio/wav")


@app.post("/tts")
async def text_to_speech(
    text: str = Form(..., description="Text to synthesise"),
    language: str = Form("en", description="Language code (e.g. en, fr, de, es, zh)"),
    voice: str = Form("", description="Pre-loaded voice name (e.g. Emily, Adrian). Leave empty for default."),
):
    """Start async TTS generation. Returns a job_id to poll for progress."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    voice_path = None
    if voice:
        vfile = VOICES_DIR / f"{voice}.wav"
        if not vfile.exists():
            raise HTTPException(status_code=400, detail=f"Voice '{voice}' not found")
        voice_path = str(vfile)

    if language != "en" and language not in MULTILINGUAL_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {language}. Supported: en, {', '.join(sorted(MULTILINGUAL_LANGS))}",
        )

    cleanup_old_jobs()

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "created": time.time(),
        "started": None,
        "finished": None,
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=run_tts_job,
        args=(job_id, text, language, voice_path, None),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.post("/tts-clone")
async def text_to_speech_clone(
    text: str = Form(..., description="Text to synthesise"),
    voice: UploadFile = File(..., description="Reference WAV file (~10s) for voice cloning"),
):
    """Start async voice-cloned TTS generation. Returns a job_id to poll for progress."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    if not voice.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Voice sample must be a WAV file")

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await voice.read())
        clone_path = tmp.name

    cleanup_old_jobs()

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "created": time.time(),
        "started": None,
        "finished": None,
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=run_tts_job,
        args=(job_id, text, "en", None, clone_path),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/tts/status/{job_id}")
async def tts_status(job_id: str):
    """Poll generation progress."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Estimate progress based on elapsed time
    progress = 0
    if job["status"] == "queued":
        progress = 0
    elif job["status"] == "generating":
        elapsed = time.time() - (job["started"] or job["created"])
        # Asymptotic curve: always climbing, never stuck at a cap.
        # Approaches 90% but keeps moving — feels alive even on slow hardware.
        # At 30s → ~43%, 60s → ~63%, 120s → ~78%, 300s → ~88%, 600s → ~89%
        progress = int(90 * (1 - 1 / (1 + elapsed / 45)))
    elif job["status"] == "converting":
        progress = 95
    elif job["status"] == "done":
        progress = 100
    elif job["status"] == "error":
        progress = 0

    return {
        "status": job["status"],
        "progress": progress,
        "error": job.get("error"),
    }


@app.get("/tts/result/{job_id}")
async def tts_result(job_id: str):
    """Download the generated MP3."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not finished yet")

    mp3_data = job["result"]
    # Clean up job after download
    jobs.pop(job_id, None)

    return StreamingResponse(
        io.BytesIO(mp3_data),
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'attachment; filename="speech.mp3"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
