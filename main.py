import io
import os
import time
import logging
import tempfile
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dfy-voice")

# Global model references
tts_model = None
multilingual_model = None

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


@app.post("/tts")
async def text_to_speech(
    text: str = Form(..., description="Text to synthesise"),
    language: str = Form("en", description="Language code (e.g. en, fr, de, es, zh)"),
):
    """Generate speech from text. Returns MP3 audio."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    try:
        if language == "en":
            wav = tts_model.generate(text)
            sr = tts_model.sr
        elif language in MULTILINGUAL_LANGS:
            wav = multilingual_model.generate(text, language_id=language)
            sr = multilingual_model.sr
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {language}. Supported: en, {', '.join(sorted(MULTILINGUAL_LANGS))}",
            )

        mp3_buf = wav_to_mp3(wav, sr)
        return StreamingResponse(
            mp3_buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'attachment; filename="speech.mp3"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts-clone")
async def text_to_speech_clone(
    text: str = Form(..., description="Text to synthesise"),
    voice: UploadFile = File(..., description="Reference WAV file (~10s) for voice cloning"),
):
    """Generate speech in a cloned voice. Upload a ~10 second WAV sample. Returns MP3 audio."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    if not voice.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Voice sample must be a WAV file")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await voice.read())
            tmp_path = tmp.name

        wav = tts_model.generate(text, audio_prompt_path=tmp_path)
        mp3_buf = wav_to_mp3(wav, tts_model.sr)

        return StreamingResponse(
            mp3_buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'attachment; filename="cloned_speech.mp3"'},
        )
    except Exception as e:
        logger.exception("Voice cloning failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
