# DFY Voice

Self-hosted Text-to-Speech API powered by [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) (ResembleAI).

## Endpoints

### `GET /health`

Returns API status and model availability.

```bash
curl https://tts.99dfy.com/health
```

```json
{
  "status": "ok",
  "models": { "turbo": true, "multilingual": true },
  "device": "cuda"
}
```

### `POST /tts`

Generate speech from text. Returns MP3 audio.

| Field      | Type   | Required | Default | Description                              |
|------------|--------|----------|---------|------------------------------------------|
| `text`     | string | yes      |         | Text to synthesise                       |
| `language` | string | no       | `en`    | Language code (en, fr, de, es, zh, etc.) |

```bash
curl -X POST https://tts.99dfy.com/tts \
  -F "text=Hello, welcome to DFY Voice!" \
  -F "language=en" \
  --output speech.mp3
```

Supported languages: `en`, `ar`, `cs`, `de`, `es`, `fr`, `hi`, `hu`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `ro`, `ru`, `sv`, `tr`, `zh`.

### `POST /tts-clone`

Generate speech in a cloned voice. Upload a ~10 second WAV reference sample. Returns MP3 audio.

| Field   | Type   | Required | Description                                  |
|---------|--------|----------|----------------------------------------------|
| `text`  | string | yes      | Text to synthesise                           |
| `voice` | file   | yes      | Reference WAV file (~10s) for voice cloning  |

```bash
curl -X POST https://tts.99dfy.com/tts-clone \
  -F "text=This is my cloned voice speaking." \
  -F "voice=@sample.wav" \
  --output cloned.mp3
```

## Interactive Docs

Swagger UI is available at `https://tts.99dfy.com/docs`.

## Deployment

The app runs on Docker with GPU support and auto-deploys on push to `main` via GitHub Actions.

### GitHub Secrets Required

| Secret             | Value                     |
|--------------------|---------------------------|
| `CONTABO_HOST`     | Server IP (109.205.182.135) |
| `CONTABO_USER`     | SSH username              |
| `CONTABO_SSH_KEY`  | SSH private key           |

### Manual Deploy

```bash
ssh user@109.205.182.135
cd /opt/dfy-voice
git pull origin main
docker compose down
docker compose up -d --build
```

### First-Time Server Setup

```bash
# Clone repo on the server
cd /opt
git clone <repo-url> dfy-voice
cd dfy-voice
docker compose up -d --build
```

Nginx reverse proxy (`tts.99dfy.com` -> `localhost:8000`) is managed by HestiaCP.
