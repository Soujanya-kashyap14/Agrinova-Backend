"""
Text-to-Speech Service
"""

import os
import uuid
from gtts import gTTS

OUTPUT_DIR = "generated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def text_to_speech(text: str, language: str = "en"):

    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        tts = gTTS(text=text, lang=language)
    except Exception:
        # fallback if language code isn't supported by gTTS
        tts = gTTS(text=text, lang="en")

    tts.save(filepath)

    return {
        "audio_url": f"/generated_audio/{filename}"
    }