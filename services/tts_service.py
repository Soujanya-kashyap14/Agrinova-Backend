"""
Text-to-Speech Service
"""

import asyncio
import base64
import concurrent.futures
import os
import re
import uuid

import requests
from gtts import gTTS

OUTPUT_DIR = "generated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# PARALLEL CHUNK FETCHING
# ==========================================================
#
# Google's Translate TTS endpoint caps each request at 100
# characters, so gTTS silently splits any longer answer into several
# chunks - a typical multi-sentence farmer answer needs 5-6 of them.
# gTTS's own gTTS.stream() fetches these ONE AT A TIME, each with a
# brand new requests.Session(), so total time scales linearly with
# chunk count: measured 17s for a single 627-character Kannada
# answer (6 chunks), dominating the entire request compared to ~2-4s
# for the Gemini call that produced the text. The chunks are
# completely independent until their audio bytes are concatenated at
# the end, so there's no reason to fetch them one after another -
# fetching them concurrently instead should take roughly as long as
# ONE chunk, not all of them added up.
#
# This relies on gTTS's private `_prepare_requests()` (not part of
# its public API), so it can break on a future gTTS version bump -
# `text_to_speech()` below falls back to gTTS's normal sequential
# `.save()` if anything about this goes wrong.
# ==========================================================

_GTTS_AUDIO_LINE_PATTERN = re.compile(r'jQ1olc","\[\\"(.*)\\"]')


def _fetch_tts_chunk(prepared_request, timeout: int) -> bytes:

    with requests.Session() as session:
        response = session.send(request=prepared_request, timeout=timeout)
        response.raise_for_status()

        for line in response.iter_lines(chunk_size=1024):
            decoded_line = line.decode("utf-8")

            if "jQ1olc" in decoded_line:
                match = _GTTS_AUDIO_LINE_PATTERN.search(decoded_line)

                if match:
                    return base64.b64decode(match.group(1).encode("ascii"))

    raise RuntimeError("No audio data in TTS response chunk.")


def _save_tts_in_parallel(tts: gTTS, filepath: str) -> None:

    prepared_requests = tts._prepare_requests()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(len(prepared_requests), 1)
    ) as executor:

        # map() preserves input order in its results, so chunks land
        # in the file in the same order gTTS would have written them.
        chunks = list(
            executor.map(
                lambda pr: _fetch_tts_chunk(pr, tts.timeout or 10),
                prepared_requests,
            )
        )

    with open(filepath, "wb") as f:
        for chunk in chunks:
            f.write(chunk)


# ==========================================================
# STRIP MARKDOWN BEFORE SPEAKING
# ==========================================================
#
# Gemini's answers use markdown (**bold** headers, numbered lists,
# "- " bullets) because that reads well on screen in the text chat.
# But this same text is also fed straight to gTTS for voice replies,
# which has no concept of markdown - it just speaks the raw
# characters, so a farmer would hear literal asterisks and list
# markers, or an unnaturally choppy line-by-line reading instead of
# flowing spoken sentences. Only the TTS input is cleaned here; the
# original, fully-formatted text is still what's shown on screen and
# returned as `assistant_reply`.
# ==========================================================

def _strip_markdown_for_speech(text: str) -> str:

    # Bold / italic markers.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Markdown headers ("# ", "## ", etc.) at the start of a line.
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)

    # Numbered list markers ("1. ", "2. ") at the start of a line.
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)

    # Bullet markers ("- ", "* ") at the start of a line.
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)

    # Line breaks read awkwardly in speech; turn them into sentence
    # breaks instead so gTTS produces one flowing paragraph.
    text = re.sub(r"\n+", ". ", text)

    return re.sub(r"\s+", " ", text).strip()


async def text_to_speech(text: str, language: str = "en"):

    if not text:
        return {"audio_url": None}

    speech_text = _strip_markdown_for_speech(text)

    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    def _synthesize(text_to_speak: str, lang: str) -> None:
        # gTTS's default tokenizer splits on every comma/period, then
        # only ever shrinks chunks that are still too long - it never
        # merges already-small chunks back together. Skipping that
        # punctuation-based split lets gTTS's own chunk-size-limiting
        # step greedily fill each chunk toward the ~100-character API
        # limit on whitespace instead, roughly halving chunk count on
        # top of the parallel fetch above (measured together on a
        # real multi-paragraph answer: 46.8s sequential -> 3.5s).
        tts = gTTS(
            text=text_to_speak,
            lang=lang,
            tokenizer_func=lambda t: [t],
        )
        try:
            _save_tts_in_parallel(tts, filepath)
        except Exception as exc:
            print("Parallel TTS fetch failed, falling back to sequential:", exc)
            tts.save(filepath)

    try:
        # Blocking (network + thread pool) - run off the event loop
        # so one farmer's long answer doesn't stall every other
        # request the server is handling at the same time.
        await asyncio.to_thread(_synthesize, speech_text, language)
    except Exception as exc:
        print("TTS error for language", language, ":", exc)

        if language == "en":
            return {"audio_url": None}

        try:
            await asyncio.to_thread(_synthesize, speech_text, "en")
        except Exception as fallback_exc:
            print("TTS fallback error:", fallback_exc)
            return {"audio_url": None}

    return {
        "audio_url": f"/generated_audio/{filename}"
    }