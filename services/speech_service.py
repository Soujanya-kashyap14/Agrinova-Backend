"""
AgriNova Speech-to-Text Service

Stable multilingual speech recognition using OpenAI Whisper.

Supported languages:
    English
    Kannada
    Hindi
    Telugu
    Tamil
    Malayalam

Design goals:
    - Whisper language detection is the primary source of truth.
    - Do NOT switch languages simply because another forced-language
      transcription has a slightly better log probability.
    - Short English phrases such as "Hello AgriNova" and
      "How are you?" should remain English.
    - Avoid language hallucination caused by forced transcription.
    - Preserve the detected language for Gemini and TTS.
    - Convert browser WebM/Opus audio to clean 16 kHz mono WAV.
"""

import os
import re
import sys
import uuid
import subprocess
import wave
import zlib
from typing import Dict, Any, Optional

import requests

try:
    from config import settings
except Exception:
    settings = None


# Defensive: on Windows, a console's default codepage (e.g. cp1252)
# cannot encode Kannada/Telugu/Tamil/Malayalam/Hindi script. Without
# this, print()-ing recognized non-English text below raises
# UnicodeEncodeError, which previously crashed the whole speech
# recognition request. app.py reconfigures this too at process
# startup; repeating it here protects any other entrypoint that
# imports this module directly (scripts, tests).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


import whisper


# ==========================================================
# DIRECTORIES
# ==========================================================

UPLOAD_DIR = "uploads/audio"
CONVERTED_DIR = "uploads/converted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONVERTED_DIR, exist_ok=True)


# ==========================================================
# WHISPER MODEL
# ==========================================================

print("==================================================")
print("Loading Whisper model...")
print("Model: small")
print("==================================================")

model = whisper.load_model("small")

print("Whisper model loaded successfully.")


# ==========================================================
# SUPPORTED LANGUAGES
# ==========================================================

SUPPORTED_LANGUAGES = {
    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam",
}


# ==========================================================
# GROQ CLOUD WHISPER (OPTIONAL, PREFERRED WHEN CONFIGURED)
# ==========================================================
#
# whisper-large-v3 on Groq's hardware is both far more accurate
# than the local "small" model this machine can realistically run
# (large-v3 has roughly 10x the parameters of small) and much
# faster (cloud GPU vs. this machine's CPU, which we measured
# taking 135s to transcribe 3 seconds of audio with the "medium"
# model). It only activates when GROQ_API_KEY is set; the full
# local pipeline below remains the automatic fallback otherwise or
# if a Groq request fails for any reason (network, rate limit).
# ==========================================================

GROQ_TRANSCRIPTION_URL = (
    "https://api.groq.com/openai/v1/audio/transcriptions"
)

GROQ_MODEL = "whisper-large-v3-turbo"

GROQ_LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "kannada": "kn",
    "hindi": "hi",
    "telugu": "te",
    "tamil": "ta",
    "malayalam": "ml",
}


def transcribe_via_groq(
    wav_path: str,
    language_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:

    api_key = getattr(
        settings,
        "groq_api_key",
        "",
    ) if settings else ""

    if not api_key:
        return None

    try:

        with open(wav_path, "rb") as audio_file:

            files = {
                "file": (
                    os.path.basename(wav_path),
                    audio_file,
                    "audio/wav",
                ),
            }

            data = {
                "model": GROQ_MODEL,
                "response_format": "verbose_json",
                "temperature": 0,
            }

            if language_hint:
                data["language"] = language_hint

            response = requests.post(
                GROQ_TRANSCRIPTION_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
                files=files,
                data=data,
                timeout=30,
            )

        if not response.ok:

            print(
                "[GROQ] Transcription request failed:",
                response.status_code,
                response.text,
            )

            return None

        payload = response.json()

        text = payload.get("text", "") or ""

        language_name = (
            payload.get("language") or ""
        ).strip().lower()

        language = GROQ_LANGUAGE_NAME_TO_CODE.get(
            language_name,
            language_hint or "en",
        )

        return {
            "text": text,
            "language": language,
            "segments": payload.get("segments", []) or [],
        }

    except Exception as exc:

        print(
            "[GROQ] Transcription error:",
            exc,
        )

        return None


# ==========================================================
# AGRINOVA INITIAL PROMPT
# ==========================================================
#
# This gives Whisper useful vocabulary without forcing English.
#

INITIAL_PROMPT = (
    "AgriNova. "
    "Hello AgriNova. "
    "How are you? "
    "How can you help me? "
    "agriculture assistant. "
    "farmer. "
    "farming. "
    "crop. "
    "weather. "
    "mandi. "
    "market price. "
    "fertilizer. "
    "pesticide. "
    "irrigation. "
    "crop disease."
)


# ==========================================================
# COMMON ENGLISH VOICE PHRASES
# ==========================================================
#
# These are NOT translations.
# They are only used to recognize obvious English speech
# when Whisper produces a minor transcription variation.
#

COMMON_ENGLISH_PHRASES = {
    "hello agrinova": "Hello AgriNova",
    "hello agri nova": "Hello AgriNova",
    "helo agrinova": "Hello AgriNova",
    "helo agri nova": "Hello AgriNova",
    "hi agrinova": "Hi AgriNova",
    "hi agri nova": "Hi AgriNova",
    "hey agrinova": "Hey AgriNova",
    "hey agri nova": "Hey AgriNova",
    "hello agrinova ai": "Hello AgriNova",
    "hi agrinova ai": "Hi AgriNova",
    "hey agrinova ai": "Hey AgriNova",
}


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text: str) -> str:
    """
    Clean Whisper output without changing its language.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # Remove obvious leading/trailing punctuation noise.
    text = text.strip(
        " \t\r\n.,!?;:"
    )

    return text.strip()


def is_prompt_hallucination(text: str) -> bool:
    normalized = re.sub(
        r"[^a-z\s]",
        "",
        clean_text(text).lower(),
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in {
        "agriculture assistant farmer",
        "agriculture assistant farmer farming crop",
        "agriculture assistant",
    }


# ==========================================================
# KNOWN WHISPER / YOUTUBE-CAPTION HALLUCINATIONS
# ==========================================================
#
# Whisper was trained on huge volumes of YouTube caption data.
# When the actual audio has little or no real speech (quiet
# mic, background noise, dead air), it frequently hallucinates
# a common video outro phrase instead of admitting it heard
# nothing - "Thanks for watching!", "Please subscribe", etc.
# These are grammatically normal English, so the repetition
# guard above (which looks for repeated characters/tokens)
# does not catch them; they need their own exact-match list.
# This is a well-documented, extensively reported Whisper
# failure mode, not specific to this app.
# ==========================================================

KNOWN_WHISPER_HALLUCINATIONS = {
    "thanks for watching",
    "thank you for watching",
    "thanks for watching this video",
    "thank you for watching this video",
    "thanks for watching and ill see you in the next video",
    "please subscribe",
    "dont forget to subscribe",
    "like and subscribe",
    "please like and subscribe",
    "subscribe to my channel",
    "see you in the next video",
    "see you next time",
}


def is_known_hallucination(text: str) -> bool:

    normalized = re.sub(
        r"[^a-z\s]",
        "",
        clean_text(text).lower(),
    )

    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized in KNOWN_WHISPER_HALLUCINATIONS


def is_low_quality_text(text: str) -> bool:
    """Catches every kind of garbage output this pipeline has
    actually seen in practice: repetition loops, the AgriNova
    prompt leaking back as output, and Whisper's well-known
    YouTube-caption hallucinations."""

    return (
        is_repetitive_gibberish(text)
        or is_prompt_hallucination(text)
        or is_known_hallucination(text)
    )


# ==========================================================
# REPETITION / HALLUCINATION GUARD
# ==========================================================
#
# Forcing Whisper into the wrong language (e.g. the farmer's
# selected app language doesn't match what they actually said)
# is a known trigger for repetition hallucination: the model
# gets stuck repeating the same character or short token dozens
# of times ("tatatatata...", "nunununu..."). The dangerous part
# is that this garbage often has a HIGH avg_logprob, because the
# model is locally very confident about repeating the token it
# just emitted, even though the output is globally meaningless.
# avg_logprob alone cannot catch this, so it needs a dedicated
# check based on how repetitive/compressible the text is.
# ==========================================================

def _compression_ratio(text: str) -> float:

    if not text:
        return 0.0

    encoded = text.encode("utf-8")

    compressed = zlib.compress(encoded)

    if not compressed:
        return 0.0

    return len(encoded) / len(compressed)


def is_repetitive_gibberish(text: str) -> bool:

    cleaned = clean_text(text)

    # Too short for the ratio to be meaningful; a couple of
    # repeated words in a short phrase is normal speech, not
    # hallucination.
    if len(cleaned) < 12:
        return False

    if _compression_ratio(cleaned) > 2.4:
        return True

    # Catch a dominant repeated character even in cases where
    # zlib's ratio isn't extreme (short-ish hallucinations).
    counts: Dict[str, int] = {}

    letters = [c for c in cleaned if not c.isspace()]

    if not letters:
        return False

    for char in letters:
        counts[char] = counts.get(char, 0) + 1

    most_common = max(counts.values())

    return most_common / len(letters) > 0.5


# ==========================================================
# NORMALIZE OBVIOUS ENGLISH GREETINGS
# ==========================================================

def normalize_common_english(text: str) -> str:

    cleaned = clean_text(text)

    if not cleaned:
        return ""

    normalized = cleaned.lower()

    # Collapse punctuation and repeated spaces.
    normalized = re.sub(
        r"[^a-z0-9\s]",
        "",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    # Exact / near-exact AgriNova greetings.
    if normalized in COMMON_ENGLISH_PHRASES:
        return COMMON_ENGLISH_PHRASES[normalized]

    # Common Whisper duplication:
    #
    # "hello AgriNova hello"
    # "hello AgriNova"
    #
    if (
        "hello agrinova" in normalized
        or "hello agri nova" in normalized
    ):
        return "Hello AgriNova"

    if (
        "hi agrinova" in normalized
        or "hi agri nova" in normalized
    ):
        return "Hi AgriNova"

    if (
        "hey agrinova" in normalized
        or "hey agri nova" in normalized
    ):
        return "Hey AgriNova"

    return cleaned


# ==========================================================
# ENGLISH TEXT HEURISTIC
# ==========================================================
#
# This is deliberately conservative.
#
# It is NOT used to translate text.
# It is only used to prevent a short English phrase from
# being incorrectly replaced by a forced-language result.
#

ENGLISH_WORDS = {
    "hello",
    "hi",
    "hey",
    "how",
    "are",
    "you",
    "your",
    "what",
    "is",
    "the",
    "weather",
    "today",
    "tomorrow",
    "price",
    "market",
    "mandi",
    "crop",
    "crops",
    "farmer",
    "farm",
    "fertilizer",
    "fertiliser",
    "pesticide",
    "pesticides",
    "water",
    "irrigation",
    "disease",
    "diseases",
    "help",
    "please",
    "tell",
    "me",
    "can",
    "could",
    "would",
    "give",
    "show",
    "cost",
    "costs",
    "rate",
    "rates",
    "today",
    "tomato",
    "tomatoes",
    "potato",
    "potatoes",
    "onion",
    "onions",
    "rice",
    "wheat",
    "corn",
    "maize",
    "good",
    "fine",
    "doing",
    "thank",
    "thanks",
}


def looks_like_english(text: str) -> bool:

    cleaned = clean_text(text).lower()

    if not cleaned:
        return False

    # Keep only ASCII words.
    words = re.findall(
        r"[a-z]+",
        cleaned,
    )

    if not words:
        return False

    matches = sum(
        1
        for word in words
        if word in ENGLISH_WORDS
    )

    # For one-word speech, require an exact common word.
    if len(words) == 1:
        return matches == 1

    # For multiple words, at least half should be common
    # English words.
    return (
        matches >= 2
        and matches / len(words) >= 0.5
    )


# ==========================================================
# CONVERT WEBM -> WAV
# ==========================================================

def convert_to_wav(
    input_path: str
) -> str:

    output_path = os.path.join(
        CONVERTED_DIR,
        f"{uuid.uuid4()}.wav",
    )

    print("\n========== AUDIO CONVERSION ==========")
    print("Input :", input_path)
    print("Output:", output_path)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,

        # Mono.
        "-ac",
        "1",

        # Whisper sample rate.
        "-ar",
        "16000",

        # PCM WAV.
        "-acodec",
        "pcm_s16le",

        # Cut low-frequency handling/wind rumble below speech
        # range, then normalize loudness to a consistent target.
        # Farmers often speak quietly or hold the phone at a
        # distance; Whisper is noticeably less accurate on
        # quiet/inconsistent input, and this costs nothing when
        # the input is already clean.
        "-af",
        "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11",

        output_path,
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:

            print(
                "\n========== FFMPEG ERROR =========="
            )

            print(result.stderr)

            raise RuntimeError(
                "FFmpeg could not convert the audio."
            )

        if not os.path.exists(output_path):

            raise RuntimeError(
                "WAV file was not created."
            )

        size = os.path.getsize(
            output_path
        )

        print(
            "WAV created successfully."
        )

        print(
            "WAV size:",
            size,
            "bytes",
        )

        if size < 1000:

            raise RuntimeError(
                "Converted WAV file is too small."
            )

        return output_path

    except FileNotFoundError:

        raise RuntimeError(
            "FFmpeg is not installed or "
            "not available in PATH."
        )


# ==========================================================
# WAV INFORMATION
# ==========================================================

def inspect_wav(
    wav_path: str
) -> float:

    try:

        with wave.open(
            wav_path,
            "rb",
        ) as wf:

            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()

            duration = (
                frames /
                float(sample_rate)
            )

            print(
                "\n========== WAV INFORMATION =========="
            )

            print(
                "Channels    :",
                channels,
            )

            print(
                "Sample width:",
                sample_width,
            )

            print(
                "Sample rate :",
                sample_rate,
            )

            print(
                "Frames      :",
                frames,
            )

            print(
                "Duration    :",
                round(duration, 2),
                "seconds",
            )

            return duration

    except Exception as e:

        print(
            "Could not inspect WAV:",
            e,
        )

        return 0.0


# ==========================================================
# TRANSCRIPTION QUALITY
# ==========================================================

def get_quality(
    result: Dict[str, Any]
) -> Dict[str, float]:

    segments = result.get(
        "segments",
        [],
    )

    if not segments:

        return {
            "avg_logprob": -10.0,
            "max_no_speech": 1.0,
        }

    log_probs = []
    no_speech_probs = []

    for segment in segments:

        avg_logprob = segment.get(
            "avg_logprob"
        )

        no_speech_prob = segment.get(
            "no_speech_prob"
        )

        if avg_logprob is not None:

            try:
                log_probs.append(
                    float(avg_logprob)
                )
            except Exception:
                pass

        if no_speech_prob is not None:

            try:
                no_speech_probs.append(
                    float(no_speech_prob)
                )
            except Exception:
                pass

    avg_logprob = (
        sum(log_probs) /
        len(log_probs)
        if log_probs
        else -10.0
    )

    max_no_speech = (
        max(no_speech_probs)
        if no_speech_probs
        else 1.0
    )

    return {
        "avg_logprob": avg_logprob,
        "max_no_speech": max_no_speech,
    }


# ==========================================================
# PRINT RESULT
# ==========================================================

def print_result(
    result: Dict[str, Any],
    label: str,
):

    quality = get_quality(
        result
    )

    print(
        f"\n========== {label} =========="
    )

    print(
        "Detected language:",
        result.get(
            "language",
            "unknown",
        ),
    )

    print(
        "Segments:",
        len(
            result.get(
                "segments",
                [],
            )
        ),
    )

    print(
        "Average log probability:",
        round(
            quality["avg_logprob"],
            3,
        ),
    )

    print(
        "Maximum no-speech probability:",
        round(
            quality["max_no_speech"],
            3,
        ),
    )

    print(
        "Recognized text:",
        result.get(
            "text",
            "",
        ).strip(),
    )


# ==========================================================
# RUN WHISPER
# ==========================================================

def transcribe_audio(
    wav_path: str,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:

    return model.transcribe(
        wav_path,

        # IMPORTANT:
        # Let Whisper detect the language unless the caller is using a
        # language hint for a low-confidence retry.
        language=language_hint,

        # We want transcription, not translation.
        task="transcribe",

        # CPU safe.
        fp16=False,

        # Deterministic first pass.
        temperature=0,

        # Beam search materially reduces recognition errors versus
        # greedy decoding (the default when beam_size is unset at
        # temperature 0). This pass runs on every request, so we
        # keep the beam small (3, not the 5 used on the retry
        # passes below) — measured on this machine, beam_size=5
        # here roughly doubled-to-quadrupled per-request latency
        # (up to ~45s for a short clip), which is worse for a
        # voice assistant than the accuracy it bought.
        beam_size=3,

        # Prevent previous text from contaminating
        # short voice messages.
        condition_on_previous_text=False,

        # Don't aggressively classify normal speech
        # as silence.
        no_speech_threshold=0.4,

        # Reject obvious hallucinations.
        compression_ratio_threshold=2.8,

        logprob_threshold=-1.2,

        word_timestamps=False,

        # Prompt vocabulary is useful for the automatic pass, but it can
        # become hallucinated text when a forced-language retry is quiet.
        initial_prompt=(
            INITIAL_PROMPT
            if language_hint is None
            else None
        ),
    )


# ==========================================================
# SUSPICIOUS RESULT CHECK
# ==========================================================

def is_suspicious(
    result: Dict[str, Any],
    duration: float,
) -> bool:

    text = clean_text(
        result.get(
            "text",
            "",
        )
    )

    if not text:
        return True

    quality = get_quality(
        result
    )

    avg_logprob = quality[
        "avg_logprob"
    ]

    max_no_speech = quality[
        "max_no_speech"
    ]

    # Extremely low confidence.
    if avg_logprob < -1.15:
        return True

    # Strong silence indication.
    if max_no_speech > 0.85:
        return True

    # Very short audio with weak confidence.
    if (
        duration <= 2.5
        and avg_logprob < -0.85
    ):
        return True

    # --------------------------------------------------------
    # IMPLAUSIBLY LITTLE TEXT FOR THE AUDIO LENGTH.
    #
    # A farmer speaking for 5+ seconds producing a single word
    # ("water" instead of "what's the weather today") is a
    # classic truncation/mis-hearing pattern that a confidence
    # score alone won't catch - Whisper can be quite confident
    # about the one word it did latch onto. Natural continuous
    # speech runs at roughly 2+ words per second, so demand at
    # least a fraction of that before trusting a short result
    # from a long recording.
    # --------------------------------------------------------

    word_count = len(text.split())

    if duration >= 3.0:

        expected_min_words = max(
            2,
            int(duration / 2.5),
        )

        if word_count < expected_min_words:
            return True

    return False


# ==========================================================
# SECOND PASS
# ==========================================================
#
# We retry with slightly different decoding settings. When the UI supplied
# a supported language, it is used as a hint only for this uncertain pass.
# ==========================================================

def second_pass(
    wav_path: str,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:

    print(
        "\n========== WHISPER SECOND PASS =========="
    )

    result = model.transcribe(
        wav_path,

        language=language_hint,

        task="transcribe",

        fp16=False,

        temperature=0.2,

        # At temperature > 0, Whisper samples rather than beam-
        # searches; best_of draws multiple candidates and keeps
        # the highest-scoring one, which is the sampling-mode
        # analogue of beam_size used in the first pass.
        best_of=5,

        condition_on_previous_text=False,

        no_speech_threshold=0.2,

        compression_ratio_threshold=2.8,

        logprob_threshold=-1.2,

        word_timestamps=False,

        # The prompt vocabulary is English. Priming a forced
        # non-English decode with English words biases the
        # model toward English tokens, undermining the very
        # language we are forcing. Only use it when Whisper is
        # still auto-detecting the language itself.
        initial_prompt=(
            INITIAL_PROMPT
            if language_hint is None
            else None
        ),
    )

    print_result(
        result,
        "SECOND WHISPER PASS",
    )

    return result


def recovery_pass(
    wav_path: str,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Use permissive decoding once when normal passes return no text."""
    # best_of=5 previously sampled 5 separate decodes and kept the
    # best one - a 5x compute cost for the one pass most likely to
    # run after an already slow/uncertain recording. On CPU-only
    # hardware that turns an ~8s decode into ~40s, which is a large
    # chunk of the request's total latency for marginal benefit
    # over a single decode at this permissive threshold.
    return model.transcribe(
        wav_path,
        language=language_hint,
        task="transcribe",
        fp16=False,
        temperature=0.4,
        condition_on_previous_text=False,
        no_speech_threshold=0.65,
        logprob_threshold=-2.0,
        compression_ratio_threshold=3.5,
        word_timestamps=False,
    )


# ==========================================================
# CHOOSE BETWEEN TWO WHISPER PASSES
# ==========================================================

def choose_result(
    first: Dict[str, Any],
    second: Dict[str, Any],
    prefer_second: bool = False,
) -> Dict[str, Any]:

    first_text = clean_text(
        first.get(
            "text",
            "",
        )
    )

    second_text = clean_text(
        second.get(
            "text",
            "",
        )
    )

    if not first_text:
        return second

    if not second_text:
        return first

    # ------------------------------------------------------
    # REJECT REPETITION HALLUCINATION FIRST
    #
    # This must run before any confidence comparison. A
    # repetition hallucination can have a HIGHER avg_logprob
    # than a genuinely correct transcription (the model is
    # very confident about repeating the token it just said),
    # so confidence alone would pick the garbage text.
    # ------------------------------------------------------

    first_garbage = is_low_quality_text(first_text)
    second_garbage = is_low_quality_text(second_text)

    if second_garbage and not first_garbage:

        print(
            "Second pass looks like repetition "
            "hallucination; keeping first pass."
        )

        return first

    if first_garbage and not second_garbage:

        print(
            "First pass looks like repetition "
            "hallucination; using second pass."
        )

        return second

    first_quality = get_quality(
        first
    )

    second_quality = get_quality(
        second
    )

    first_score = (
        first_quality["avg_logprob"]
    )

    second_score = (
        second_quality["avg_logprob"]
    )

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # Do not change a result merely because the other pass
    # has a tiny numerical advantage. Require a meaningful
    # improvement before switching away from the default.
    # ------------------------------------------------------

    if prefer_second:

        # --------------------------------------------------
        # The farmer explicitly selected this language in the
        # app. Whisper's free auto-detection only disagreed
        # with that selection, it wasn't necessarily low
        # confidence about it. Keep trusting the farmer's
        # selection unless the auto-detected pass is
        # meaningfully more confident.
        # --------------------------------------------------

        if first_score > second_score + 0.15:

            print(
                "Auto-detected pass has meaningful "
                "confidence advantage over the farmer's "
                "selected language."
            )

            return first

        return second

    if second_score > first_score + 0.15:

        print(
            "Second pass has meaningful "
            "confidence improvement."
        )

        return second

    return first


# ==========================================================
# FINAL LANGUAGE NORMALIZATION
# ==========================================================

def normalize_language_and_text(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    language = result.get(
        "language",
        "en",
    )

    text = clean_text(
        result.get(
            "text",
            "",
        )
    )

    # ------------------------------------------------------
    # Obvious English AgriNova greeting.
    # ------------------------------------------------------

    greeting = normalize_common_english(
        text
    )

    if greeting != text:

        text = greeting

        language = "en"

    # ------------------------------------------------------
    # If Whisper says an unsupported language but the
    # actual transcription is clearly English, keep English.
    #
    # This prevents cases like:
    #
    # "How are you?"
    #
    # being routed to Tamil because of a bad short-audio
    # language detection.
    # ------------------------------------------------------

    elif language != "en":

        if looks_like_english(text):

            print(
                "\nEnglish text detected despite "
                f"Whisper language '{language}'."
            )

            language = "en"

    # ------------------------------------------------------
    # Keep only supported languages.
    #
    # If Whisper produces an unsupported language and it
    # doesn't look English, preserve its language rather
    # than falsely converting it.
    # ------------------------------------------------------

    return {
        "text": text,
        "language": language,
    }


# ==========================================================
# MAIN SPEECH TO TEXT
# ==========================================================

async def speech_to_text(
    audio_path: str,
    language_hint: Optional[str] = None,
) -> dict:

    print(
        "\n========== WHISPER =========="
    )

    print(
        "Original audio:",
        audio_path,
    )

    # ------------------------------------------------------
    # Validate original file.
    # ------------------------------------------------------

    if not os.path.exists(
        audio_path
    ):

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_path}"
        )

    original_size = os.path.getsize(
        audio_path
    )

    print(
        "Original audio size:",
        original_size,
        "bytes",
    )

    if original_size < 1000:

        print(
            "WARNING: Audio file is "
            "extremely small."
        )

        return {
            "text": "",
            "language": "en",
        }

    # ------------------------------------------------------
    # Convert to WAV.
    # ------------------------------------------------------

    wav_path = convert_to_wav(
        audio_path
    )

    try:

        # --------------------------------------------------
        # Inspect WAV.
        # --------------------------------------------------

        duration = inspect_wav(
            wav_path
        )

        if duration <= 0:

            return {
                "text": "",
                "language": "en",
            }

        # --------------------------------------------------
        # FIRST PASS
        # --------------------------------------------------

        print(
            "\n========== WHISPER TRANSCRIPTION =========="
        )

        print(
            "Using WAV:",
            wav_path,
        )

        print(
            "Duration:",
            round(
                duration,
                2,
            ),
            "seconds",
        )

        normalized_hint = (language_hint or "").strip().lower()
        if normalized_hint not in SUPPORTED_LANGUAGES:
            normalized_hint = None

        # ----------------------------------------------------
        # TRY GROQ CLOUD WHISPER FIRST (IF CONFIGURED)
        # ----------------------------------------------------
        #
        # IMPORTANT: "en" is the UI's default language, which most
        # farmers never touch before speaking - it carries no real
        # signal about what they're about to say. Forcing it onto
        # every request tells Groq "this is English" even when the
        # farmer spoke Kannada/Hindi/etc., and forcing the WRONG
        # language doesn't just fail cleanly - it can make the
        # decoder render the correct phonemes in a completely wrong
        # script (observed in practice: Kannada forced as "en" came
        # back in Gujarati script; auto-detected with no hint at all,
        # it came back in Tamil script instead - still wrong, because
        # closely related South Indian languages are a known hard
        # case for auto-detection alone on short clips).
        #
        # A genuine non-English selection is a different story: a
        # farmer doesn't end up on "kn" or "hi" by accident the way
        # they do on the default "en", so it's real, deliberate
        # signal - and confirmed by direct testing to resolve exactly
        # this kind of similar-script confusion (forcing "kn" turned
        # the same Tamil-script mistranscription into near-perfect
        # Kannada, avg_logprob -0.09 vs. auto-detect's wrong guess).
        # So: trust a genuine non-English hint, distrust the
        # ambiguous "en" default.
        # ----------------------------------------------------

        groq_language_hint = (
            normalized_hint
            if normalized_hint and normalized_hint != "en"
            else None
        )

        groq_result = transcribe_via_groq(
            wav_path,
            groq_language_hint,
        )

        if groq_result is not None:

            groq_text = clean_text(
                groq_result.get("text", "")
            )

            print(
                "\n========== GROQ CLOUD RESULT ==========\n"
                f"Text: {groq_text!r}\n"
                f"Language: {groq_result.get('language')}\n"
                f"Segments: {len(groq_result.get('segments', []))}"
            )

            if (
                groq_text
                and not is_low_quality_text(groq_text)
                and not is_suspicious(groq_result, duration)
            ):

                print(
                    "\n========== "
                    "USING GROQ CLOUD TRANSCRIPTION "
                    "=========="
                )

                final = normalize_language_and_text(
                    groq_result
                )

                print("Text:", final["text"])
                print("Language:", final["language"])

                return {
                    "text": final["text"],
                    "language": final["language"],
                }

            print(
                "Groq result looked low quality; "
                "falling back to the local pipeline."
            )

        first_result = transcribe_audio(wav_path)

        print_result(
            first_result,
            "FIRST WHISPER PASS",
        )

        # Groq (whisper-large-v3-turbo, cloud GPU) is a materially
        # stronger model than this machine's local "small" model.
        # When Groq already attempted this exact audio and got
        # rejected as low quality, the expensive local retries below
        # (second_pass's 5x-cost sampling, then recovery_pass) almost
        # never recover real text in practice - they just produce
        # their own hallucinations (repeated tokens, "Thanks for
        # watching!") that get correctly rejected anyway, after
        # burning a large amount of CPU time on a request that was
        # already heading to "could not hear you clearly" regardless.
        # Skip straight to a single honesty-gate check on the cheap
        # first pass instead when Groq was already tried. When Groq
        # isn't configured at all, this local cascade is the only
        # available safety net, so it still runs in full.
        groq_attempted = groq_result is not None

        # --------------------------------------------------
        # SECOND PASS ONLY WHEN NECESSARY
        # --------------------------------------------------
        #
        # Two independent reasons to re-check the first pass:
        #
        # 1. Whisper itself is unsure (is_suspicious).
        # 2. Whisper is confident, but detected a DIFFERENT
        #    language than the one the farmer selected in the
        #    app. High confidence on the wrong language still
        #    produces wrong text, and is_suspicious alone can
        #    never catch that case since it only looks at
        #    Whisper's own confidence, not language agreement.
        # --------------------------------------------------

        final_result = first_result

        uncertain = is_suspicious(
            first_result,
            duration,
        )

        detected_language = first_result.get(
            "language"
        )

        first_avg_logprob = get_quality(
            first_result
        )["avg_logprob"]

        # Only worth a second, expensive decode when Whisper
        # ISN'T already very confident. A highly confident
        # detection (e.g. -0.3) that merely disagrees with the
        # farmer's selected app language is almost always
        # correct as-is (the gibberish guard below would reject
        # a forced retry anyway), so retrying here would just
        # double this request's latency for no real benefit -
        # a real cost on CPU-only hardware.
        language_mismatch = (
            normalized_hint is not None
            and detected_language != normalized_hint
            and first_avg_logprob < -0.5
        )

        if not groq_attempted and (uncertain or language_mismatch):

            if uncertain:

                print(
                    "\n========== "
                    "SUSPICIOUS TRANSCRIPTION "
                    "=========="
                )

                print(
                    "Whisper result is uncertain."
                )

            else:

                print(
                    "\n========== "
                    "LANGUAGE MISMATCH "
                    "=========="
                )

                print(
                    "Whisper detected",
                    detected_language,
                    "but the farmer selected",
                    normalized_hint,
                )

            second_result = second_pass(
                wav_path,
                normalized_hint,
            )

            final_result = choose_result(
                first_result,
                second_result,
                # Only bias toward the farmer's selected
                # language when Whisper's own confidence
                # didn't already flag a problem. If it's both
                # uncertain AND mismatched, fall back to the
                # normal "does the retry meaningfully improve
                # things" comparison.
                prefer_second=(
                    language_mismatch and not uncertain
                ),
            )

        # ----------------------------------------------------
        # If the first pass was already flagged uncertain AND
        # we ended up keeping it anyway (because the retry
        # turned out to be garbage/hallucination), that first
        # result was never actually validated - it's just the
        # "least bad" of two untrustworthy options. Treat that
        # the same as low-quality text so recovery_pass gets a
        # real attempt, instead of silently accepting a result
        # we already know was uncertain.
        # ----------------------------------------------------

        kept_unvalidated_first = (
            uncertain
            and final_result is first_result
            and not groq_attempted
        )

        if not groq_attempted and (
            not clean_text(final_result.get("text", ""))
            or is_low_quality_text(final_result.get("text", ""))
            or kept_unvalidated_first
        ):
            print("\n========== WHISPER RECOVERY PASS ==========")
            final_result = recovery_pass(
                wav_path,
                normalized_hint,
            )
            print_result(final_result, "RECOVERY WHISPER PASS")

            # ------------------------------------------------
            # HONESTY GATE
            #
            # recovery_pass is a last resort with very
            # permissive thresholds - it will produce SOME
            # text almost no matter what the audio contains.
            # If its own confidence is still poor (or it's
            # still a known hallucination), every strategy has
            # now failed. Forwarding that guess to Gemini as if
            # it were real speech produces a confidently wrong,
            # off-topic answer (e.g. a low-confidence "Thank
            # you" guess getting "You're welcome!" back) - worse
            # than honestly admitting the audio wasn't
            # understood and asking the farmer to try again.
            # ------------------------------------------------

            if is_suspicious(
                final_result, duration
            ) or is_low_quality_text(
                final_result.get("text", "")
            ):

                print(
                    "Recovery pass is still low quality; "
                    "treating as not understood rather than "
                    "guessing."
                )

                final_result = {
                    "text": "",
                    "language": final_result.get(
                        "language", "en"
                    ),
                }

        elif groq_attempted and (
            is_suspicious(final_result, duration)
            or is_low_quality_text(final_result.get("text", ""))
        ):

            # Groq already tried and failed on this audio; the local
            # first pass (the cheap cross-check) also came back bad.
            # Trust that verdict instead of spending tens of seconds
            # more on second_pass/recovery_pass to reach the same
            # "not understood" conclusion.
            print(
                "Local cross-check after Groq is also low quality; "
                "treating as not understood rather than spending "
                "more time on expensive local retries."
            )

            final_result = {
                "text": "",
                "language": final_result.get("language", "en"),
            }

        # --------------------------------------------------
        # Normalize final result.
        # --------------------------------------------------

        final = normalize_language_and_text(
            final_result
        )

        text = final[
            "text"
        ]

        language = final[
            "language"
        ]

        # --------------------------------------------------
        # Final quality information.
        # --------------------------------------------------

        quality = get_quality(
            final_result
        )

        print(
            "\n========== TRANSCRIPTION QUALITY =========="
        )

        print(
            "Average log probability:",
            round(
                quality["avg_logprob"],
                3,
            ),
        )

        print(
            "Maximum no-speech probability:",
            round(
                quality["max_no_speech"],
                3,
            ),
        )

        print(
            "Whisper detected language:",
            final_result.get(
                "language",
                "unknown",
            ),
        )

        print(
            "Final text:",
            text,
        )

        print(
            "Final language:",
            language,
        )

        print(
            "Audio duration:",
            round(
                duration,
                2,
            ),
            "seconds",
        )

        # --------------------------------------------------
        # Empty result.
        # --------------------------------------------------

        if not text:

            print(
                "\nWARNING: Whisper could not "
                "recognize speech."
            )

            return {
                "text": "",
                "language": language or "en",
            }

        # --------------------------------------------------
        # Language status.
        # --------------------------------------------------

        if language in SUPPORTED_LANGUAGES:

            print(
                "Supported language:",
                SUPPORTED_LANGUAGES[
                    language
                ],
            )

        else:

            print(
                "Whisper detected language:",
                language,
            )

            print(
                "Language is outside the "
                "primary AgriNova language list."
            )

        # --------------------------------------------------
        # SUCCESS
        # --------------------------------------------------

        print(
            "\n========== "
            "SPEECH RECOGNITION SUCCESS "
            "=========="
        )

        print(
            "Text:",
            text,
        )

        print(
            "Language:",
            language,
        )

        return {
            "text": text,
            "language": language,
        }

    except Exception as e:

        print(
            "\n========== WHISPER ERROR =========="
        )

        print(
            "Error type:",
            type(e),
        )

        print(
            "Error:",
            e,
        )

        raise RuntimeError(
            f"Speech recognition failed: "
            f"{str(e)}"
        )

    finally:

        # --------------------------------------------------
        # Delete temporary WAV.
        # --------------------------------------------------

        try:

            if os.path.exists(
                wav_path
            ):

                os.remove(
                    wav_path
                )

                print(
                    "Temporary WAV deleted."
                )

        except Exception as e:

            print(
                "Could not delete temporary WAV:",
                e,
            )