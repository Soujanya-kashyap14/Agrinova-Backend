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
import uuid
import subprocess
import wave
from typing import Dict, Any, Optional


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
) -> Dict[str, Any]:

    return model.transcribe(
        wav_path,

        # IMPORTANT:
        # Let Whisper detect the language.
        language=None,

        # We want transcription, not translation.
        task="transcribe",

        # CPU safe.
        fp16=False,

        # Deterministic first pass.
        temperature=0,

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

        # Domain vocabulary.
        initial_prompt=INITIAL_PROMPT,
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

    return False


# ==========================================================
# SECOND PASS
# ==========================================================
#
# We retry with slightly different decoding settings,
# BUT we still allow Whisper to detect the language.
#
# We do NOT run forced Kannada/Hindi/Telugu/Tamil/etc.
#
# This is the critical difference from the previous version.
# ==========================================================

def second_pass(
    wav_path: str,
) -> Dict[str, Any]:

    print(
        "\n========== WHISPER SECOND PASS =========="
    )

    result = model.transcribe(
        wav_path,

        language=None,

        task="transcribe",

        fp16=False,

        temperature=0.2,

        condition_on_previous_text=False,

        no_speech_threshold=0.2,

        compression_ratio_threshold=2.8,

        logprob_threshold=-1.2,

        word_timestamps=False,

        initial_prompt=INITIAL_PROMPT,
    )

    print_result(
        result,
        "SECOND WHISPER PASS",
    )

    return result


# ==========================================================
# CHOOSE BETWEEN TWO WHISPER PASSES
# ==========================================================

def choose_result(
    first: Dict[str, Any],
    second: Dict[str, Any],
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
    # Do not change a language merely because the second
    # pass has a tiny numerical advantage.
    #
    # Require a meaningful improvement.
    # ------------------------------------------------------

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

        first_result = transcribe_audio(
            wav_path
        )

        print_result(
            first_result,
            "FIRST WHISPER PASS",
        )

        # --------------------------------------------------
        # SECOND PASS ONLY WHEN NECESSARY
        # --------------------------------------------------

        final_result = first_result

        if is_suspicious(
            first_result,
            duration,
        ):

            print(
                "\n========== "
                "SUSPICIOUS TRANSCRIPTION "
                "=========="
            )

            print(
                "Whisper result is uncertain."
            )

            second_result = second_pass(
                wav_path
            )

            final_result = choose_result(
                first_result,
                second_result,
            )

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