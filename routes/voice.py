import os
import json
import uuid

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Form,
    HTTPException,
)

from services.speech_service import speech_to_text
from services.assistant_service import generate_response
from services.tts_service import text_to_speech
from utils.conversation import build_question_with_context
from utils.language import detect_script_language
from models.voice import TextChatRequest


router = APIRouter(
    prefix="/voice",
    tags=["Voice Assistant"],
)


UPLOAD_DIR = "uploads/audio"

SUPPORTED_LANGUAGES = {
    "en",
    "kn",
    "hi",
    "te",
    "ta",
    "ml",
}

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)


@router.post("/chat")
async def voice_chat(
    file: UploadFile = File(...),

    latitude: str | None = Form(None),
    longitude: str | None = Form(None),
    location: str | None = Form(None),
    language: str | None = Form(None),
    conversation: str | None = Form(None),
):

    filepath = None

    try:

        # ==================================================
        # SAVE AUDIO
        # ==================================================

        extension = (
            file.filename.split(".")[-1]
            if file.filename and "." in file.filename
            else "webm"
        )

        filename = (
            f"{uuid.uuid4()}.{extension}"
        )

        filepath = os.path.join(
            UPLOAD_DIR,
            filename,
        )

        audio_data = await file.read()

        if not audio_data:

            raise HTTPException(
                status_code=400,
                detail="Empty audio file.",
            )

        with open(
            filepath,
            "wb",
        ) as f:

            f.write(audio_data)

        print("\n========== STEP 1 ==========")

        print(
            "Voice audio:",
            filepath,
        )

        print(
            "Audio bytes:",
            len(audio_data),
        )

        # The selected UI language is only a fallback hint for uncertain
        # audio. Whisper still detects the spoken language on its fast pass.
        requested_language = (language or "").strip().lower()
        if requested_language not in SUPPORTED_LANGUAGES:
            requested_language = None

        # ==================================================
        # SPEECH
        # ==================================================

        speech = await speech_to_text(
            filepath,
            language_hint=requested_language,
        )

        print(type(speech))
        print(speech)

        text = (
            speech.get("text", "")
            .strip()
        )

        detected_language = (
            speech.get("language", "en")
            or "en"
        )

        # The transcribed TEXT's own script is ground truth for what
        # language the farmer actually spoke - it can't be stale or
        # defaulted the way a UI toggle can. Only fall back to the UI
        # selection, then to Whisper's own language guess, when the
        # text is plain Latin script and gives no script signal.
        script_language = detect_script_language(text)

        if script_language:
            detected_language = script_language
        elif requested_language:
            detected_language = requested_language

        language = detected_language

        # ==================================================
        # EMPTY SPEECH
        # ==================================================

        if not text:

            empty_answers = {

                "en":
                    "I could not hear you clearly. "
                    "Please speak again.",

                "kn":
                    "ನಿಮ್ಮ ಮಾತು ಸ್ಪಷ್ಟವಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. "
                    "ದಯವಿಟ್ಟು ಮತ್ತೆ ಮಾತನಾಡಿ.",

                "hi":
                    "मैं आपकी आवाज़ स्पष्ट रूप से नहीं सुन पाया। "
                    "कृपया फिर से बोलें।",

                "te":
                    "మీ మాట స్పష్టంగా వినిపించలేదు. "
                    "దయచేసి మళ్లీ మాట్లాడండి.",

                "ta":
                    "உங்கள் குரல் தெளிவாக கேட்கவில்லை. "
                    "மீண்டும் பேசுங்கள்.",

                "ml":
                    "നിങ്ങളുടെ ശബ്ദം വ്യക്തമായി കേൾക്കാനായില്ല. "
                    "ദയവായി വീണ്ടും സംസാരിക്കുക.",
            }

            return {
                "success": False,
                "recognized_text": empty_answers.get(
                    language,
                    empty_answers["en"],
                ),
                "language": language,
                "assistant_reply": empty_answers.get(
                    language,
                    empty_answers["en"],
                ),
                "audio_url": None,
            }

        # ==================================================
        # STEP 2
        # ==================================================

        print("\n========== STEP 2 ==========")

        print(
            "Question:",
            text,
        )

        print(
            "Language:",
            language,
        )

        print(
            "Latitude:",
            latitude,
        )

        print(
            "Longitude:",
            longitude,
        )

        print(
            "Location:",
            location,
        )

        # --------------------------------------------------
        # IMPORTANT:
        # Current assistant_service can use location text.
        #
        # We append location context to the question only
        # when the browser supplied it.
        # --------------------------------------------------

        history = None

        if conversation:
            try:
                parsed = json.loads(conversation)
                if isinstance(parsed, list):
                    history = parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        question_for_assistant = build_question_with_context(
            text,
            history,
            location,
        )

        assistant = await generate_response(
            question_for_assistant,
            language,
            latitude,
            longitude,
        )

        print(type(assistant))
        print(assistant)

        answer = (
            assistant.get("answer", "")
            .strip()
        )

        # ==================================================
        # STEP 3
        # ==================================================

        print("\n========== STEP 3 ==========")

        audio = await text_to_speech(
            answer,
            language,
        )

        print(type(audio))
        print(audio)

        # ==================================================
        # RETURN
        # ==================================================

        return {
            "success": True,
            "recognized_text": text,
            "language": language,
            "assistant_reply": answer,
            "audio_url": audio.get(
                "audio_url"
            ),
            "weather": assistant.get(
                "weather"
            ),
        }

    except HTTPException:
        raise

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        # --------------------------------------------------
        # DELETE ORIGINAL AUDIO
        # --------------------------------------------------

        try:

            if filepath and os.path.exists(
                filepath
            ):

                os.remove(filepath)

                print(
                    "Temporary original audio deleted."
                )

        except Exception as cleanup_error:

            print(
                "Audio cleanup error:",
                cleanup_error,
            )


# ==============================================================
# TEXT CHAT
# ==============================================================
#
# Same Gemini-backed reasoning as /chat, minus the speech-to-text
# step. Lets a farmer type instead of speak - useful by itself,
# and as a reliable fallback when microphone/audio conditions
# make voice recognition unreliable.
# ==============================================================

@router.post("/text-chat")
async def voice_text_chat(
    payload: TextChatRequest,
):

    try:

        question = (payload.question or "").strip()

        if not question:

            raise HTTPException(
                status_code=400,
                detail="Question is required.",
            )

        requested_language = (
            (payload.language or "").strip().lower()
        )

        if requested_language not in SUPPORTED_LANGUAGES:
            requested_language = "en"

        # Same rule as voice chat: what the farmer actually typed
        # (its script) overrides the UI toggle, since the toggle may
        # not match the language the farmer is typing in right now.
        script_language = detect_script_language(question)
        language = script_language if script_language else requested_language

        history = (
            [
                {
                    "sender": message.sender,
                    "text": message.text,
                }
                for message in payload.conversation
            ]
            if payload.conversation
            else None
        )

        question_for_assistant = build_question_with_context(
            question,
            history,
            payload.location,
        )

        print("\n========== TEXT CHAT ==========")
        print("Question:", question)
        print("Language:", language)

        assistant = await generate_response(
            question_for_assistant,
            language,
            payload.latitude,
            payload.longitude,
        )

        answer = (
            assistant.get("answer", "")
            or ""
        ).strip()

        audio = await text_to_speech(
            answer,
            language,
        )

        return {
            "success": True,
            "recognized_text": question,
            "language": language,
            "assistant_reply": answer,
            "audio_url": audio.get("audio_url"),
            "weather": assistant.get("weather"),
        }

    except HTTPException:
        raise

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )