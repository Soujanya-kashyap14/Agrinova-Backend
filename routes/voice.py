import os
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


router = APIRouter(
    prefix="/voice",
    tags=["Voice Assistant"],
)


UPLOAD_DIR = "uploads/audio"

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

        # ==================================================
        # SPEECH
        # ==================================================

        speech = await speech_to_text(
            filepath
        )

        print(type(speech))
        print(speech)

        text = (
            speech.get("text", "")
            .strip()
        )

        language = (
            speech.get("language", "en")
            or "en"
        )

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
                "recognized_text": "",
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

        question_for_assistant = text

        if location:

            question_for_assistant = (
                f"{text}\n\n"
                f"Farmer's current location: {location}"
            )

        assistant = await generate_response(
            question_for_assistant,
            language,
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