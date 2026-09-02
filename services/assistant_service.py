"""
AgriNova AI Assistant Service

Multilingual agriculture assistant.

Handles:
- Weather
- Mandi / market prices
- Market trends
- Crop diseases
- Pests
- Fertilizers
- Pesticides
- Irrigation
- Government schemes
- Organic farming
- Crop grading
- Farming costs

Gemini is used for reasoning and natural-language responses.
Live information is supplied separately whenever available.
"""

import json
import re

from google import genai

from config import get_settings
from services.weather_service import get_live_weather


print("################################################")
print("ASSISTANT SERVICE LOADED")
print("################################################")


# ==========================================================
# SETTINGS
# ==========================================================

settings = get_settings()

if not settings.gemini_api_key:
    raise Exception("GEMINI_API_KEY not found")


client = genai.Client(
    api_key=settings.gemini_api_key
)

MODEL_NAME = "gemini-flash-latest"


# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are AgriNova, a multilingual agriculture assistant
for Indian farmers.

Your job is to give practical, clear and trustworthy
agriculture information.

You can help with:

1. Weather
2. Mandi and market prices
3. Market price trends and predictions
4. Crop diseases
5. Pests and insects
6. Fertilizers
7. Pesticides
8. Irrigation
9. Crop grading
10. Government schemes
11. Subsidies
12. Organic farming
13. Soil health
14. Crop production
15. Farming costs
16. Harvesting
17. Storage
18. Selling advice

IMPORTANT LANGUAGE RULE:

Always answer in the SAME language as the farmer.

Supported languages include:

English
Kannada
Hindi
Telugu
Tamil
Malayalam

If the farmer speaks Kannada, answer Kannada.
If the farmer speaks Hindi, answer Hindi.
If the farmer speaks Telugu, answer Telugu.
If the farmer speaks Tamil, answer Tamil.
If the farmer speaks Malayalam, answer Malayalam.
Otherwise answer English.

IMPORTANT DATA RULES:

- Never invent live weather.
- Never claim a market price is live unless live market
  data has been provided.
- Never claim a prediction is guaranteed.
- Clearly distinguish current price, historical trend,
  estimate and prediction.
- If exact fertilizer/pesticide price is unavailable,
  say that prices vary by brand, dealer and location.
- Never invent a product price and present it as an exact
  current price.
- Give practical farming advice.
- For chemical pesticides, mention that the farmer should
  follow the product label and local agriculture officer
  recommendation.
- Avoid unsafe pesticide mixing instructions.
- Keep answers concise enough for voice.
"""


# ==========================================================
# LANGUAGE NAMES
# ==========================================================

LANGUAGE_NAMES = {
    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam",
}


# ==========================================================
# WEATHER KEYWORDS
# ==========================================================

WEATHER_KEYWORDS = [
    "weather",
    "rain",
    "rainfall",
    "temperature",
    "forecast",
    "humidity",
    "wind",
    "climate",
    "weather today",
    "today weather",

    "ಹವಾಮಾನ",
    "ಮಳೆ",
    "ಮಳೆಯ",
    "ಉಷ್ಣಾಂಶ",
    "ತಾಪಮಾನ",

    "मौसम",
    "बारिश",
    "वर्षा",
    "तापमान",

    "వాతావరణం",
    "వర్షం",
    "ఉష్ణోగ్రత",

    "வானிலை",
    "மழை",
    "வெப்பநிலை",

    "കാലാവസ്ഥ",
    "മഴ",
    "താപനില",
]


def is_weather_question(question: str) -> bool:

    q = question.lower()

    return any(
        keyword.lower() in q
        for keyword in WEATHER_KEYWORDS
    )


# ==========================================================
# MARKET / MANDI KEYWORDS
# ==========================================================

PRICE_KEYWORDS = [
    "price",
    "prices",
    "market",
    "mandi",
    "mandi price",
    "market price",
    "today price",
    "rate",
    "rates",
    "selling price",
    "wholesale price",
    "agmarknet",

    "ಬೆಲೆ",
    "ಮಾರುಕಟ್ಟೆ",
    "ಮಂಡಿ",
    "ದರ",

    "कीमत",
    "मंडी",
    "बाजार",
    "भाव",
    "दर",

    "ధర",
    "మార్కెట్",
    "మండి",

    "விலை",
    "சந்தை",

    "വില",
    "മാർക്കറ്റ്",
]


def is_price_question(question: str) -> bool:

    q = question.lower()

    return any(
        keyword.lower() in q
        for keyword in PRICE_KEYWORDS
    )


# ==========================================================
# FERTILIZER
# ==========================================================

FERTILIZER_KEYWORDS = [
    "fertilizer",
    "fertiliser",
    "npk",
    "urea",
    "dap",
    "potash",
    "mop",
    "ssp",
    "compost",
    "manure",
    "micronutrient",
    "zinc",

    "ಗೊಬ್ಬರ",
    "ಯೂರಿಯಾ",
    "ಡಿಎಪಿ",

    "उर्वरक",
    "खाद",
    "यूरिया",

    "ఎరువు",
    "యూరియా",

    "உரம்",
    "யூரியா",

    "വളം",
    "യൂറിയ",
]


def is_fertilizer_question(question: str) -> bool:

    q = question.lower()

    return any(
        keyword.lower() in q
        for keyword in FERTILIZER_KEYWORDS
    )


# ==========================================================
# PESTICIDE
# ==========================================================

PESTICIDE_KEYWORDS = [
    "pesticide",
    "insecticide",
    "fungicide",
    "herbicide",
    "spray",
    "spraying",
    "chemical",
    "medicine for crop",

    "ಕೀಟನಾಶಕ",
    "ಶಿಲೀಂಧ್ರನಾಶಕ",
    "ಸಿಂಪಡಣೆ",

    "कीटनाशक",
    "फफूंदनाशक",
    "दवा",
    "स्प्रे",

    "పురుగుమందు",
    "శిలీంధ్రనాశిని",
    "స్ప్రే",

    "பூச்சிக்கொல்லி",
    "பூஞ்சைக்கொல்லி",
    "தெளிப்பு",

    "കീടനാശിനി",
    "ഫംഗിസൈഡ്",
    "സ്പ്രേ",
]


def is_pesticide_question(question: str) -> bool:

    q = question.lower()

    return any(
        keyword.lower() in q
        for keyword in PESTICIDE_KEYWORDS
    )


# ==========================================================
# DISEASE
# ==========================================================

DISEASE_KEYWORDS = [
    "disease",
    "leaf disease",
    "leaf spot",
    "fungus",
    "infection",
    "blight",
    "yellow leaf",
    "yellow leaves",
    "wilting",
    "rot",

    "ರೋಗ",
    "ಎಲೆ ರೋಗ",
    "ಹಳದಿ ಎಲೆ",
    "ಶಿಲೀಂಧ್ರ",

    "रोग",
    "पत्ती रोग",
    "पीली पत्ती",
    "फफूंद",

    "వ్యాధి",
    "ఆకు వ్యాధి",

    "நோய்",
    "இலை நோய்",

    "രോഗം",
    "ഇല രോഗം",
]


def is_disease_question(question: str) -> bool:

    q = question.lower()

    return any(
        keyword.lower() in q
        for keyword in DISEASE_KEYWORDS
    )


# ==========================================================
# IRRIGATION
# ==========================================================

IRRIGATION_KEYWORDS = [
    "irrigation",
    "water",
    "watering",
    "drip",
    "sprinkler",

    "ನೀರು",
    "ನೀರಾವರಿ",

    "सिंचाई",
    "पानी",

    "నీరు",
    "నీటిపారుదల",

    "நீர்ப்பாசனம்",
    "தண்ணீர்",

    "ജലസേചനം",
    "വെള്ളം",
]


def is_irrigation_question(question: str) -> bool:

    q = question.lower()

    return any(
        keyword.lower() in q
        for keyword in IRRIGATION_KEYWORDS
    )


# ==========================================================
# SCHEMES
# ==========================================================

SCHEME_KEYWORDS = [
    "scheme",
    "subsidy",
    "pm kisan",
    "pm-kisan",
    "kisan credit card",
    "kcc",
    "government",
    "insurance",
    "crop insurance",
    "farmer scheme",

    "ಯೋಜನೆ",
    "ಸಬ್ಸಿಡಿ",

    "योजना",
    "सब्सिडी",

    "పథకం",
    "సబ్సిడీ",

    "திட்டம்",
    "மானியம்",

    "പദ്ധതി",
    "സബ്സിഡി",
]


def is_scheme_question(question: str) -> bool:

    q = question.lower()

    return any(
        keyword.lower() in q
        for keyword in SCHEME_KEYWORDS
    )


# ==========================================================
# EXTRACT LOCATION
# ==========================================================

def extract_city(question: str) -> str:

    prompt = f"""
Extract the location from the farmer's question.

The location may be:
- city
- district
- village
- town

Return ONLY valid JSON.

Example:

{{"city":"Davangere"}}

If there is no location:

{{"city":""}}

Farmer question:

{question}
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        text = (response.text or "").strip()

        text = text.replace(
            "```json",
            "",
        )

        text = text.replace(
            "```",
            "",
        ).strip()

        data = json.loads(text)

        city = str(
            data.get("city", "")
        ).strip()

        if city:
            return city

    except Exception as e:

        print(
            "City extraction error:",
            e,
        )

    return ""


# ==========================================================
# FORMAT WEATHER
# ==========================================================

def format_weather(weather: dict) -> str:

    if not weather:
        return "Live weather is unavailable."

    # Your weather service returns:
    #
    # {
    #   "current": {...},
    #   "forecast": [...],
    #   "alerts": [...]
    # }

    current = weather.get(
        "current",
        weather,
    )

    if not current:
        return "Live weather is unavailable."

    return f"""
Location: {current.get("location", "Unknown")}

Temperature: {current.get("temperature", "N/A")} °C

Condition: {current.get("condition", "N/A")}

Feels Like: {current.get("feels_like", "N/A")} °C

Humidity: {current.get("humidity", "N/A")} %

Rain Probability: {current.get("rain_probability", "N/A")} %

Wind Speed: {current.get("wind_speed", "N/A")}

Wind Direction: {current.get("wind_direction", "N/A")}

Sunrise: {current.get("sunrise", "N/A")}

Sunset: {current.get("sunset", "N/A")}
"""


# ==========================================================
# WEATHER CITY
# ==========================================================

def get_weather_for_question(question: str):

    city = extract_city(question)

    print(
        "Weather location extracted:",
        city,
    )

    if city:

        try:

            return get_live_weather(
                location=city,
            )

        except Exception as e:

            print(
                "Weather lookup by city failed:",
                e,
            )

    return None


# ==========================================================
# GENERATE RESPONSE
# ==========================================================

async def generate_response(
    question: str,
    language: str,
):

    question = (question or "").strip()

    language = (
        language
        if language in LANGUAGE_NAMES
        else "en"
    )

    language_name = LANGUAGE_NAMES[
        language
    ]

    # ------------------------------------------------------
    # EMPTY INPUT
    # ------------------------------------------------------

    if not question:

        empty_messages = {

            "en":
                "I could not hear you clearly. "
                "Please press the microphone and ask your question again.",

            "kn":
                "ನಿಮ್ಮ ಮಾತು ಸ್ಪಷ್ಟವಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. "
                "ಮತ್ತೆ ಮೈಕ್ರೋಫೋನ್ ಒತ್ತಿ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ.",

            "hi":
                "मैं आपकी आवाज़ स्पष्ट रूप से नहीं सुन पाया। "
                "कृपया माइक्रोफोन दबाकर फिर से प्रश्न पूछें।",

            "te":
                "మీ మాట స్పష్టంగా వినిపించలేదు. "
                "మైక్రోఫోన్ నొక్కి మళ్లీ ప్రశ్న అడగండి.",

            "ta":
                "உங்கள் குரல் தெளிவாக கேட்கவில்லை. "
                "மைக்ரோஃபோனை அழுத்தி மீண்டும் கேள்வி கேளுங்கள்.",

            "ml":
                "നിങ്ങളുടെ ശബ്ദം വ്യക്തമായി കേൾക്കാനായില്ല. "
                "മൈക്രോഫോൺ അമർത്തി വീണ്ടും ചോദിക്കൂ.",
        }

        return {
            "question": "",
            "language": language,
            "weather": None,
            "answer": empty_messages[language],
        }

    # ------------------------------------------------------
    # WEATHER
    # ------------------------------------------------------

    weather = None

    if is_weather_question(question):

        weather = get_weather_for_question(
            question
        )

        prompt = f"""
{SYSTEM_PROMPT}

The farmer is asking about WEATHER.

Farmer language:
{language_name}

Farmer question:
{question}

LIVE WEATHER DATA:
{format_weather(weather)}

Instructions:

- Use ONLY the supplied live weather data.
- Do not invent missing weather values.
- Explain the weather in {language_name}.
- Give practical advice about irrigation,
  spraying, fertilizer application and harvesting.
- If the weather data is unavailable, clearly say
  that live weather could not be retrieved.
- Do not pretend that unavailable weather data is real.
- Keep the answer suitable for spoken conversation.
"""

    # ------------------------------------------------------
    # MARKET
    # ------------------------------------------------------

    elif is_price_question(question):

        prompt = f"""
{SYSTEM_PROMPT}

The farmer is asking about MANDI / MARKET PRICES.

Farmer language:
{language_name}

Farmer question:
{question}

There is currently no guaranteed live mandi price
provided directly to you in this request.

Therefore:

- Do NOT invent an exact current market price.
- Explain how the price is determined.
- If the farmer asks for a prediction, provide only
  a cautious trend-based estimate and clearly label it
  as an estimate.
- Explain factors such as arrivals, demand, quality,
  season, location and transportation.
- Ask for crop and location when needed.
- Mention that exact prices should be checked against
  the relevant local mandi data.
- Reply in {language_name}.
"""

    # ------------------------------------------------------
    # FERTILIZER
    # ------------------------------------------------------

    elif is_fertilizer_question(question):

        prompt = f"""
{SYSTEM_PROMPT}

The farmer is asking about FERTILIZER.

Farmer language:
{language_name}

Farmer question:
{question}

Give practical advice.

Include, when relevant:

- suitable fertilizer type
- NPK purpose
- urea / DAP / potash use
- approximate application timing
- soil-test importance
- precautions
- approximate cost range ONLY if you can clearly
  label it as an approximate range

Do not invent an exact current shop price.

Ask for crop, crop age and area if exact dosage
is required.

Reply in {language_name}.
"""

    # ------------------------------------------------------
    # PESTICIDE
    # ------------------------------------------------------

    elif is_pesticide_question(question):

        prompt = f"""
{SYSTEM_PROMPT}

The farmer is asking about PESTICIDES or CROP SPRAYING.

Farmer language:
{language_name}

Farmer question:
{question}

Give practical and safe guidance.

Cover, when relevant:

- likely pest/disease
- prevention
- biological/organic options
- suitable treatment category
- spraying precautions
- cost considerations

Do not invent an exact current product price.

Do not recommend unsafe pesticide combinations.

Tell the farmer to follow the product label and
local agriculture department recommendations.

Reply in {language_name}.
"""

    # ------------------------------------------------------
    # DISEASE
    # ------------------------------------------------------

    elif is_disease_question(question):

        prompt = f"""
{SYSTEM_PROMPT}

The farmer is asking about a CROP DISEASE.

Farmer language:
{language_name}

Farmer question:
{question}

Explain:

1. likely cause
2. visible symptoms
3. immediate action
4. prevention
5. organic options
6. chemical treatment category when appropriate

Do not claim a diagnosis with certainty from text alone.

If a photo would help, tell the farmer to upload
a clear crop/leaf photo.

Reply in {language_name}.
"""

    # ------------------------------------------------------
    # IRRIGATION
    # ------------------------------------------------------

    elif is_irrigation_question(question):

        prompt = f"""
{SYSTEM_PROMPT}

The farmer is asking about IRRIGATION.

Farmer language:
{language_name}

Farmer question:
{question}

Give practical advice about:

- watering frequency
- soil moisture
- crop stage
- drip irrigation
- rainfall considerations
- avoiding overwatering

If exact scheduling requires crop, soil and weather,
say what information is needed.

Reply in {language_name}.
"""

    # ------------------------------------------------------
    # GOVERNMENT SCHEMES
    # ------------------------------------------------------

    elif is_scheme_question(question):

        prompt = f"""
{SYSTEM_PROMPT}

The farmer is asking about GOVERNMENT AGRICULTURE
SCHEMES or SUBSIDIES.

Farmer language:
{language_name}

Farmer question:
{question}

Explain the scheme simply.

Include:

- purpose
- who may qualify
- major benefit
- typical application process
- documents normally required

Do not invent eligibility or current payment amounts.

If exact current rules are required, tell the farmer
that official government information should be checked.

Reply in {language_name}.
"""

    # ------------------------------------------------------
    # GENERAL AGRICULTURE
    # ------------------------------------------------------

    else:

        prompt = f"""
{SYSTEM_PROMPT}

Farmer language:
{language_name}

Farmer question:
{question}

Answer the farmer directly.

If the question concerns:

- crop disease → symptoms and treatment
- fertilizer → suitable fertilizer and timing
- pesticide → safe treatment options
- irrigation → watering guidance
- market → price/trend explanation
- weather → weather guidance
- government schemes → scheme explanation
- farming cost → explain major cost components
- crop grading → quality and selling advice

If important information is missing, ask a short
follow-up question.

Reply ONLY in {language_name}.

Keep the answer concise and useful for voice.
"""

    # ------------------------------------------------------
    # GEMINI
    # ------------------------------------------------------

    try:

        print("=" * 60)
        print("Calling Gemini...")
        print("Model:", MODEL_NAME)
        print("Language:", language_name)
        print("Question:", question)
        print("=" * 60)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        answer = (
            response.text or ""
        ).strip()

        if not answer:

            raise RuntimeError(
                "Gemini returned an empty response."
            )

    except Exception as e:

        print(
            "\n========== GEMINI ERROR =========="
        )

        print(type(e))
        print(e)

        # --------------------------------------------------
        # WEATHER FALLBACK
        # --------------------------------------------------

        if weather:

            current = weather.get(
                "current",
                {},
            )

            location = current.get(
                "location",
                "your location",
            )

            temperature = current.get(
                "temperature",
                "N/A",
            )

            condition = current.get(
                "condition",
                "N/A",
            )

            humidity = current.get(
                "humidity",
                "N/A",
            )

            rain_probability = current.get(
                "rain_probability",
                "N/A",
            )

            if language == "kn":

                answer = (
                    f"{location} ಇಂದಿನ ಹವಾಮಾನ: "
                    f"ತಾಪಮಾನ {temperature}°C, "
                    f"{condition}. "
                    f"ಆರ್ದ್ರತೆ {humidity}% ಮತ್ತು "
                    f"ಮಳೆಯ ಸಾಧ್ಯತೆ {rain_probability}%. "
                    f"ಮಳೆಯ ಸಾಧ್ಯತೆ ಇದ್ದರೆ ನೀರಾವರಿ ಮತ್ತು "
                    f"ಸಿಂಪಡಿಸುವಿಕೆಯನ್ನು ಎಚ್ಚರಿಕೆಯಿಂದ ನಿರ್ವಹಿಸಿ."
                )

            elif language == "hi":

                answer = (
                    f"{location} का आज का मौसम: "
                    f"तापमान {temperature}°C और "
                    f"{condition}। "
                    f"नमी {humidity}% है और बारिश की "
                    f"संभावना {rain_probability}% है। "
                    f"बारिश की संभावना होने पर सिंचाई और "
                    f"छिड़काव सावधानी से करें।"
                )

            elif language == "te":

                answer = (
                    f"{location} నేటి వాతావరణం: "
                    f"ఉష్ణోగ్రత {temperature}°C, "
                    f"{condition}. "
                    f"తేమ {humidity}% మరియు వర్షం "
                    f"సంభావ్యత {rain_probability}%. "
                    f"వర్షం ఉంటే నీటిపారుదల మరియు "
                    f"పిచికారీని జాగ్రత్తగా చేయండి."
                )

            elif language == "ta":

                answer = (
                    f"{location} இன்றைய வானிலை: "
                    f"வெப்பநிலை {temperature}°C, "
                    f"{condition}. "
                    f"ஈரப்பதம் {humidity}% மற்றும் "
                    f"மழை வாய்ப்பு {rain_probability}%. "
                    f"மழை இருந்தால் நீர்ப்பாசனம் மற்றும் "
                    f"தெளிப்பை கவனமாக செய்யுங்கள்."
                )

            elif language == "ml":

                answer = (
                    f"{location} ഇന്നത്തെ കാലാവസ്ഥ: "
                    f"താപനില {temperature}°C, "
                    f"{condition}. "
                    f"ഈർപ്പം {humidity}% കൂടാതെ മഴയുടെ "
                    f"സാധ്യത {rain_probability}%. "
                    f"മഴയുണ്ടെങ്കിൽ ജലസേചനവും സ്പ്രേയിംഗും "
                    f"ശ്രദ്ധയോടെ നടത്തുക."
                )

            else:

                answer = (
                    f"Today's weather in {location}: "
                    f"{temperature}°C and {condition}. "
                    f"Humidity is {humidity}% and the "
                    f"rain probability is {rain_probability}%. "
                    f"If rain is expected, manage irrigation "
                    f"and spraying carefully."
                )

        else:

            fallback_messages = {

                "en":
                    "I am temporarily unable to process "
                    "that request. Please try again.",

                "kn":
                    "ಈ ಪ್ರಶ್ನೆಯನ್ನು ಈಗ ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಲು "
                    "ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",

                "hi":
                    "मैं अभी इस प्रश्न को संसाधित नहीं कर पा रहा हूँ। "
                    "कृपया फिर से प्रयास करें।",

                "te":
                    "ఈ ప్రశ్నను ప్రస్తుతం ప్రాసెస్ చేయలేకపోతున్నాను. "
                    "దయచేసి మళ్లీ ప్రయత్నించండి.",

                "ta":
                    "இந்த கேள்வியை தற்போது செயல்படுத்த முடியவில்லை. "
                    "மீண்டும் முயற்சிக்கவும்.",

                "ml":
                    "ഈ ചോദ്യം ഇപ്പോൾ പ്രോസസ്സ് ചെയ്യാൻ കഴിയുന്നില്ല. "
                    "ദയവായി വീണ്ടും ശ്രമിക്കുക.",
            }

            answer = fallback_messages[
                language
            ]

    # ------------------------------------------------------
    # RETURN
    # ------------------------------------------------------

    return {
        "question": question,
        "language": language,
        "weather": weather,
        "answer": answer,
    }


print("############################")
print("NEW ASSISTANT SERVICE LOADED")
print("############################")