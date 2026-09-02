"""
Gemini Intent Detection Service

Uses Gemini to understand:
- User intent
- City
- Crop
- Language

Returns structured JSON.
"""

import json

import google.generativeai as genai

from config import get_settings

settings = get_settings()

genai.configure(api_key=settings.gemini_api_key)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

INTENT_PROMPT = """
You are an AI Intent Classifier for an Agriculture Assistant.

Your job is ONLY to analyse the farmer's question.

Return ONLY valid JSON.

Supported intents:

weather
market_price
price_prediction
disease
fertilizer
irrigation
crop_recommendation
general

Extract:

intent
city
crop
language

Rules:

- Detect the language used by the farmer.
- Extract city if mentioned.
- Extract crop if mentioned.
- If missing, return "".

Example:

Question:
What is today's tomato price in Bangalore?

Output:

{
 "intent":"market_price",
 "city":"Bangalore",
 "crop":"Tomato",
 "language":"en"
}

Question:
ಇವತ್ತು ದಾವಣಗೆರೆಯ ಹವಾಮಾನ ಹೇಗಿದೆ?

Output:

{
 "intent":"weather",
 "city":"Davangere",
 "crop":"",
 "language":"kn"
}

Question:
आज दिल्ली का मौसम कैसा है?

Output:

{
 "intent":"weather",
 "city":"Delhi",
 "crop":"",
 "language":"hi"
}

Return ONLY JSON.
"""


def detect_intent(question: str) -> dict:
    """
    Returns:
    {
        intent,
        city,
        crop,
        language
    }
    """

    response = model.generate_content(
        f"{INTENT_PROMPT}\n\nQuestion:\n{question}"
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        return {
            "intent": "general",
            "city": "",
            "crop": "",
            "language": "en",
        }