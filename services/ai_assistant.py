"""
AgriNova AI Assistant
Uses OpenAI to answer agriculture questions.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are AgriNova AI.

You are an expert agricultural assistant.

You help farmers with:

- Crop diseases
- Fertilizers
- Tomato grading
- Weather advice
- Irrigation
- Pest control
- Market prices
- Harvesting
- Storage
- Organic farming

Rules:

1. Give practical farmer-friendly answers.
2. Keep answers under 120 words.
3. Never answer unrelated topics.
4. If unsure, recommend consulting the local agriculture officer.
"""


async def ask_ai(question: str) -> str:
    """
    Ask AgriNova AI.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.4,
        max_tokens=300,
    )

    return response.choices[0].message.content