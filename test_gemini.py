from google import genai
from config import get_settings

settings = get_settings()

print("Testing Gemini...")
print("Key loaded:", bool(settings.gemini_api_key))

try:
    client = genai.Client(
        api_key=settings.gemini_api_key
    )

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents="Say hello in one short sentence."
    )

    print("\n========== SUCCESS ==========")
    print(response.text)

except Exception as e:
    print("\n========== FAILED ==========")
    print(type(e))
    print(e)