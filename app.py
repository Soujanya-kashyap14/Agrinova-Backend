"""
EcoAgri Intelligence — FastAPI Application

Main application entry point.

This version:
- Loads configuration safely
- Connects MongoDB
- Loads AI models during startup
- Registers all API routers
- Creates required directories
- Serves generated audio
- Provides health/root endpoints
- Gives useful startup diagnostics
"""

from __future__ import annotations

import sys

# ------------------------------------------------------------------
# Windows consoles default stdout/stderr to a legacy codepage (e.g.
# cp1252), which cannot encode Kannada/Telugu/Tamil/Malayalam/Hindi
# script characters. Any print() of recognized non-English speech
# (see services/speech_service.py) then raises UnicodeEncodeError,
# which is uncaught and turns into a 500 for the whole voice
# request — the farmer's speech was transcribed correctly, but the
# request crashes before ever reaching Gemini. Force UTF-8 output
# as early as possible so debug logging never crashes a request.
# ------------------------------------------------------------------

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ------------------------------------------------------------------
# Every external API this app calls (Gemini, Groq, Open-Meteo,
# OpenWeatherMap, the geocoder, Google Translate for TTS) publishes
# an IPv6 address alongside its IPv4 ones. On this machine/network,
# IPv6 routes are effectively dead - not refused, just silently
# unreachable - so any connection attempt on them hangs until a
# ~20-25s OS-level timeout before urllib3 (which `requests` and the
# Gemini SDK use under the hood) falls back to IPv4, which then
# succeeds instantly. Measured directly: a single Groq transcription
# call dropped from ~23s to ~2s once IPv6 was taken out of the
# picture. Since nearly every voice/weather/chat request makes
# multiple such calls, this one setting was very likely the single
# biggest contributor to "everything is slow" across the whole app.
# Forcing IPv4-only resolution here (before anything else imports
# requests/urllib3) applies to every outbound call in the process.
# ------------------------------------------------------------------

try:
    import socket

    # Patching urllib3 alone isn't enough: the Gemini SDK (google-genai)
    # goes through httpx, not requests/urllib3. Patching the stdlib
    # socket.getaddrinfo directly is the one point both libraries (and
    # anything else built on Python sockets) ultimately resolve
    # through, so this single patch covers Groq, weather/geocoding
    # (requests) and Gemini (httpx) alike.
    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return _original_getaddrinfo(
            host, port, socket.AF_INET, type, proto, flags
        )

    socket.getaddrinfo = _ipv4_only_getaddrinfo
except Exception:
    pass

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings

from database.mongodb import (
    connect_to_mongodb,
    close_mongodb_connection,
)

# IMPORTANT:
# Import the AI service only here.
#
# If ai_service.py does not contain load_ai_models(),
# the application will report the exact problem during startup.
try:
    from services.ai_service import load_ai_models
except ImportError as exc:

    print()
    print("=" * 70)
    print("[AI IMPORT ERROR]")
    print("=" * 70)
    print(str(exc))
    print()
    print(
        "The application could not import load_ai_models "
        "from services.ai_service."
    )
    print(
        "Check backend/services/ai_service.py."
    )
    print("=" * 70)
    print()

    load_ai_models = None


# ============================================================
# SETTINGS
# ============================================================

settings = get_settings()


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GENERATED_AUDIO_DIR = (
    BASE_DIR / "generated_audio"
)

UPLOAD_AUDIO_DIR = (
    BASE_DIR / "uploads" / "audio"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

Path(
    settings.upload_dir
).mkdir(
    parents=True,
    exist_ok=True,
)

GENERATED_AUDIO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

UPLOAD_AUDIO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print()
    print("=" * 70)
    print("AGRIWISE INTELLIGENCE")
    print("=" * 70)

    print()
    print("[SYSTEM] Python:")
    print(sys.version)

    print()
    print("[SYSTEM] Backend directory:")
    print(BASE_DIR)

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("[DATABASE]")
    print("-" * 70)

    try:

        await connect_to_mongodb()

        print(
            "[DATABASE] MongoDB connected successfully."
        )

    except Exception as exc:

        print(
            "[DATABASE] MongoDB connection failed."
        )

        print(
            f"[DATABASE] {exc}"
        )

        print(
            "[DATABASE] Continuing without database."
        )

    # --------------------------------------------------------
    # AI MODELS
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("[AI MODELS]")
    print("-" * 70)

    if load_ai_models is None:

        print(
            "[AI] load_ai_models() is unavailable."
        )

        print(
            "[AI] Check services/ai_service.py."
        )

    else:

        try:

            print(
                "[AI] Loading AgriWise AI models..."
            )

            load_ai_models()

            print(
                "[AI] AI models loaded successfully."
            )

        except Exception as exc:

            print()
            print(
                "[AI] MODEL LOADING ERROR"
            )
            print(
                f"[AI] {type(exc).__name__}: {exc}"
            )
            print()
            print(
                "[AI] FastAPI will continue running,"
            )
            print(
                "[AI] but AI prediction endpoints may fail."
            )

    # --------------------------------------------------------
    # STARTUP COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("APPLICATION STARTUP COMPLETE")
    print("=" * 70)

    print(
        "[API] Root    : /"
    )

    print(
        "[API] Health  : /health"
    )

    print(
        "[API] Docs    : /docs"
    )

    print(
        "[API] API     : /api"
    )

    print(
        "[MANDI] Dynamic market discovery enabled."
    )

    print(
        "[MANDI] No hardcoded city list."
    )

    print("=" * 70)
    print()

    yield

    # ========================================================
    # SHUTDOWN
    # ========================================================

    print()
    print("=" * 70)
    print("SHUTTING DOWN AGRIWISE INTELLIGENCE")
    print("=" * 70)

    try:

        await close_mongodb_connection()

        print(
            "[DATABASE] MongoDB connection closed."
        )

    except Exception as exc:

        print(
            "[DATABASE] MongoDB shutdown error:"
        )

        print(
            f"[DATABASE] {exc}"
        )

    print(
        "[SYSTEM] Shutdown complete."
    )

    print("=" * 70)
    print()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

configured_origins = list(
    settings.cors_origins_list
)

local_development_origins = [

    "http://localhost:8080",

    "http://127.0.0.1:8080",

    "http://localhost:5173",

    "http://127.0.0.1:5173",

]

for origin in local_development_origins:

    if origin not in configured_origins:

        configured_origins.append(
            origin
        )


app.add_middleware(
    CORSMiddleware,

    allow_origins=(
        configured_origins
    ),

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# STATIC GENERATED AUDIO
# ============================================================

app.mount(
    "/generated_audio",

    StaticFiles(
        directory=str(
            GENERATED_AUDIO_DIR
        )
    ),

    name="generated_audio",
)


# ============================================================
# API PREFIX
# ============================================================

API_PREFIX = "/api"


# ============================================================
# ROUTES
# ============================================================

#
# IMPORTANT:
#
# We import the route modules individually instead of using
# one large:
#
#     from routes import (...)
#
# This makes startup errors much easier to identify.
#

print()
print("=" * 70)
print("LOADING API ROUTES")
print("=" * 70)


# ------------------------------------------------------------
# AUTH
# ------------------------------------------------------------

try:

    from routes import auth

    app.include_router(
        auth.router,
        prefix=API_PREFIX,
    )

    print(
        "[ROUTE] auth              OK"
    )

except Exception as exc:

    print(
        "[ROUTE] auth              FAILED"
    )

    print(
        f"        {type(exc).__name__}: {exc}"
    )


# ------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------

try:

    from routes import upload

    app.include_router(
        upload.router,
        prefix=API_PREFIX,
    )

    print(
        "[ROUTE] upload            OK"
    )

except Exception as exc:

    print(
        "[ROUTE] upload            FAILED"
    )

    print(
        f"        {type(exc).__name__}: {exc}"
    )


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

try:

    from routes import predict

    app.include_router(
        predict.router,
        prefix=API_PREFIX,
    )

    print(
        "[ROUTE] predict           OK"
    )

except Exception as exc:

    print(
        "[ROUTE] predict           FAILED"
    )

    print(
        f"        {type(exc).__name__}: {exc}"
    )


# ------------------------------------------------------------
# WEATHER
# ------------------------------------------------------------

try:

    from routes import weather

    app.include_router(
        weather.router,
        prefix=API_PREFIX,
    )

    print(
        "[ROUTE] weather           OK"
    )

except Exception as exc:

    print(
        "[ROUTE] weather           FAILED"
    )

    print(
        f"        {type(exc).__name__}: {exc}"
    )


# ------------------------------------------------------------
# MANDI
# ------------------------------------------------------------

try:

    from routes import mandi

    app.include_router(
        mandi.router,
        prefix=API_PREFIX,
    )

    print(
        "[ROUTE] mandi             OK"
    )

except Exception as exc:

    print(
        "[ROUTE] mandi             FAILED"
    )

    print(
        f"        {type(exc).__name__}: {exc}"
    )


# ------------------------------------------------------------
# PRICE
# ------------------------------------------------------------

try:

    from routes import price

    app.include_router(
        price.router,
        prefix=API_PREFIX,
    )

    print(
        "[ROUTE] price             OK"
    )

except Exception as exc:

    print(
        "[ROUTE] price             FAILED"
    )

    print(
        f"        {type(exc).__name__}: {exc}"
    )


# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------

try:

    from routes import history

    app.include_router(
        history.router,
        prefix=API_PREFIX,
    )

    print(
        "[ROUTE] history           OK"
    )

except Exception as exc:

    print(
        "[ROUTE] history           FAILED"
    )

    print(
        f"        {type(exc).__name__}: {exc}"
    )

    print()
    print(
        "[HISTORY] This usually means that "
        "models/prediction.py does not contain "
        "ScanHistoryItem or ScanHistoryResponse."
    )

    print(
        "[HISTORY] Make sure the prediction.py "
        "you saved is the same file shown in your editor."
    )


# ------------------------------------------------------------
# VOICE
# ------------------------------------------------------------

try:

    from routes import voice

    app.include_router(
        voice.router,
        prefix=API_PREFIX,
    )

    print(
        "[ROUTE] voice             OK"
    )

except Exception as exc:

    print(
        "[ROUTE] voice             FAILED"
    )

    print(
        f"        {type(exc).__name__}: {exc}"
    )


print("=" * 70)
print()


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():

    return {

        "status": "healthy",

        "message":
            "AgriWise Intelligence API",

        "version":
            settings.app_version,

        "docs":
            "/docs",

        "api":
            "/api",

        "mandi_endpoint":
            "/api/mandis/nearby",

    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get(
    "/health"
)
async def health():

    return {

        "status": "ok",

        "service":
            "AgriWise Intelligence",

        "mandi":
            "dynamic",

        "ai_models":
            (
                "available"
                if load_ai_models is not None
                else "unavailable"
            ),

    }


# ============================================================
# AI STATUS ENDPOINT
# ============================================================

@app.get(
    "/api/ai-status"
)
async def ai_status():

    """
    Simple diagnostic endpoint.

    Open:

        http://127.0.0.1:8000/api/ai-status

    to verify that the AI loading function exists.
    """

    return {

        "load_ai_models_available":
            load_ai_models is not None,

        "status":
            (
                "ready"
                if load_ai_models is not None
                else "missing_load_ai_models"
            ),

    }


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(
    Exception
)
async def exception_handler(
    request: Request,
    exc: Exception,
):

    print()
    print("=" * 70)
    print("[API ERROR]")
    print("=" * 70)

    print(
        "Method:",
        request.method,
    )

    print(
        "URL:",
        request.url,
    )

    print(
        "Exception:",
        type(exc).__name__,
    )

    print(
        "Error:",
        str(exc),
    )

    print("=" * 70)
    print()

    return JSONResponse(

        status_code=500,

        content={

            "detail":
                str(exc),

            "error_type":
                type(exc).__name__,

        },

    )


# ============================================================
# APPLICATION READY
# ============================================================

print()
print("=" * 70)
print("AGRIWISE APP MODULE LOADED")
print("=" * 70)
print()