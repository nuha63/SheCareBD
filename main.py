import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.ai_routes import router as ai_router

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="SheCare BD Backend API")

# ─── CORS ────────────────────────────────────────────────────────────────────
# Flutter native Android apps do not enforce browser-style CORS restrictions.
# allow_origins=["*"] is safe for a mobile-only backend.
# If you ever add a web client, restrict origins here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ─── Startup validation ───────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables. AI features will use fallback responses.")

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(ai_router, prefix="/api/ai", tags=["AI"])

# ─── Health endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Simple health check — used by Render and uptime monitors."""
    return {"status": "ok"}

@app.get("/api/health")
def api_health():
    """Backwards-compatible health check for Flutter app."""
    return {"status": "ok", "message": "SheCare BD Backend is running securely"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
