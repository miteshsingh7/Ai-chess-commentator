"""Local development server: FastAPI commentary API + static frontend at root.

Usage:
    python scripts/serve_app.py

Opens:
    http://localhost:8001        -> code.html (interactive chess app)
    http://localhost:8001/health -> API health check
    http://localhost:8001/docs   -> FastAPI Swagger UI

CORS is intentionally open (allow_origins=["*"]) for local development only.
Do not deploy this configuration to production.
"""

import sys
import pathlib

# Ensure the package is importable when run from the project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import FastAPI

from chess_commentator.serving.api import create_app
from chess_commentator.serving.engine import CommentaryInferenceEngine
from chess_commentator.serving.service import CommentaryService

# HuggingFace Hub adapter — v10 fine-tuned weights
V10_ADAPTER_ID = "miteshsingh7/phi35-chess-adapter-v10"


def build_server_app() -> FastAPI:
    """Build the combined FastAPI app with CORS and static HTML serving."""
    inference_engine = CommentaryInferenceEngine()
    if inference_engine._groq_client is not None:
        print("✓ Groq LLM API active (qwen/qwen3.8-27b) — sub-second live Grandmaster commentary.")
    else:
        print("✓ Dynamic position-grounded commentary engine active.")

    service = CommentaryService(inference_engine=inference_engine)
    app: FastAPI = create_app(service=service)

    # CORS: allow the frontend (file:// or localhost on any port) to POST
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    frontend_path = PROJECT_ROOT / "code.html"

    @app.get("/", include_in_schema=False)
    def serve_frontend() -> FileResponse:
        if not frontend_path.exists():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"code.html not found at {frontend_path}. Run the update script first.",
            )
        return FileResponse(str(frontend_path), media_type="text/html")

    return app


def main() -> None:
    print("=" * 60)
    print("AI Chess Commentator - Local Development Server")
    print("=" * 60)
    print("  Frontend : http://localhost:8001/")
    print("  API docs : http://localhost:8001/docs")
    print("  Health   : http://localhost:8001/health")
    print("=" * 60)

    server_app = build_server_app()
    uvicorn.run(server_app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
