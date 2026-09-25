"""FastAPI REST service exposing live chess commentary."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    has_fastapi = True
except ImportError:
    has_fastapi = False
    FastAPI = None

from chess_commentator.serving.service import CommentaryService


class CommentaryRequest(BaseModel):
    fen: str = Field(..., description="FEN position before the move")
    move: str = Field(..., description="Played move in UCI or SAN notation")
    depth: Optional[int] = Field(18, description="Stockfish search depth")
    mode: Optional[str] = Field("deep", description="Analysis mode: 'deep' or 'fast'")


class CommentaryResponse(BaseModel):
    fen: str
    move: str
    commentary: str
    taxonomy: str
    cp_loss: Optional[int]
    eval_before: Optional[int]
    eval_after: Optional[int]
    best_move: Optional[str]
    is_blunder: bool
    phase: str
    grounded_features: Dict[str, Any]


def create_app(service: Optional[CommentaryService] = None) -> Any:
    """Factory creating configured FastAPI application."""
    if not has_fastapi:
        raise ImportError("FastAPI is not installed. Install via `pip install fastapi uvicorn`.")

    app = FastAPI(
        title="AI Chess Commentator API",
        description="REST service generating natural-language chess commentary grounded in Stockfish and taxonomy.",
        version="0.1.0",
    )
    commentary_service = service or CommentaryService()

    @app.get("/health")
    def health_check():
        return {"status": "ok", "service": "chess-commentator"}

    @app.post("/v1/commentary", response_model=CommentaryResponse)
    def comment_endpoint(payload: CommentaryRequest):
        try:
            result = commentary_service.comment(
                fen=payload.fen,
                move=payload.move,
                depth=payload.depth,
                mode=payload.mode,
            )
            return CommentaryResponse(**result.to_dict())
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return app
