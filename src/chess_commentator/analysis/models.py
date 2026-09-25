"""Domain models for chess analysis, evaluation, and commentary."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class BoardFeatures:
    """Positional, mobility, and pawn structure features extracted from a board."""
    phase: str
    castled: bool
    open_files_near_king: int
    doubled_pawns: int
    isolated_pawns: int
    passed_pawns: int
    mobility: int
    time_pressure: bool
    pawns: int
    knights: int
    bishops: int
    rooks: int
    queens: int
    opp_pawns: int
    opp_knights: int
    opp_bishops: int
    opp_rooks: int
    opp_queens: int
    material_balance: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaxonomyResult:
    """Classification of tactical mistakes or themes."""
    mistake_category: str
    tactic_type: str
    piece_lost: str = "none"
    mate_missed: int = 0
    tactical_elements: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MoveMetadata:
    """Metadata regarding a played move in a game."""
    game_id: str
    move_number: int
    player_color: str
    move_san: str
    move_uci: str
    time_left: Optional[int] = None
    result: str = "unknown"
    eco: str = "?"
    opening: str = "?"
    white_player: str = ""
    black_player: str = ""
    white_elo: Optional[str] = None
    black_elo: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PositionAnalysis:
    """Complete analysis record for a chess move grounded in Stockfish and taxonomy."""
    fen: str
    move_uci: str
    move_san: str
    player_color: str
    move_number: int
    eval_before: Optional[int]
    eval_after: Optional[int]
    cp_loss: Optional[int]
    best_move: Optional[str]
    played_best: bool
    mistake_type: str
    taxonomy: TaxonomyResult
    features: BoardFeatures
    metadata: Optional[MoveMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Flatten taxonomy and features for easy tabular Parquet export
        data.update(self.taxonomy.to_dict())
        data.update(self.features.to_dict())
        if self.metadata:
            data.update(self.metadata.to_dict())
        return data


@dataclass(frozen=True)
class CommentaryResult:
    """Final output from live commentary generation."""
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
    grounded_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
