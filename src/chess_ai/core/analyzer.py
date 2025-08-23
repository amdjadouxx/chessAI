"""
Module d'analyse avancée pour Chess AI.

Ce module fournit des outils d'analyse sophistiqués
pour l'évaluation de positions d'échecs.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
import chess
from chess import Board, Square, Piece, Color, Move, PieceType

from ..exceptions import ChessBoardStateError, InvalidSquareError
from ..utils.validation import validate_square, validate_color, validate_piece_type
from ..utils.logging_config import get_logger


class ChessAnalyzer:
    """
    Analyseur avancé pour positions d'échecs.

    Cette classe fournit des méthodes d'analyse sophistiquées
    pour évaluer et comprendre les positions d'échecs.
    """

    def __init__(self, board: Board, enable_logging: bool = True):
        """
        Initialise l'analyseur.

        Args:
            board: Plateau d'échecs à analyser
            enable_logging: Active le logging
        """
        self.board = board
        self.logger = get_logger(__name__) if enable_logging else None

    def get_piece_map(self) -> Dict[Square, Piece]:
        """
        Retourne la carte complète des pièces sur le plateau.

        Returns:
            Dictionnaire mapping les cases aux pièces
        """
        try:
            return self.board.piece_map()
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Erreur lors de l'obtention de la carte des pièces: {e}"
                )
            raise ChessBoardStateError("Impossible d'obtenir la carte des pièces")

    def get_pieces_by_color(self, color: Color) -> List[Tuple[str, Piece]]:
        """
        Retourne toutes les pièces d'une couleur avec leurs positions.

        Args:
            color: Couleur des pièces à retourner

        Returns:
            Liste de tuples (case_name, piece)
        """
        try:
            validated_color = validate_color(color)
            pieces = []

            piece_map = self.get_piece_map()
            for square, piece in piece_map.items():
                if piece.color == validated_color:
                    square_name = chess.square_name(square)
                    pieces.append((square_name, piece))

            if self.logger:
                color_name = "blanches" if validated_color == chess.WHITE else "noires"
                self.logger.debug(f"Trouvé {len(pieces)} pièces {color_name}")

            return pieces

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Erreur lors de l'obtention des pièces par couleur: {e}"
                )
            raise ChessBoardStateError("Impossible d'obtenir les pièces par couleur")

    def count_material(self) -> Dict[str, Dict[str, int]]:
        """
        Compte le matériel pour chaque camp.

        Returns:
            Dictionnaire avec le décompte par couleur et type de pièce
        """
        try:
            material = {
                "white": {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0, "K": 0},
                "black": {"p": 0, "n": 0, "b": 0, "r": 0, "q": 0, "k": 0},
            }

            piece_types = [
                chess.PAWN,
                chess.KNIGHT,
                chess.BISHOP,
                chess.ROOK,
                chess.QUEEN,
                chess.KING,
            ]
            piece_symbols_white = ["P", "N", "B", "R", "Q", "K"]
            piece_symbols_black = ["p", "n", "b", "r", "q", "k"]

            for piece_type, white_symbol, black_symbol in zip(
                piece_types, piece_symbols_white, piece_symbols_black
            ):
                white_count = len(self.board.pieces(piece_type, chess.WHITE))
                black_count = len(self.board.pieces(piece_type, chess.BLACK))

                material["white"][white_symbol] = white_count
                material["black"][black_symbol] = black_count

            if self.logger:
                total_white = sum(material["white"].values())
                total_black = sum(material["black"].values())
                self.logger.debug(
                    f"Matériel compté: {total_white} blancs, {total_black} noirs"
                )

            return material

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors du comptage de matériel: {e}")
            raise ChessBoardStateError("Impossible de compter le matériel")

    def get_attackers(self, square: Square, by_color: Color) -> Set[Square]:
        """
        Retourne toutes les cases contenant des pièces qui attaquent une case donnée.

        Args:
            square: Case cible
            by_color: Couleur des attaquants

        Returns:
            Ensemble des cases d'attaquants
        """
        try:
            validated_square = validate_square(square)
            validated_color = validate_color(by_color)

            attackers = self.board.attackers(validated_color, validated_square)

            if self.logger:
                square_name = chess.square_name(validated_square)
                color_name = "blanches" if validated_color == chess.WHITE else "noires"
                self.logger.debug(
                    f"{len(attackers)} pièces {color_name} attaquent {square_name}"
                )

            return attackers

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'obtention des attaquants: {e}")
            raise ChessBoardStateError("Impossible d'obtenir les attaquants")

    def is_square_attacked(self, square: Square, by_color: Color) -> bool:
        """
        Vérifie si une case est attaquée par une couleur.

        Args:
            square: Case à vérifier
            by_color: Couleur de l'attaquant

        Returns:
            True si la case est attaquée
        """
        try:
            validated_square = validate_square(square)
            validated_color = validate_color(by_color)

            return self.board.is_attacked_by(validated_color, validated_square)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de la vérification d'attaque: {e}")
            return False

    def get_piece_mobility(self, square: Square) -> int:
        """
        Calcule la mobilité d'une pièce (nombre de mouvements possibles).

        Args:
            square: Case de la pièce

        Returns:
            Nombre de mouvements légaux pour cette pièce
        """
        try:
            validated_square = validate_square(square)

            mobility = 0
            for move in self.board.legal_moves:
                if move.from_square == validated_square:
                    mobility += 1

            if self.logger:
                square_name = chess.square_name(validated_square)
                self.logger.debug(f"Mobilité de la pièce en {square_name}: {mobility}")

            return mobility

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors du calcul de mobilité: {e}")
            return 0

    def get_king_safety_score(self, color: Color) -> Dict[str, Any]:
        """
        Évalue la sécurité du roi d'une couleur donnée.

        Args:
            color: Couleur du roi à évaluer

        Returns:
            Dictionnaire avec des métriques de sécurité
        """
        try:
            validated_color = validate_color(color)
            king_square = self.board.king(validated_color)

            if king_square is None:
                return {"error": "Roi non trouvé"}

            # Cases autour du roi
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)

            surrounding_squares = []
            for file_offset in [-1, 0, 1]:
                for rank_offset in [-1, 0, 1]:
                    if file_offset == 0 and rank_offset == 0:
                        continue

                    new_file = king_file + file_offset
                    new_rank = king_rank + rank_offset

                    if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                        square = chess.square(new_file, new_rank)
                        surrounding_squares.append(square)

            # Analyser les menaces
            enemy_color = not validated_color
            threats = 0
            protected_squares = 0

            for square in surrounding_squares:
                if self.is_square_attacked(square, enemy_color):
                    threats += 1
                if self.is_square_attacked(square, validated_color):
                    protected_squares += 1

            safety_score = {
                "king_square": chess.square_name(king_square),
                "threats_around_king": threats,
                "protected_squares": protected_squares,
                "is_in_check": self.board.is_check(),
                "can_castle_kingside": self.board.has_kingside_castling_rights(
                    validated_color
                ),
                "can_castle_queenside": self.board.has_queenside_castling_rights(
                    validated_color
                ),
                "safety_rating": max(0, 10 - threats + protected_squares),
            }

            if self.logger:
                color_name = "blanc" if validated_color == chess.WHITE else "noir"
                self.logger.debug(
                    f"Sécurité du roi {color_name}: {safety_score['safety_rating']}/10"
                )

            return safety_score

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Erreur lors de l'évaluation de sécurité du roi: {e}"
                )
            return {"error": str(e)}

    def get_piece_development_score(self, color: Color) -> Dict[str, Any]:
        """
        Évalue le développement des pièces pour une couleur.

        Args:
            color: Couleur à évaluer

        Returns:
            Score de développement et détails
        """
        try:
            validated_color = validate_color(color)

            # Cases de départ pour les pièces
            if validated_color == chess.WHITE:
                knight_start_squares = [chess.B1, chess.G1]
                bishop_start_squares = [chess.C1, chess.F1]
                back_rank = 0
            else:
                knight_start_squares = [chess.B8, chess.G8]
                bishop_start_squares = [chess.C8, chess.F8]
                back_rank = 7

            developed_knights = 0
            developed_bishops = 0

            # Vérifier les cavaliers
            for square in knight_start_squares:
                piece = self.board.piece_at(square)
                if piece is None or piece.piece_type != chess.KNIGHT:
                    developed_knights += 1

            # Vérifier les fous
            for square in bishop_start_squares:
                piece = self.board.piece_at(square)
                if piece is None or piece.piece_type != chess.BISHOP:
                    developed_bishops += 1

            development_score = {
                "developed_knights": developed_knights,
                "total_knights": 2,
                "developed_bishops": developed_bishops,
                "total_bishops": 2,
                "development_percentage": (developed_knights + developed_bishops)
                / 4
                * 100,
            }

            if self.logger:
                color_name = "blancs" if validated_color == chess.WHITE else "noirs"
                self.logger.debug(
                    f"Développement {color_name}: {development_score['development_percentage']:.1f}%"
                )

            return development_score

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'évaluation du développement: {e}")
            return {"error": str(e)}

    def analyze_position(self) -> Dict[str, Any]:
        """
        Analyse complète de la position actuelle.

        Returns:
            Analyse détaillée de la position
        """
        try:
            analysis = {
                "material": self.count_material(),
                "white_king_safety": self.get_king_safety_score(chess.WHITE),
                "black_king_safety": self.get_king_safety_score(chess.BLACK),
                "white_development": self.get_piece_development_score(chess.WHITE),
                "black_development": self.get_piece_development_score(chess.BLACK),
                "position_info": {
                    "fen": self.board.fen(),
                    "turn": "White" if self.board.turn == chess.WHITE else "Black",
                    "is_check": self.board.is_check(),
                    "is_checkmate": self.board.is_checkmate(),
                    "is_stalemate": self.board.is_stalemate(),
                    "legal_moves_count": len(list(self.board.legal_moves)),
                },
            }

            if self.logger:
                self.logger.info("Analyse de position complétée")

            return analysis

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'analyse de position: {e}")
            raise ChessBoardStateError("Impossible d'analyser la position")
