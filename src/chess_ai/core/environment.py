"""
Module principal de l'environnement d'échecs.

Ce module contient la classe ChessEnvironment qui encapsule
la logique principale de gestion du plateau d'échecs.
"""

import logging
from typing import Optional, List, Union, Dict, Any
import chess
from chess import Board, Move, Square, Piece, Color

from ..exceptions import (
    ChessError,
    InvalidMoveError,
    InvalidSquareError,
    GameOverError,
    InvalidFENError,
    ChessBoardStateError,
)


class ChessEnvironment:
    """
    Environnement d'échecs professionnel avec gestion d'erreurs robuste.

    Cette classe encapsule un objet chess.Board et fournit une interface
    sûre et extensible pour la manipulation d'un plateau d'échecs.

    Attributes:
        board: Instance de chess.Board
        logger: Logger pour tracer les opérations
        move_history: Historique complet des mouvements

    Example:
        >>> env = ChessEnvironment()
        >>> env.make_move('e2e4')
        True
        >>> env.display_unicode()
    """

    def __init__(self, fen: Optional[str] = None, enable_logging: bool = True):
        """
        Initialise l'environnement d'échecs.

        Args:
            fen: Position FEN optionnelle pour initialiser le plateau
            enable_logging: Active ou désactive le logging

        Raises:
            InvalidFENError: Si la notation FEN fournie est invalide
            ChessBoardStateError: Si l'initialisation échoue
        """
        self.logger = logging.getLogger(__name__) if enable_logging else None
        self._move_history: List[Move] = []

        try:
            if fen:
                self.board = Board(fen)
                if self.logger:
                    self.logger.info(f"Plateau initialisé avec FEN: {fen}")
            else:
                self.board = Board()
                if self.logger:
                    self.logger.info("Plateau initialisé en position de départ")

        except ValueError as e:
            error_msg = f"Impossible d'initialiser le plateau avec FEN: {fen}"
            if self.logger:
                self.logger.error(error_msg)
            raise InvalidFENError(fen or "", str(e))
        except Exception as e:
            error_msg = "Erreur inattendue lors de l'initialisation du plateau"
            if self.logger:
                self.logger.error(f"{error_msg}: {e}")
            raise ChessBoardStateError(error_msg)

    def reset_board(self) -> None:
        """
        Remet le plateau à la position de départ.

        Raises:
            ChessBoardStateError: Si la réinitialisation échoue
        """
        try:
            self.board.reset()
            self._move_history.clear()
            if self.logger:
                self.logger.info("Plateau réinitialisé à la position de départ")
        except Exception as e:
            error_msg = "Erreur lors de la réinitialisation du plateau"
            if self.logger:
                self.logger.error(f"{error_msg}: {e}")
            raise ChessBoardStateError(error_msg)

    def get_piece_at(self, square: Union[str, int, Square]) -> Optional[Piece]:
        """
        Retourne la pièce présente sur une case donnée.

        Args:
            square: Case à examiner (notation algébrique, indice ou Square)

        Returns:
            Pièce présente sur la case ou None si vide

        Raises:
            InvalidSquareError: Si la notation de case est invalide
        """
        try:
            # Valider la case
            if not isinstance(square, (int, str)):
                raise InvalidSquareError(f"Case invalide: {square}")

            if isinstance(square, str):
                try:
                    validated_square = chess.parse_square(square)
                except ValueError:
                    raise InvalidSquareError(f"Nom de case invalide: {square}")
            else:
                if not (0 <= square <= 63):
                    raise InvalidSquareError(f"Numéro de case invalide: {square}")
                validated_square = square
            piece = self.board.piece_at(validated_square)

            if self.logger and piece:
                square_name = chess.square_name(validated_square)
                self.logger.debug(f"Pièce trouvée en {square_name}: {piece.symbol()}")

            return piece

        except ValueError as e:
            if self.logger:
                self.logger.warning(f"Case invalide demandée: {square}")
            raise InvalidSquareError(str(square))

    def make_move(
        self, move: Union[str, Move], validate_game_state: bool = True
    ) -> bool:
        """
        Effectue un mouvement sur le plateau.

        Args:
            move: Mouvement à effectuer (UCI string ou objet Move)
            validate_game_state: Vérifie si la partie n'est pas terminée

        Returns:
            True si le mouvement a été effectué avec succès

        Raises:
            GameOverError: Si la partie est terminée et validate_game_state=True
            InvalidMoveError: Si le mouvement est invalide
        """
        if validate_game_state and self.is_game_over():
            result = self.get_game_result()
            raise GameOverError("effectuer un mouvement", result or "Inconnu")

        try:
            # Valider et convertir le mouvement
            if isinstance(move, str):
                try:
                    validated_move = chess.Move.from_uci(move)
                except ValueError:
                    raise InvalidMoveError(f"Format de mouvement invalide: {move}")
            elif isinstance(move, chess.Move):
                validated_move = move
            else:
                raise InvalidMoveError(f"Type de mouvement invalide: {type(move)}")

            if validated_move not in self.board.legal_moves:
                move_str = str(validated_move)
                if self.logger:
                    self.logger.warning(f"Mouvement illégal tenté: {move_str}")
                raise InvalidMoveError(
                    move_str,
                    "Le mouvement n'est pas dans la liste des mouvements légaux",
                )

            # Effectuer le mouvement
            self.board.push(validated_move)
            self._move_history.append(validated_move)

            if self.logger:
                self.logger.info(f"Mouvement effectué: {validated_move}")

            return True

        except InvalidMoveError:
            raise
        except Exception as e:
            move_str = str(move)
            if self.logger:
                self.logger.error(f"Erreur lors du mouvement {move_str}: {e}")
            raise InvalidMoveError(move_str, f"Erreur inattendue: {e}")

    def undo_move(self) -> bool:
        """
        Annule le dernier mouvement effectué.

        Returns:
            True si un mouvement a été annulé, False sinon

        Raises:
            ChessBoardStateError: Si une erreur survient lors de l'annulation
        """
        try:
            if not self.board.move_stack:
                if self.logger:
                    self.logger.info("Tentative d'annulation sans mouvement à annuler")
                return False

            undone_move = self.board.pop()
            if self._move_history:
                self._move_history.pop()

            if self.logger:
                self.logger.info(f"Mouvement annulé: {undone_move}")

            return True

        except Exception as e:
            error_msg = "Erreur lors de l'annulation du mouvement"
            if self.logger:
                self.logger.error(f"{error_msg}: {e}")
            raise ChessBoardStateError(error_msg)

    def get_legal_moves(self) -> List[Move]:
        """
        Retourne la liste de tous les mouvements légaux.

        Returns:
            Liste des mouvements légaux pour le joueur actuel
        """
        try:
            legal_moves = list(self.board.legal_moves)
            if self.logger:
                self.logger.debug(f"{len(legal_moves)} mouvements légaux disponibles")
            return legal_moves
        except Exception as e:
            error_msg = "Erreur lors de la génération des mouvements légaux"
            if self.logger:
                self.logger.error(f"{error_msg}: {e}")
            raise ChessBoardStateError(error_msg)

    def is_game_over(self) -> bool:
        """
        Vérifie si la partie est terminée.

        Returns:
            True si la partie est terminée
        """
        try:
            return self.board.is_game_over()
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Erreur lors de la vérification de fin de partie: {e}"
                )
            return False

    def get_game_result(self) -> Optional[str]:
        """
        Retourne le résultat de la partie si elle est terminée.

        Returns:
            Résultat de la partie ('1-0', '0-1', '1/2-1/2') ou None
        """
        try:
            if self.is_game_over():
                return self.board.result()
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'obtention du résultat: {e}")
            return None

    def get_board_fen(self) -> str:
        """
        Retourne la position actuelle en notation FEN.

        Returns:
            Position FEN du plateau actuel

        Raises:
            ChessBoardStateError: Si l'obtention du FEN échoue
        """
        try:
            return self.board.fen()
        except Exception as e:
            error_msg = "Erreur lors de l'obtention du FEN"
            if self.logger:
                self.logger.error(f"{error_msg}: {e}")
            raise ChessBoardStateError(error_msg)

    def get_current_player(self) -> Color:
        """
        Retourne la couleur du joueur actuel.

        Returns:
            chess.WHITE ou chess.BLACK
        """
        return self.board.turn

    def is_check(self) -> bool:
        """
        Vérifie si le roi actuel est en échec.

        Returns:
            True si le roi est en échec
        """
        try:
            return self.board.is_check()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de la vérification d'échec: {e}")
            return False

    def get_move_history(self) -> List[Move]:
        """
        Retourne l'historique complet des mouvements.

        Returns:
            Liste des mouvements effectués
        """
        return self._move_history.copy()

    def get_board_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques détaillées sur le plateau.

        Returns:
            Dictionnaire contenant diverses statistiques
        """
        try:
            return {
                "fen": self.get_board_fen(),
                "current_player": (
                    "White" if self.get_current_player() == chess.WHITE else "Black"
                ),
                "is_check": self.is_check(),
                "is_game_over": self.is_game_over(),
                "game_result": self.get_game_result(),
                "legal_moves_count": len(self.get_legal_moves()),
                "move_count": len(self._move_history),
                "castling_rights": {
                    "white_kingside": self.board.has_kingside_castling_rights(
                        chess.WHITE
                    ),
                    "white_queenside": self.board.has_queenside_castling_rights(
                        chess.WHITE
                    ),
                    "black_kingside": self.board.has_kingside_castling_rights(
                        chess.BLACK
                    ),
                    "black_queenside": self.board.has_queenside_castling_rights(
                        chess.BLACK
                    ),
                },
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'obtention des statistiques: {e}")
            return {}

    def __str__(self) -> str:
        """Représentation string du plateau."""
        return str(self.board)

    def __repr__(self) -> str:
        """Représentation officielle de l'objet."""
        try:
            return f"ChessEnvironment('{self.get_board_fen()}')"
        except:
            return "ChessEnvironment(<état invalide>)"
