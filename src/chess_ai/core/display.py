"""
Module d'affichage pour Chess AI.

Ce module gère tous les aspects d'affichage et de visualisation
du plateau d'échecs avec des options avancées.
"""

from typing import Optional, Dict, Any
import chess
from chess import Board, Color

from ..exceptions import ChessBoardStateError
from ..utils.logging_config import get_logger


class ChessDisplay:
    """
    Gestionnaire d'affichage pour plateaux d'échecs.

    Cette classe fournit diverses méthodes d'affichage
    avec des options de personnalisation avancées.
    """

    # Symboles Unicode pour les pièces
    UNICODE_PIECES = {
        "K": "♔",
        "Q": "♕",
        "R": "♖",
        "B": "♗",
        "N": "♘",
        "P": "♙",
        "k": "♚",
        "q": "♛",
        "r": "♜",
        "b": "♝",
        "n": "♞",
        "p": "♟",
    }

    # Symboles ASCII pour les pièces
    ASCII_PIECES = {
        "K": "K",
        "Q": "Q",
        "R": "R",
        "B": "B",
        "N": "N",
        "P": "P",
        "k": "k",
        "q": "q",
        "r": "r",
        "b": "b",
        "n": "n",
        "p": "p",
    }

    def __init__(self, board: Board, enable_logging: bool = True):
        """
        Initialise le gestionnaire d'affichage.

        Args:
            board: Plateau d'échecs à afficher
            enable_logging: Active le logging
        """
        self.board = board
        self.logger = get_logger(__name__) if enable_logging else None

    def display_unicode(
        self,
        perspective: Color = chess.WHITE,
        show_coordinates: bool = True,
        highlight_squares: Optional[list] = None,
    ) -> None:
        """
        Affiche le plateau avec des symboles Unicode.

        Args:
            perspective: Point de vue (WHITE ou BLACK)
            show_coordinates: Affiche les coordonnées
            highlight_squares: Cases à mettre en évidence
        """
        try:
            print("\n" + "=" * 35)
            print("    PLATEAU D'ÉCHECS (Unicode)")
            print("=" * 35)

            # Utiliser l'affichage Unicode intégré de python-chess
            if perspective == chess.WHITE:
                board_str = self.board.unicode(
                    invert_color=False, borders=show_coordinates
                )
            else:
                board_str = self.board.unicode(
                    invert_color=True, borders=show_coordinates
                )

            print(board_str)

            # Informations sur le jeu
            self._display_game_info()

            if self.logger:
                perspective_name = "blancs" if perspective == chess.WHITE else "noirs"
                self.logger.debug(f"Plateau affiché (perspective: {perspective_name})")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'affichage Unicode: {e}")
            raise ChessBoardStateError("Impossible d'afficher le plateau en Unicode")

    def display_ascii(
        self,
        perspective: Color = chess.WHITE,
        show_coordinates: bool = True,
        show_border: bool = True,
    ) -> None:
        """
        Affiche le plateau avec des caractères ASCII simples.

        Args:
            perspective: Point de vue (WHITE ou BLACK)
            show_coordinates: Affiche les coordonnées
            show_border: Affiche les bordures
        """
        try:
            print("\n" + "=" * 35)
            print("    PLATEAU D'ÉCHECS (ASCII)")
            print("=" * 35)

            # Créer l'affichage ASCII personnalisé
            board_lines = str(self.board).split("\n")

            if show_border:
                print("  +" + "-" * 17 + "+")

            if perspective == chess.WHITE:
                for i, line in enumerate(board_lines):
                    rank = 8 - i
                    if show_coordinates:
                        formatted_line = line.replace(".", "·")  # Point visible
                        print(f"{rank} | {formatted_line} |")
                    else:
                        print(f"  | {line} |")
            else:
                for i, line in enumerate(reversed(board_lines)):
                    rank = i + 1
                    if show_coordinates:
                        formatted_line = line.replace(".", "·")
                        # Inverser l'ordre des colonnes pour la vue noire
                        chars = list(formatted_line.replace(" ", ""))
                        reversed_line = " ".join(reversed(chars))
                        print(f"{rank} | {reversed_line} |")
                    else:
                        print(f"  | {line} |")

            if show_border:
                print("  +" + "-" * 17 + "+")

            if show_coordinates:
                if perspective == chess.WHITE:
                    print("    a b c d e f g h")
                else:
                    print("    h g f e d c b a")

            # Informations sur le jeu
            self._display_game_info()

            if self.logger:
                self.logger.debug("Plateau affiché en ASCII")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'affichage ASCII: {e}")
            raise ChessBoardStateError("Impossible d'afficher le plateau en ASCII")

    def display_compact(self) -> None:
        """
        Affiche une version compacte du plateau.
        """
        try:
            fen = self.board.fen()
            position_part = fen.split(" ")[0]

            print(f"\nPosition FEN: {fen}")
            print(f"Position: {position_part}")

            # Tour actuel
            current_player = "Blancs" if self.board.turn == chess.WHITE else "Noirs"
            print(f"Tour: {current_player}")

            # État du jeu
            if self.board.is_check():
                print("État: ÉCHEC!")
            elif self.board.is_checkmate():
                print("État: ÉCHEC ET MAT!")
            elif self.board.is_stalemate():
                print("État: PAT (Match nul)")
            else:
                print("État: En cours")

            print()

            if self.logger:
                self.logger.debug("Affichage compact effectué")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'affichage compact: {e}")
            print("Erreur lors de l'affichage compact du plateau")

    def display_statistics(self) -> None:
        """
        Affiche des statistiques détaillées sur la position.
        """
        try:
            print("\n" + "=" * 40)
            print("       STATISTIQUES DE POSITION")
            print("=" * 40)

            # Statistiques de base
            legal_moves = list(self.board.legal_moves)
            print(f"Mouvements légaux: {len(legal_moves)}")
            print(
                f"Joueur actuel: {'Blancs' if self.board.turn == chess.WHITE else 'Noirs'}"
            )

            # Matériel
            material = self._count_material()
            print("\nMatériel:")
            print(f"  Blancs: {material['white']}")
            print(f"  Noirs: {material['black']}")

            # Droits de roque
            print("\nDroits de roque:")
            print(
                f"  Blancs - Côté roi: {self.board.has_kingside_castling_rights(chess.WHITE)}"
            )
            print(
                f"  Blancs - Côté dame: {self.board.has_queenside_castling_rights(chess.WHITE)}"
            )
            print(
                f"  Noirs - Côté roi: {self.board.has_kingside_castling_rights(chess.BLACK)}"
            )
            print(
                f"  Noirs - Côté dame: {self.board.has_queenside_castling_rights(chess.BLACK)}"
            )

            # État de la partie
            print("\nÉtat de la partie:")
            if self.board.is_checkmate():
                print("  🏁 Échec et mat")
            elif self.board.is_stalemate():
                print("  🤝 Pat (match nul)")
            elif self.board.is_check():
                print("  ⚠️  Échec")
            elif self.board.is_insufficient_material():
                print("  📉 Matériel insuffisant")
            else:
                print("  ▶️  Partie en cours")

            print()

            if self.logger:
                self.logger.debug("Statistiques affichées")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'affichage des statistiques: {e}")
            print("Erreur lors de l'affichage des statistiques")

    def display_move_history(self, move_history: list, max_moves: int = 10) -> None:
        """
        Affiche l'historique des mouvements.

        Args:
            move_history: Liste des mouvements
            max_moves: Nombre maximum de mouvements à afficher
        """
        try:
            if not move_history:
                print("\nAucun mouvement dans l'historique")
                return

            print(
                f"\nHistorique des mouvements (derniers {min(max_moves, len(move_history))}):"
            )
            print("-" * 30)

            start_index = max(0, len(move_history) - max_moves)
            for i, move in enumerate(move_history[start_index:], start_index + 1):
                print(f"{i:2d}. {move}")

            print()

            if self.logger:
                self.logger.debug(
                    f"Historique affiché ({len(move_history)} mouvements)"
                )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'affichage de l'historique: {e}")
            print("Erreur lors de l'affichage de l'historique")

    def _display_game_info(self) -> None:
        """Affiche les informations de base sur l'état du jeu."""
        try:
            current_player = "Blancs" if self.board.turn == chess.WHITE else "Noirs"
            print(f"\nTour: {current_player}")

            # États spéciaux
            status_messages = []
            if self.board.is_check():
                status_messages.append("⚠️  ÉCHEC!")
            if self.board.is_checkmate():
                status_messages.append("🏁 ÉCHEC ET MAT!")
            if self.board.is_stalemate():
                status_messages.append("🤝 PAT (Match nul)")
            if self.board.is_insufficient_material():
                status_messages.append("📉 Matériel insuffisant")
            if self.board.is_seventyfive_moves():
                status_messages.append("🔄 Règle des 75 coups")
            if self.board.is_fivefold_repetition():
                status_messages.append("🔄 Répétition quintuple")

            for message in status_messages:
                print(message)

            print()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors de l'affichage des infos de jeu: {e}")

    def _count_material(self) -> Dict[str, Dict[str, int]]:
        """Compte le matériel pour l'affichage des statistiques."""
        try:
            material = {
                "white": {"R": 0, "N": 0, "B": 0, "Q": 0, "P": 0, "K": 0},
                "black": {"r": 0, "n": 0, "b": 0, "q": 0, "p": 0, "k": 0},
            }

            piece_types = [
                chess.ROOK,
                chess.KNIGHT,
                chess.BISHOP,
                chess.QUEEN,
                chess.PAWN,
                chess.KING,
            ]
            white_symbols = ["R", "N", "B", "Q", "P", "K"]
            black_symbols = ["r", "n", "b", "q", "p", "k"]

            for piece_type, white_sym, black_sym in zip(
                piece_types, white_symbols, black_symbols
            ):
                white_count = len(self.board.pieces(piece_type, chess.WHITE))
                black_count = len(self.board.pieces(piece_type, chess.BLACK))

                material["white"][white_sym] = white_count
                material["black"][black_sym] = black_count

            return material

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erreur lors du comptage de matériel: {e}")
            return {"white": {}, "black": {}}
