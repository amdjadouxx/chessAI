"""
Évaluateur de référence pour les échecs
=======================================

Utilise des moteurs d'échecs établis comme Stockfish
pour fournir une évaluation de référence fiable.
"""

import chess
import chess.engine
from typing import Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


class ReferenceEvaluator:
    """
    Évaluateur utilisant Stockfish comme référence.

    Fournit une évaluation objective et fiable des positions,
    indépendante de l'IA en apprentissage.
    """

    def __init__(self, engine_path: Optional[str] = None, depth: int = 15):
        """
        Initialise l'évaluateur de référence.

        Args:
            engine_path: Chemin vers l'exécutable Stockfish (optionnel)
            depth: Profondeur d'analyse (par défaut 15)
        """
        self.engine_path = engine_path
        self.depth = depth
        self.engine = None
        self._setup_engine()

    def _setup_engine(self):
        """Configure le moteur Stockfish."""
        try:
            if self.engine_path and os.path.exists(self.engine_path):
                # Chemin spécifique fourni
                self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            else:
                # Essayer de trouver Stockfish automatiquement
                possible_paths = [
                    "stockfish",  # Dans le PATH
                    "./stockfish/stockfish.exe",  # Dans le projet (Windows)
                    "./stockfish/stockfish",  # Dans le projet (Linux/Mac)
                    "/usr/bin/stockfish",  # Linux
                    "/opt/homebrew/bin/stockfish",  # macOS avec Homebrew
                    "C:\\stockfish\\stockfish.exe",  # Windows installation classique
                    "stockfish.exe",  # Windows dans le PATH
                ]

                for path in possible_paths:
                    try:
                        self.engine = chess.engine.SimpleEngine.popen_uci(path)
                        logger.info(f"Stockfish trouvé : {path}")
                        break
                    except (FileNotFoundError, chess.engine.EngineTerminatedError):
                        continue

                if self.engine is None:
                    logger.warning(
                        "Stockfish non trouvé. Utilisation de l'évaluation basique."
                    )

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Stockfish : {e}")
            self.engine = None

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Évalue une position d'échecs.

        Args:
            board: Position à évaluer

        Returns:
            Évaluation entre -1.0 et 1.0
            (+1 = blanc gagne, 0 = égal, -1 = noir gagne)
        """
        if self.engine is None:
            return self._basic_evaluation(board)

        try:
            # Analyse avec Stockfish
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info.get("score")

            if score is None:
                return self._basic_evaluation(board)

            # Convertir le score du point de vue des blancs
            white_score = score.white()

            # Convertir le score en valeur normalisée
            try:
                # Vérifier si c'est un mat
                if hasattr(white_score, "is_mate") and white_score.is_mate():
                    # Mat en X coups
                    if hasattr(white_score, "mate") and white_score.mate() is not None:
                        mate_in = white_score.mate()
                        if mate_in > 0:
                            return 0.95  # Blanc gagne bientôt
                        else:
                            return -0.95  # Noir gagne bientôt
                    else:
                        # Mat détecté mais pas de nombre de coups
                        return 0.9 if str(white_score).startswith("+") else -0.9
                else:
                    # Score en centipions (1 pion = 100 centipions)
                    if (
                        hasattr(white_score, "score")
                        and white_score.score() is not None
                    ):
                        centipawns = white_score.score()
                    else:
                        # Fallback : essayer de parser la string
                        score_str = str(white_score)
                        if score_str.startswith("+"):
                            centipawns = 200  # Avantage blanc
                        elif score_str.startswith("-"):
                            centipawns = -200  # Avantage noir
                        else:
                            centipawns = 0

                    # Normaliser le score (sigmoid)
                    # Un avantage de ~400 centipions = position très favorable
                    normalized = 2.0 / (1.0 + pow(10, -centipawns / 400.0)) - 1.0
                    return max(-1.0, min(1.0, normalized))

            except AttributeError as attr_error:
                # Fallback pour les objets POV score problématiques
                score_str = str(white_score)
                if "#" in score_str:  # Mat
                    return 0.9 if score_str.startswith("+") else -0.9
                elif "+" in score_str:
                    return 0.3  # Léger avantage blanc
                elif "-" in score_str:
                    return -0.3  # Léger avantage noir
                else:
                    return 0.0  # Égalité

        except Exception as e:
            logger.warning(f"Erreur Stockfish, utilisation évaluation basique : {e}")
            return self._basic_evaluation(board)

    def _basic_evaluation(self, board: chess.Board) -> float:
        """
        Évaluation basique sans moteur externe.

        Compte simplement le matériel et quelques facteurs positionnels.
        """
        if board.is_checkmate():
            return -1.0 if board.turn else 1.0

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        # Valeurs des pièces
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }

        white_score = 0
        black_score = 0

        # Compter le matériel
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_score += value
                else:
                    black_score += value

        # Bonus/malus positionnels simples
        white_score += len(list(board.legal_moves)) * 0.1 if board.turn else 0

        # Normaliser
        total_material = white_score + black_score
        if total_material == 0:
            return 0.0

        advantage = (white_score - black_score) / total_material
        return max(-1.0, min(1.0, advantage))

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Obtient le meilleur coup selon Stockfish.

        Args:
            board: Position actuelle

        Returns:
            Meilleur coup ou None si pas disponible
        """
        if self.engine is None:
            return None

        try:
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
            return result.move
        except Exception as e:
            logger.warning(f"Erreur lors de la recherche du meilleur coup : {e}")
            return None

    def get_analysis(self, board: chess.Board) -> Dict[str, Any]:
        """
        Analyse détaillée de la position.

        Returns:
            Dictionnaire avec évaluation, meilleur coup, etc.
        """
        evaluation = self.evaluate_position(board)
        best_move = self.get_best_move(board)

        return {
            "evaluation": evaluation,
            "best_move": best_move,
            "engine": "Stockfish" if self.engine else "Basic",
            "depth": self.depth if self.engine else 1,
        }

    def close(self):
        """Ferme le moteur."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def __del__(self):
        """Destructeur - ferme le moteur."""
        self.close()


# Instance globale pour éviter les multiples connexions
_reference_evaluator = None


def get_reference_evaluator() -> ReferenceEvaluator:
    """
    Obtient l'instance globale de l'évaluateur de référence.
    """
    global _reference_evaluator
    if _reference_evaluator is None:
        _reference_evaluator = ReferenceEvaluator()
    return _reference_evaluator
