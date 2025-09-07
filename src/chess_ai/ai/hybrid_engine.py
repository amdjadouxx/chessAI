"""
Moteur d'IA Hybride : Stockfish + Neural Network + MCTS
======================================================

Combine intelligemment :
- Stockfish pour la précision et force immédiate
- Neural Network pour la vitesse et créativité
- MCTS pour l'exploration optimale

Stratégie adaptative selon le contexte (temps, criticité, phase de jeu)
"""

import chess
import chess.engine
import torch
import time
import os
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass

from .network import ChessNet, encode_board, decode_policy
from .mcts import MCTS
from .reference_evaluator import get_reference_evaluator


class AIMode(Enum):
    """Modes de fonctionnement de l'IA hybride."""

    STOCKFISH_ONLY = "stockfish_only"
    NEURAL_ONLY = "neural_only"
    HYBRID_FAST = "hybrid_fast"
    HYBRID_DEEP = "hybrid_deep"
    ADAPTIVE = "adaptive"


@dataclass
class GameContext:
    """Contexte de jeu pour adaptation intelligente."""

    time_left: float = 60.0  # Temps restant (secondes)
    increment: float = 0.0  # Incrément par coup
    move_number: int = 1  # Numéro du coup
    material_balance: float = 0.0  # Équilibre matériel
    position_complexity: float = 0.5  # Complexité de la position (0-1)
    is_critical: bool = False  # Position critique (mat proche, etc.)
    player_strength: int = 1500  # Force estimée de l'adversaire


@dataclass
class AIDecision:
    """Résultat d'une décision de l'IA."""

    move: chess.Move
    evaluation: float
    confidence: float
    method_used: str
    calculation_time: float
    principal_variation: List[chess.Move]
    alternative_moves: List[Tuple[chess.Move, float]]


class HybridAI:
    """
    IA Hybride combinant Stockfish, Neural Network et MCTS.

    Sélectionne automatiquement la meilleure approche selon le contexte.
    """

    def __init__(
        self,
        stockfish_path: str = None,
        neural_model_path: str = None,
        device: str = "cpu",
        default_mode: AIMode = AIMode.ADAPTIVE,
    ):
        """
        Initialise l'IA hybride.

        Args:
            stockfish_path: Chemin vers l'exécutable Stockfish
            neural_model_path: Chemin vers le modèle neural network
            device: Device PyTorch (cpu/cuda)
            default_mode: Mode par défaut
        """
        self.device = torch.device(device)
        self.default_mode = default_mode
        self.current_mode = default_mode  # Mode actuel de l'IA

        # Initialiser Stockfish
        self.stockfish = None
        self.stockfish_available = False
        if stockfish_path and os.path.exists(stockfish_path):
            try:
                self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                self.stockfish_available = True
                print("✅ Stockfish initialisé")
            except Exception as e:
                print(f"⚠️ Stockfish non disponible: {e}")

        if not self.stockfish_available:
            # Fallback sur l'évaluateur de référence
            self.reference_evaluator = get_reference_evaluator()
            print("📊 Utilisation de l'évaluateur de référence")

        # Initialiser Neural Network
        self.neural_net = ChessNet().to(self.device)
        self.neural_available = False
        if neural_model_path and os.path.exists(neural_model_path):
            try:
                checkpoint = torch.load(neural_model_path, map_location=self.device)
                self.neural_net.load_state_dict(
                    checkpoint.get("model_state_dict", checkpoint)
                )
                self.neural_net.eval()
                self.neural_available = True
                print("🧠 Neural Network chargé")
            except Exception as e:
                print(f"⚠️ Neural Network non disponible: {e}")

        # Initialiser MCTS
        self.mcts = MCTS(
            neural_net=self.neural_net if self.neural_available else None,
            c_puct=1.4,
            device=self.device,
        )

        # Statistiques
        self.stats = {
            "stockfish_calls": 0,
            "neural_calls": 0,
            "hybrid_calls": 0,
            "total_time": 0.0,
        }

    def evaluate_position(self, board: chess.Board, depth: int = 8) -> float:
        """
        Évalue une position avec la méthode la plus appropriée.

        Args:
            board: Position à évaluer
            depth: Profondeur d'analyse (pour Stockfish)

        Returns:
            Évaluation de la position (-1 à +1, perspective du joueur au trait)
        """
        if self.stockfish_available:
            return self._stockfish_evaluate(board, depth)
        elif self.neural_available:
            return self._neural_evaluate(board)
        else:
            return self._reference_evaluate(board)

    def _stockfish_evaluate(self, board: chess.Board, depth: int) -> float:
        """Évaluation par Stockfish."""
        try:
            info = self.stockfish.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].relative

            if score.is_mate():
                return 1.0 if score.mate() > 0 else -1.0
            else:
                # Convertir centipawns en évaluation normalisée
                cp = score.score()
                return max(-1.0, min(1.0, cp / 1000.0))

        except Exception as e:
            print(f"Erreur Stockfish: {e}")
            return self._reference_evaluate(board)

    def _neural_evaluate(self, board: chess.Board) -> float:
        """Évaluation par Neural Network."""
        try:
            with torch.no_grad():
                board_tensor = encode_board(board).unsqueeze(0).to(self.device)
                _, value = self.neural_net(board_tensor)
                evaluation = value.item()

                # Ajuster pour la perspective du joueur
                if board.turn == chess.BLACK:
                    evaluation = -evaluation

                return evaluation

        except Exception as e:
            print(f"Erreur Neural Network: {e}")
            return self._reference_evaluate(board)

    def _reference_evaluate(self, board: chess.Board) -> float:
        """Évaluation de référence (fallback)."""
        if hasattr(self, "reference_evaluator"):
            return self.reference_evaluator.evaluate_position(board)
        else:
            # Évaluation basique matérielle
            return self._basic_material_evaluation(board)

    def _basic_material_evaluation(self, board: chess.Board) -> float:
        """Évaluation matérielle basique."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }

        white_material = sum(
            values[piece.piece_type]
            for piece in board.piece_map().values()
            if piece.color == chess.WHITE
        )

        black_material = sum(
            values[piece.piece_type]
            for piece in board.piece_map().values()
            if piece.color == chess.BLACK
        )

        material_diff = white_material - black_material
        normalized = material_diff / 40.0  # Normaliser approximativement

        return normalized if board.turn == chess.WHITE else -normalized

    def get_move(
        self, board: chess.Board, context: GameContext = None, mode: AIMode = None
    ) -> AIDecision:
        """
        Obtient le meilleur coup selon le contexte.

        Args:
            board: Position actuelle
            context: Contexte de jeu
            mode: Mode spécifique (override le mode par défaut)

        Returns:
            Décision de l'IA avec tous les détails
        """
        start_time = time.time()
        context = context or GameContext()
        mode = mode or self._select_optimal_mode(context)

        try:
            if mode == AIMode.STOCKFISH_ONLY:
                decision = self._stockfish_move(board, context)
            elif mode == AIMode.NEURAL_ONLY:
                decision = self._neural_move(board, context)
            elif mode == AIMode.HYBRID_FAST:
                decision = self._hybrid_fast_move(board, context)
            elif mode == AIMode.HYBRID_DEEP:
                decision = self._hybrid_deep_move(board, context)
            else:  # ADAPTIVE
                decision = self._adaptive_move(board, context)

            decision.calculation_time = time.time() - start_time
            self.stats["total_time"] += decision.calculation_time

            return decision

        except Exception as e:
            print(f"Erreur lors du calcul du coup: {e}")
            # Fallback sur coup aléatoire légal
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return AIDecision(
                    move=legal_moves[0],
                    evaluation=0.0,
                    confidence=0.0,
                    method_used="fallback_random",
                    calculation_time=time.time() - start_time,
                    principal_variation=[],
                    alternative_moves=[],
                )
            else:
                raise ValueError("Aucun coup légal disponible")

    def _select_optimal_mode(self, context: GameContext) -> AIMode:
        """Sélectionne le mode optimal selon le contexte."""

        # Position critique → Stockfish profond
        if context.is_critical:
            return (
                AIMode.STOCKFISH_ONLY
                if self.stockfish_available
                else AIMode.HYBRID_DEEP
            )

        # Pression temporelle → Neural rapide
        if context.time_left < 5.0:
            return AIMode.NEURAL_ONLY if self.neural_available else AIMode.HYBRID_FAST

        # Ouverture → Neural (patterns)
        if context.move_number < 10:
            return AIMode.HYBRID_FAST

        # Finale → Stockfish (précision)
        if abs(context.material_balance) > 15:
            return (
                AIMode.STOCKFISH_ONLY
                if self.stockfish_available
                else AIMode.HYBRID_DEEP
            )

        # Milieu de jeu → Hybride adaptatif
        return AIMode.HYBRID_DEEP

    def _stockfish_move(self, board: chess.Board, context: GameContext) -> AIDecision:
        """Calcul de coup par Stockfish pur."""
        if not self.stockfish_available:
            return self._neural_move(board, context)

        self.stats["stockfish_calls"] += 1

        # Profondeur adaptative selon le temps
        if context.time_left > 10.0:
            depth = 15
        elif context.time_left > 2.0:
            depth = 10
        else:
            depth = 6

        try:
            result = self.stockfish.play(board, chess.engine.Limit(depth=depth))
            evaluation = self._stockfish_evaluate(board, depth)

            # Obtenir les variantes principales
            info = self.stockfish.analyse(
                board, chess.engine.Limit(depth=depth), multipv=3
            )
            alternatives = []
            pv = []

            if isinstance(info, list):
                main_info = info[0]
                pv = main_info.get("pv", [])

                for i, variant in enumerate(info[:3]):
                    if "pv" in variant and variant["pv"]:
                        move = variant["pv"][0]
                        score = variant["score"].relative.score() or 0
                        alternatives.append((move, score / 100.0))

            return AIDecision(
                move=result.move,
                evaluation=evaluation,
                confidence=0.95,  # Stockfish très fiable
                method_used=f"stockfish_d{depth}",
                calculation_time=0.0,  # Sera calculé dans get_move
                principal_variation=pv[:5],
                alternative_moves=alternatives,
            )

        except Exception as e:
            print(f"Erreur Stockfish: {e}")
            return self._neural_move(board, context)

    def _neural_move(self, board: chess.Board, context: GameContext) -> AIDecision:
        """Calcul de coup par Neural Network pur."""
        if not self.neural_available:
            return self._reference_move(board, context)

        self.stats["neural_calls"] += 1

        try:
            with torch.no_grad():
                board_tensor = encode_board(board).unsqueeze(0).to(self.device)
                policy_logits, value = self.neural_net(board_tensor)

                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    raise ValueError("Aucun coup légal")

                move_probs = decode_policy(policy_logits[0], legal_moves)

                # Sélectionner le meilleur coup
                best_move = max(move_probs, key=move_probs.get)
                evaluation = value.item()

                if board.turn == chess.BLACK:
                    evaluation = -evaluation

                # Alternatives
                sorted_moves = sorted(
                    move_probs.items(), key=lambda x: x[1], reverse=True
                )
                alternatives = [(move, prob) for move, prob in sorted_moves[:3]]

                return AIDecision(
                    move=best_move,
                    evaluation=evaluation,
                    confidence=max(move_probs.values()),
                    method_used="neural_net",
                    calculation_time=0.0,
                    principal_variation=[best_move],
                    alternative_moves=alternatives,
                )

        except Exception as e:
            print(f"Erreur Neural Network: {e}")
            return self._reference_move(board, context)

    def _hybrid_fast_move(self, board: chess.Board, context: GameContext) -> AIDecision:
        """Hybride rapide : Neural filtre, Stockfish valide."""
        self.stats["hybrid_calls"] += 1

        # Phase 1 : Neural Network identifie les coups prometteurs
        neural_decision = self._neural_move(board, context)

        if not self.stockfish_available:
            neural_decision.method_used = "hybrid_fast_neural_only"
            return neural_decision

        # Phase 2 : Stockfish valide rapidement le top coup
        try:
            quick_eval = self._stockfish_evaluate(board, depth=6)

            # Test : jouer le coup suggéré par Neural Net
            board_copy = board.copy()
            board_copy.push(neural_decision.move)
            after_move_eval = self._stockfish_evaluate(board_copy, depth=6)

            # Si le coup Neural Net semble bon, l'utiliser
            if abs(after_move_eval - (-quick_eval)) < 0.5:  # Cohérence
                neural_decision.method_used = "hybrid_fast_validated"
                neural_decision.confidence = min(0.9, neural_decision.confidence + 0.2)
                return neural_decision
            else:
                # Sinon, demander à Stockfish
                return self._stockfish_move(board, context)

        except Exception as e:
            print(f"Erreur hybrid fast: {e}")
            return neural_decision

    def _hybrid_deep_move(self, board: chess.Board, context: GameContext) -> AIDecision:
        """Hybride profond : Neural + MCTS + Stockfish."""
        self.stats["hybrid_calls"] += 1

        try:
            # Phase 1 : MCTS avec Neural Network
            if self.neural_available:
                num_simulations = min(200, max(50, int(context.time_left * 20)))
                mcts_policy = self.mcts.run(board, num_simulations)

                if mcts_policy:
                    mcts_move = max(mcts_policy, key=mcts_policy.get)
                    mcts_confidence = mcts_policy[mcts_move]
                else:
                    # Fallback Neural pur
                    neural_decision = self._neural_move(board, context)
                    mcts_move = neural_decision.move
                    mcts_confidence = neural_decision.confidence
            else:
                # Pas de Neural Net, utiliser Stockfish
                return self._stockfish_move(board, context)

            # Phase 2 : Validation/correction par Stockfish
            if self.stockfish_available and context.time_left > 1.0:
                stockfish_decision = self._stockfish_move(board, context)

                # Comparer les suggestions
                if mcts_move == stockfish_decision.move:
                    # Accord parfait
                    return AIDecision(
                        move=mcts_move,
                        evaluation=stockfish_decision.evaluation,
                        confidence=min(
                            0.95, max(mcts_confidence, stockfish_decision.confidence)
                        ),
                        method_used="hybrid_deep_consensus",
                        calculation_time=0.0,
                        principal_variation=stockfish_decision.principal_variation,
                        alternative_moves=stockfish_decision.alternative_moves,
                    )
                else:
                    # Désaccord : utiliser Stockfish mais noter le désaccord
                    stockfish_decision.method_used = "hybrid_deep_stockfish_override"
                    stockfish_decision.confidence *= 0.8  # Réduire confiance
                    return stockfish_decision
            else:
                # Pas de temps pour Stockfish, utiliser MCTS
                evaluation = (
                    self._neural_evaluate(board) if self.neural_available else 0.0
                )
                return AIDecision(
                    move=mcts_move,
                    evaluation=evaluation,
                    confidence=mcts_confidence,
                    method_used="hybrid_deep_mcts_only",
                    calculation_time=0.0,
                    principal_variation=[mcts_move],
                    alternative_moves=[],
                )

        except Exception as e:
            print(f"Erreur hybrid deep: {e}")
            return (
                self._stockfish_move(board, context)
                if self.stockfish_available
                else self._neural_move(board, context)
            )

    def _adaptive_move(self, board: chess.Board, context: GameContext) -> AIDecision:
        """Mode adaptatif intelligent."""
        # Analyser la complexité de la position
        context.position_complexity = self._assess_position_complexity(board)
        context.is_critical = self._is_critical_position(board)

        # Sélectionner la méthode optimale
        if context.is_critical and self.stockfish_available:
            return self._stockfish_move(board, context)
        elif context.position_complexity > 0.7:
            return self._hybrid_deep_move(board, context)
        elif context.time_left < 2.0:
            return self._neural_move(board, context)
        else:
            return self._hybrid_fast_move(board, context)

    def _assess_position_complexity(self, board: chess.Board) -> float:
        """Évalue la complexité d'une position (0-1)."""
        complexity = 0.0

        # Nombre de pièces (plus = plus complexe)
        num_pieces = len(board.piece_map())
        complexity += (num_pieces / 32.0) * 0.3

        # Nombre de coups légaux (plus = plus complexe)
        num_moves = len(list(board.legal_moves))
        complexity += min(num_moves / 50.0, 1.0) * 0.4

        # Présence de menaces tactiques
        if board.is_check():
            complexity += 0.2

        # Phase de jeu (milieu = plus complexe)
        if 10 <= num_pieces <= 20:
            complexity += 0.1

        return min(complexity, 1.0)

    def _is_critical_position(self, board: chess.Board) -> bool:
        """Détermine si une position est critique."""
        # Échec
        if board.is_check():
            return True

        # Peu de matériel (finale)
        if len(board.piece_map()) < 10:
            return True

        # Mat proche détectable
        if self.stockfish_available:
            try:
                info = self.stockfish.analyse(board, chess.engine.Limit(depth=6))
                score = info["score"].relative
                if score.is_mate() and abs(score.mate()) <= 3:
                    return True
            except:
                pass

        return False

    def _reference_move(self, board: chess.Board, context: GameContext) -> AIDecision:
        """Coup de référence basique (dernier recours)."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("Aucun coup légal")

        # Choisir un coup semi-intelligent
        best_move = legal_moves[0]
        best_score = -999999

        for move in legal_moves[:10]:  # Limiter pour la vitesse
            board_copy = board.copy()
            board_copy.push(move)
            score = self._basic_material_evaluation(board_copy)

            if score > best_score:
                best_score = score
                best_move = move

        return AIDecision(
            move=best_move,
            evaluation=best_score,
            confidence=0.3,
            method_used="reference_basic",
            calculation_time=0.0,
            principal_variation=[best_move],
            alternative_moves=[(move, 0.1) for move in legal_moves[:3]],
        )

    def get_stats(self) -> Dict:
        """Retourne les statistiques d'utilisation."""
        total_calls = sum(
            [
                self.stats["stockfish_calls"],
                self.stats["neural_calls"],
                self.stats["hybrid_calls"],
            ]
        )

        if total_calls == 0:
            return self.stats

        return {
            **self.stats,
            "stockfish_ratio": self.stats["stockfish_calls"] / total_calls,
            "neural_ratio": self.stats["neural_calls"] / total_calls,
            "hybrid_ratio": self.stats["hybrid_calls"] / total_calls,
            "avg_time_per_move": (
                self.stats["total_time"] / total_calls if total_calls > 0 else 0
            ),
        }

    def cleanup(self):
        """Nettoie les ressources."""
        if self.stockfish:
            try:
                self.stockfish.quit()
            except:
                pass

    def __del__(self):
        """Destructeur."""
        self.cleanup()


# Fonction helper pour créer une IA hybride configurée
def create_hybrid_ai(
    stockfish_path: str = "stockfish/stockfish.exe",
    neural_model_path: str = "models/alphazero_final.pth",
    mode: AIMode = AIMode.ADAPTIVE,
) -> HybridAI:
    """
    Crée une IA hybride avec les chemins par défaut.

    Args:
        stockfish_path: Chemin vers Stockfish
        neural_model_path: Chemin vers le modèle neural
        mode: Mode par défaut

    Returns:
        Instance d'IA hybride configurée
    """
    return HybridAI(
        stockfish_path=stockfish_path,
        neural_model_path=neural_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        default_mode=mode,
    )


if __name__ == "__main__":
    # Test de l'IA hybride
    ai = create_hybrid_ai()

    board = chess.Board()
    context = GameContext(time_left=30.0, move_number=1)

    print("🎯 Test de l'IA Hybride")
    print("=" * 40)

    decision = ai.get_move(board, context)

    print(f"Coup choisi: {decision.move}")
    print(f"Évaluation: {decision.evaluation:.3f}")
    print(f"Confiance: {decision.confidence:.3f}")
    print(f"Méthode: {decision.method_used}")
    print(f"Temps: {decision.calculation_time:.3f}s")

    print(f"\nStatistiques: {ai.get_stats()}")

    ai.cleanup()
