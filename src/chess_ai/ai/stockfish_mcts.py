"""
MCTS Hybride avec Stockfish - Évaluations Précises
=================================================

Version améliorée du MCTS qui utilise Stockfish pour les évaluations
au lieu du Neural Network, garantissant des simulations correctes.
"""

import chess
import torch
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from .mcts import MCTSNode, MCTS
from .network import encode_board, decode_policy


class StockfishMCTSNode(MCTSNode):
    """Nœud MCTS qui utilise Stockfish pour l'évaluation."""

    def __init__(self, board: chess.Board, parent=None, move=None, prior=0.0):
        super().__init__(board, parent, move, prior)
        self._stockfish_value = None  # Cache pour l'évaluation Stockfish


class StockfishMCTS(MCTS):
    """
    MCTS hybride qui utilise :
    - Neural Network pour les politiques (rapidité)
    - Stockfish pour les évaluations (précision)
    """

    def __init__(
        self,
        neural_net: torch.nn.Module,
        stockfish_evaluator,
        c_puct: float = 1.4,
        device: str = "cpu",
    ):
        super().__init__(neural_net, c_puct, device)
        self.stockfish_evaluator = stockfish_evaluator
        self.evaluation_cache = {}  # Cache des évaluations Stockfish

    def predict(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """
        🚀 HYBRIDE : Neural Network pour politique, Stockfish pour valeur

        Returns:
            Tuple (move_probabilities, stockfish_value)
        """
        # 1. 🧠 Neural Network pour la politique (rapide)
        encoded = encode_board(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.neural_net(encoded)  # Ignorer la valeur NN

        # Décoder la politique pour les coups légaux
        legal_moves = list(board.legal_moves)
        move_probs = decode_policy(policy_logits[0], legal_moves)

        # 2. 🤖 Stockfish pour l'évaluation (précise)
        stockfish_value = self.get_stockfish_evaluation(board)

        return move_probs, stockfish_value

    def get_stockfish_evaluation(self, board: chess.Board) -> float:
        """
        Obtient l'évaluation Stockfish de la position avec cache.

        Returns:
            Valeur normalisée entre -1 et 1
        """
        # Utiliser le FEN comme clé de cache
        fen = board.fen()

        if fen in self.evaluation_cache:
            return self.evaluation_cache[fen]

        try:
            # Évaluation Stockfish en centipawns
            centipawns = self.stockfish_evaluator.evaluate_position(board)

            # Normaliser entre -1 et 1
            if abs(centipawns) > 1000:  # Mat forcé
                value = 1.0 if centipawns > 0 else -1.0
            else:
                # Fonction sigmoïde pour normaliser
                value = np.tanh(centipawns / 400.0)  # 400cp ≈ 0.76

            # Ajuster selon le joueur à jouer
            if board.turn == chess.BLACK:
                value = -value

            # Mettre en cache
            self.evaluation_cache[fen] = value

            return value

        except Exception as e:
            print(f"⚠️ Erreur évaluation Stockfish: {e}")
            # Fallback : évaluation Neural Network
            encoded = encode_board(board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, nn_value = self.neural_net(encoded)
            return nn_value.item()

    def search(self, state: chess.Board) -> "StockfishMCTSNode":
        """
        Lance une simulation MCTS hybride.

        Utilise :
        - Neural Network pour guider la sélection (politique)
        - Stockfish pour évaluer les positions (valeur)
        """
        # Créer la racine si première simulation
        if self.root is None or not self.root.board.board_fen() == state.board_fen():
            self.root = StockfishMCTSNode(state)

        node = self.root
        path = []  # Chemin traversé

        # 1. SÉLECTION : descendre dans l'arbre
        while not node.is_terminal and node.is_expanded:
            move, node = node.select_child(self.c_puct)
            path.append(node)

        # 2. EXPANSION et 3. ÉVALUATION
        if node.is_terminal:
            # Nœud terminal : valeur exacte
            value = node.get_terminal_value()
        else:
            # 🚀 HYBRIDE : NN pour politique, Stockfish pour valeur
            move_probs, stockfish_value = self.predict(node.board)
            value = stockfish_value

            # Expansion si pas encore fait
            if not node.is_expanded:
                node.expand(move_probs)

        # 4. BACKPROPAGATION : remonter la valeur Stockfish
        for ancestor in reversed(path):
            ancestor.visit_count += 1
            ancestor.value_sum += value
            # Inverser la valeur car on change de joueur
            value = -value

        return self.root

    def run(
        self, board: chess.Board, simulations: int = 800
    ) -> Dict[chess.Move, float]:
        """
        Lance les simulations MCTS hybrides.

        Args:
            board: Position d'échecs
            simulations: Nombre de simulations

        Returns:
            Distribution de probabilités sur les coups
        """
        print(f"🎯 MCTS Hybride: {simulations} simulations (NN+Stockfish)")

        # Réinitialiser pour nouvelle position
        if self.root is None or not self.root.board.board_fen() == board.board_fen():
            self.root = None
            self.evaluation_cache.clear()  # Nettoyer le cache

        # Lancer les simulations
        for i in range(simulations):
            self.search(board)

            # Affichage du progrès
            if (i + 1) % 100 == 0:
                print(f"   Simulation {i+1}/{simulations}")

        # Calculer la distribution finale
        if self.root is None:
            return {}

        visit_counts = {}
        total_visits = 0

        for move, child in self.root.children.items():
            visit_counts[move] = child.visit_count
            total_visits += child.visit_count

        if total_visits == 0:
            # Fallback : distribution uniforme
            legal_moves = list(board.legal_moves)
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 1.0
            return {move: uniform_prob for move in legal_moves}

        # Normaliser en probabilités
        move_probs = {
            move: visits / total_visits for move, visits in visit_counts.items()
        }

        # Afficher les meilleurs coups
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        print(f"🏆 Top 3 coups MCTS Hybride:")
        for i, (move, prob) in enumerate(sorted_moves[:3]):
            print(f"   {i+1}. {move}: {prob:.3f}")

        return move_probs

    def get_cache_stats(self) -> Dict[str, int]:
        """Statistiques du cache d'évaluations."""
        return {
            "cache_size": len(self.evaluation_cache),
            "cache_hits": getattr(self, "_cache_hits", 0),
            "cache_misses": getattr(self, "_cache_misses", 0),
        }


def create_stockfish_mcts(
    neural_net, stockfish_evaluator, c_puct: float = 1.4, device: str = "cpu"
):
    """
    Crée un MCTS hybride avec Stockfish.

    Args:
        neural_net: Réseau de neurones pour les politiques
        stockfish_evaluator: Évaluateur Stockfish pour les valeurs
        c_puct: Paramètre d'exploration
        device: Device de calcul

    Returns:
        Instance de StockfishMCTS
    """
    return StockfishMCTS(neural_net, stockfish_evaluator, c_puct, device)


if __name__ == "__main__":
    # Test du MCTS hybride
    print("🧪 Test du MCTS Hybride")

    from .network import ChessNet
    from .reference_evaluator import get_reference_evaluator

    # Créer les composants
    neural_net = ChessNet()
    stockfish_evaluator = get_reference_evaluator()

    # Créer le MCTS hybride
    hybrid_mcts = create_stockfish_mcts(neural_net, stockfish_evaluator)

    # Test sur position initiale
    board = chess.Board()
    move_probs = hybrid_mcts.run(board, simulations=100)

    print(f"✅ Test réussi ! {len(move_probs)} coups évalués")
    print(f"📊 Cache: {hybrid_mcts.get_cache_stats()}")
