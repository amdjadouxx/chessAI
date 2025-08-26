"""
Module d'IA AlphaZero pour Chess AI
==================================

Ce module contient l'implémentation d'un réseau de neurones
de type AlphaZero, l'algorithme MCTS et le système d'entraînement
par auto-jeu pour jouer aux échecs.
"""

from .network import ChessNet, encode_board, decode_policy
from .mcts import MCTS, MCTSNode

# Import conditionnel du module d'entraînement
try:
    from .training import (
        SelfPlayTrainer,
        TrainingConfig,
        quick_training_session,
        continue_training,
    )

    __all__ = [
        "ChessNet",
        "encode_board",
        "decode_policy",
        "MCTS",
        "MCTSNode",
        "SelfPlayTrainer",
        "TrainingConfig",
        "quick_training_session",
        "continue_training",
    ]
except ImportError:
    __all__ = ["ChessNet", "encode_board", "decode_policy", "MCTS", "MCTSNode"]
