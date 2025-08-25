"""
Module d'IA AlphaZero pour Chess AI
==================================

Ce module contient l'implémentation d'un réseau de neurones
de type AlphaZero pour jouer aux échecs.
"""

from .network import ChessNet, encode_board, decode_policy

__all__ = ["ChessNet", "encode_board", "decode_policy"]
