"""
Interface graphique moderne pour Chess AI.

Ce module fournit une interface graphique 3D interactive utilisant Pygame
pour une expérience utilisateur moderne et fluide avec IA AlphaZero.
"""

from .chess_gui_3d import SimpleChessGUI3D
from .ai_integration import AlphaZeroPlayer

__all__ = ["SimpleChessGUI3D", "AlphaZeroPlayer"]
