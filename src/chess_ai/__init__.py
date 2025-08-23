"""
Chess AI - Modélisation professionnelle d'un plateau d'échecs avec interface graphique.

Ce package fournit une interface robuste et extensible pour la gestion
d'un plateau d'échecs utilisant la librairie python-chess, avec une
interface graphique moderne utilisant Pygame.

Version: 1.0.0
Author: Chess AI Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Chess AI Team"
__email__ = "contact@chessai.dev"

from .core.environment import ChessEnvironment
from .core.analyzer import ChessAnalyzer
from .core.display import ChessDisplay
from .exceptions import *

# Import conditionnel de l'interface graphique
try:
    from .gui.chess_gui import ChessGUI

    GUI_AVAILABLE = True
except ImportError:
    # Pygame non disponible
    ChessGUI = None
    GUI_AVAILABLE = False

__all__ = [
    "ChessEnvironment",
    "ChessAnalyzer",
    "ChessDisplay",
    "ChessGUI",
    "GUI_AVAILABLE",
    "ChessError",
    "InvalidMoveError",
    "InvalidSquareError",
    "GameOverError",
]


def launch_gui(environment=None):
    """
    Lance l'interface graphique si pygame est disponible.

    Args:
        environment: Environnement Chess AI optionnel

    Raises:
        ImportError: Si pygame n'est pas installé
    """
    if not GUI_AVAILABLE:
        raise ImportError(
            "L'interface graphique nécessite pygame. "
            "Installez-le avec: pip install pygame"
        )

    gui = ChessGUI(environment)
    gui.run()
