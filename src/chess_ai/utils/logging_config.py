"""
Configuration du logging pour Chess AI.

Ce module centralise la configuration du système de logging
pour une traçabilité complète des opérations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure le système de logging global.

    Args:
        level: Niveau de log ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Fichier de log optionnel
        format_string: Format personnalisé des messages
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    # Configuration de base
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)

    # Handler pour fichier si spécifié
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Toujours DEBUG pour les fichiers
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)

        # Ajouter aux loggers
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Obtient un logger configuré pour un module.

    Args:
        name: Nom du logger (généralement __name__)
        level: Niveau de log optionnel pour ce logger spécifique

    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)

    if level:
        logger.setLevel(getattr(logging, level.upper()))

    return logger


def log_function_call(func):
    """
    Décorateur pour logger automatiquement les appels de fonction.

    Args:
        func: Fonction à décorer

    Returns:
        Fonction décorée
    """

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = func.__name__

        logger.debug(f"Appel de {func_name} avec args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func_name} terminé avec succès")
            return result
        except Exception as e:
            logger.error(f"Erreur dans {func_name}: {e}")
            raise

    return wrapper


def log_performance(func):
    """
    Décorateur pour mesurer et logger les performances.

    Args:
        func: Fonction à décorer

    Returns:
        Fonction décorée
    """

    def wrapper(*args, **kwargs):
        import time

        logger = get_logger(func.__module__)
        func_name = func.__name__

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func_name} exécuté en {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func_name} a échoué après {execution_time:.4f}s: {e}")
            raise

    return wrapper


class ChessLogger:
    """
    Logger spécialisé pour les opérations d'échecs.
    """

    def __init__(self, name: str, level: str = "INFO"):
        """
        Initialise le logger Chess.

        Args:
            name: Nom du logger
            level: Niveau de log
        """
        self.logger = get_logger(name, level)

    def log_move(self, move: str, success: bool, details: str = ""):
        """
        Log un mouvement d'échecs.

        Args:
            move: Mouvement effectué
            success: Si le mouvement a réussi
            details: Détails supplémentaires
        """
        if success:
            self.logger.info(f"✓ Mouvement: {move} {details}")
        else:
            self.logger.warning(f"✗ Mouvement échoué: {move} {details}")

    def log_game_state(self, fen: str, player: str, check: bool = False):
        """
        Log l'état du jeu.

        Args:
            fen: Position FEN
            player: Joueur actuel
            check: Si en échec
        """
        check_status = " [ÉCHEC]" if check else ""
        self.logger.info(f"État: {player} à jouer{check_status} | FEN: {fen}")

    def log_game_over(self, result: str, reason: str = ""):
        """
        Log la fin de partie.

        Args:
            result: Résultat de la partie
            reason: Raison de la fin
        """
        self.logger.info(f"🏁 Partie terminée: {result} {reason}")


# Configuration par défaut
def configure_default_logging():
    """Configure le logging par défaut pour Chess AI."""
    setup_logging(
        level="INFO",
        log_file=None,  # Pas de fichier par défaut
        format_string="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
