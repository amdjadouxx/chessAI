"""
Exceptions personnalisées pour Chess AI.

Ce module définit toutes les exceptions spécifiques à l'application
pour une gestion d'erreurs robuste et informative.
"""

from typing import Optional


class ChessError(Exception):
    """Exception de base pour toutes les erreurs liées aux échecs."""

    def __init__(self, message: str, details: Optional[str] = None):
        """
        Initialise une exception Chess.

        Args:
            message: Message principal de l'erreur
            details: Détails supplémentaires optionnels
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        """Retourne une représentation string de l'erreur."""
        if self.details:
            return f"{self.message} - Détails: {self.details}"
        return self.message


class InvalidMoveError(ChessError):
    """Exception levée lors d'une tentative de mouvement invalide."""

    def __init__(self, move: str, reason: str):
        """
        Initialise une exception de mouvement invalide.

        Args:
            move: Le mouvement qui a causé l'erreur
            reason: La raison pour laquelle le mouvement est invalide
        """
        message = f"Mouvement invalide: {move}"
        super().__init__(message, reason)
        self.move = move
        self.reason = reason


class InvalidSquareError(ChessError):
    """Exception levée lors de l'accès à une case invalide."""

    def __init__(self, square: str):
        """
        Initialise une exception de case invalide.

        Args:
            square: La notation de case qui a causé l'erreur
        """
        message = f"Case invalide: {square}"
        details = (
            "La case doit être en notation algébrique (ex: 'e4') ou un indice 0-63"
        )
        super().__init__(message, details)
        self.square = square


class GameOverError(ChessError):
    """Exception levée lors d'une tentative d'action sur une partie terminée."""

    def __init__(self, action: str, result: str):
        """
        Initialise une exception de partie terminée.

        Args:
            action: L'action qui a été tentée
            result: Le résultat de la partie
        """
        message = f"Impossible d'effectuer '{action}': la partie est terminée"
        details = f"Résultat de la partie: {result}"
        super().__init__(message, details)
        self.action = action
        self.result = result


class ChessBoardStateError(ChessError):
    """Exception levée pour les erreurs d'état du plateau."""

    def __init__(self, state_issue: str):
        """
        Initialise une exception d'état du plateau.

        Args:
            state_issue: Description du problème d'état
        """
        message = f"Erreur d'état du plateau: {state_issue}"
        super().__init__(message)
        self.state_issue = state_issue


class InvalidFENError(ChessError):
    """Exception levée pour une notation FEN invalide."""

    def __init__(self, fen: str, parsing_error: str):
        """
        Initialise une exception FEN invalide.

        Args:
            fen: La notation FEN invalide
            parsing_error: L'erreur de parsing
        """
        message = f"Notation FEN invalide: {fen}"
        super().__init__(message, parsing_error)
        self.fen = fen
        self.parsing_error = parsing_error


class ChessConfigurationError(ChessError):
    """Exception levée pour les erreurs de configuration."""

    def __init__(self, config_issue: str):
        """
        Initialise une exception de configuration.

        Args:
            config_issue: Description du problème de configuration
        """
        message = f"Erreur de configuration: {config_issue}"
        super().__init__(message)
        self.config_issue = config_issue
