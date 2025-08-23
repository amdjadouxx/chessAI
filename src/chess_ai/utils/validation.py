"""
Fonctions utilitaires de validation pour Chess AI.

Ce module contient toutes les fonctions de validation des entrées
pour assurer la robustesse de l'application.
"""

from typing import Union
import chess
from chess import Square, Move


def validate_square(square: Union[str, int, Square]) -> Square:
    """
    Valide et convertit une case en objet chess.Square.

    Args:
        square: Case sous différents formats

    Returns:
        Objet chess.Square validé

    Raises:
        ValueError: Si la case est invalide
    """
    if isinstance(square, str):
        try:
            return chess.parse_square(square)
        except ValueError:
            raise ValueError(
                f"Notation de case invalide: '{square}'. "
                "Utilisez la notation algébrique (ex: 'e4')"
            )

    elif isinstance(square, int):
        if not (0 <= square <= 63):
            raise ValueError(
                f"Indice de case invalide: {square}. " "Doit être entre 0 et 63"
            )
        return Square(square)

    elif isinstance(square, Square):
        return square

    else:
        raise ValueError(
            f"Type de case non supporté: {type(square)}. "
            "Utilisez str, int ou chess.Square"
        )


def validate_move(move: Union[str, Move], board: chess.Board) -> Move:
    """
    Valide et convertit un mouvement en objet chess.Move.

    Args:
        move: Mouvement sous différents formats
        board: Plateau pour validation contextuelle

    Returns:
        Objet chess.Move validé

    Raises:
        ValueError: Si le mouvement est invalide
    """
    if isinstance(move, str):
        try:
            # Essayer d'abord UCI
            if len(move) >= 4:
                return Move.from_uci(move)
            else:
                raise ValueError("Format UCI trop court")
        except ValueError:
            # Essayer SAN (Standard Algebraic Notation)
            try:
                return board.parse_san(move)
            except ValueError:
                raise ValueError(
                    f"Format de mouvement invalide: '{move}'. "
                    "Utilisez la notation UCI (ex: 'e2e4') ou SAN (ex: 'e4')"
                )

    elif isinstance(move, Move):
        return move

    else:
        raise ValueError(
            f"Type de mouvement non supporté: {type(move)}. "
            "Utilisez str ou chess.Move"
        )


def validate_fen(fen: str) -> bool:
    """
    Valide une notation FEN.

    Args:
        fen: Notation FEN à valider

    Returns:
        True si la FEN est valide

    Raises:
        ValueError: Si la FEN est invalide
    """
    try:
        board = chess.Board(fen)
        return True
    except ValueError as e:
        raise ValueError(f"Notation FEN invalide: '{fen}'. Erreur: {e}")


def validate_color(color: Union[str, bool, chess.Color]) -> chess.Color:
    """
    Valide et convertit une couleur en objet chess.Color.

    Args:
        color: Couleur sous différents formats

    Returns:
        chess.WHITE ou chess.BLACK

    Raises:
        ValueError: Si la couleur est invalide
    """
    if isinstance(color, bool):
        return chess.WHITE if color else chess.BLACK

    elif isinstance(color, str):
        color_lower = color.lower()
        if color_lower in ["white", "blanc", "w"]:
            return chess.WHITE
        elif color_lower in ["black", "noir", "b"]:
            return chess.BLACK
        else:
            raise ValueError(
                f"Couleur invalide: '{color}'. "
                "Utilisez 'white'/'blanc' ou 'black'/'noir'"
            )

    elif isinstance(color, chess.Color):
        return color

    else:
        raise ValueError(f"Type de couleur non supporté: {type(color)}")


def validate_piece_type(
    piece_type: Union[str, int, chess.PieceType],
) -> chess.PieceType:
    """
    Valide et convertit un type de pièce.

    Args:
        piece_type: Type de pièce sous différents formats

    Returns:
        Objet chess.PieceType validé

    Raises:
        ValueError: Si le type de pièce est invalide
    """
    if isinstance(piece_type, str):
        piece_map = {
            "p": chess.PAWN,
            "pawn": chess.PAWN,
            "pion": chess.PAWN,
            "n": chess.KNIGHT,
            "knight": chess.KNIGHT,
            "cavalier": chess.KNIGHT,
            "b": chess.BISHOP,
            "bishop": chess.BISHOP,
            "fou": chess.BISHOP,
            "r": chess.ROOK,
            "rook": chess.ROOK,
            "tour": chess.ROOK,
            "q": chess.QUEEN,
            "queen": chess.QUEEN,
            "dame": chess.QUEEN,
            "k": chess.KING,
            "king": chess.KING,
            "roi": chess.KING,
        }

        piece_lower = piece_type.lower()
        if piece_lower in piece_map:
            return piece_map[piece_lower]
        else:
            raise ValueError(f"Type de pièce invalide: '{piece_type}'")

    elif isinstance(piece_type, int):
        if piece_type in chess.PIECE_TYPES:
            return chess.PieceType(piece_type)
        else:
            raise ValueError(f"Type de pièce numérique invalide: {piece_type}")

    elif isinstance(piece_type, chess.PieceType):
        return piece_type

    else:
        raise ValueError(f"Type de pièce non supporté: {type(piece_type)}")


def is_valid_square_name(square_name: str) -> bool:
    """
    Vérifie si un nom de case est valide.

    Args:
        square_name: Nom de case à vérifier

    Returns:
        True si le nom de case est valide
    """
    try:
        validate_square(square_name)
        return True
    except ValueError:
        return False


def is_valid_move_format(move_str: str) -> bool:
    """
    Vérifie si un format de mouvement est potentiellement valide.

    Args:
        move_str: String de mouvement à vérifier

    Returns:
        True si le format semble valide
    """
    if not isinstance(move_str, str):
        return False

    # UCI format (au moins 4 caractères)
    if len(move_str) >= 4:
        return True

    # SAN format basique
    if len(move_str) >= 2:
        return True

    return False
