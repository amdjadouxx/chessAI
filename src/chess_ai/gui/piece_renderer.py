"""
Rendu des pièces d'échecs pour l'interface graphique.

Ce module gère le rendu visuel des pièces avec support
pour différents styles et animations.
"""

import pygame
import math
from typing import Dict, Tuple, Optional
import os

import chess
from .vector_renderer import VectorPieceRenderer
from .image_renderer import ImagePieceRenderer


class PieceRenderer:
    """
    Gestionnaire de rendu pour les pièces d'échecs.

    Fonctionnalités :
    - Rendu des pièces avec symboles Unicode et vectoriel
    - Support pour différents styles
    - Animations de mouvement
    - Cache de surfaces pour les performances
    """

    # Symboles Unicode des pièces
    PIECE_SYMBOLS = {
        chess.PAWN: {"white": "♙", "black": "♟"},
        chess.ROOK: {"white": "♖", "black": "♜"},
        chess.KNIGHT: {"white": "♘", "black": "♞"},
        chess.BISHOP: {"white": "♗", "black": "♝"},
        chess.QUEEN: {"white": "♕", "black": "♛"},
        chess.KING: {"white": "♔", "black": "♚"},
    }

    # Couleurs par défaut
    DEFAULT_COLORS = {
        "white_piece": (255, 255, 255),
        "black_piece": (50, 50, 50),
        "piece_outline": (0, 0, 0),
        "piece_shadow": (128, 128, 128),
    }

    def __init__(self, square_size: int, style: str = "vector"):
        """
        Initialise le renderer de pièces.

        Args:
            square_size: Taille d'une case en pixels
            style: Style de rendu ('unicode', 'vector', 'minimal', 'image')
        """
        self.square_size = square_size
        self.style = style

        # Cache des surfaces de pièces
        self._piece_cache: Dict[str, pygame.Surface] = {}

        # Configuration des fonts
        pygame.font.init()
        piece_size = int(square_size * 0.8)  # 80% de la case
        self.piece_font = pygame.font.Font(None, piece_size)

        # Couleurs
        self.colors = self.DEFAULT_COLORS.copy()

        # Renderers
        self.vector_renderer = VectorPieceRenderer(square_size)
        self.image_renderer = ImagePieceRenderer(square_size)

        # Pré-calculer toutes les pièces
        self._precalculate_pieces()

    def _precalculate_pieces(self) -> None:
        """Pré-calcule toutes les surfaces de pièces pour les performances."""
        for piece_type in chess.PIECE_TYPES:
            for color in [chess.WHITE, chess.BLACK]:
                piece = chess.Piece(piece_type, color)
                self._get_piece_surface(piece)

    def _get_piece_surface(self, piece: chess.Piece) -> pygame.Surface:
        """
        Récupère la surface d'une pièce (avec cache).

        Args:
            piece: Pièce à rendre

        Returns:
            Surface pygame de la pièce
        """
        cache_key = f"{self.style}_{piece.piece_type}_{piece.color}"

        if cache_key not in self._piece_cache:
            self._piece_cache[cache_key] = self._create_piece_surface(piece)

        return self._piece_cache[cache_key]

    def _create_piece_surface(self, piece: chess.Piece) -> pygame.Surface:
        """
        Crée la surface d'une pièce selon le style configuré.

        Args:
            piece: Pièce à créer

        Returns:
            Surface pygame de la pièce
        """
        if self.style == "image":
            # Essayer d'abord l'image, sinon fallback sur vectoriel
            image_surface = self.image_renderer.get_piece_surface(piece)
            if image_surface is not None:
                return image_surface
            else:
                # Fallback sur vectoriel si l'image n'existe pas
                return self.vector_renderer.get_piece_surface(piece)
        elif self.style == "vector":
            return self.vector_renderer.get_piece_surface(piece)
        elif self.style == "unicode":
            return self._create_unicode_piece(piece)
        elif self.style == "minimal":
            return self._create_minimal_piece(piece)
        else:
            # Style par défaut = vector
            return self.vector_renderer.get_piece_surface(piece)

    def _create_unicode_piece(self, piece: chess.Piece) -> pygame.Surface:
        """Crée une pièce avec symboles Unicode."""
        surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)

        # Symbole de la pièce
        color_key = "white" if piece.color == chess.WHITE else "black"
        symbol = self.PIECE_SYMBOLS[piece.piece_type][color_key]

        # Couleur de la pièce - amélioration pour plus de contraste
        if piece.color == chess.WHITE:
            piece_color = (255, 255, 255)  # Blanc pur
            outline_color = (0, 0, 0)  # Contour noir
            shadow_color = (128, 128, 128)  # Ombre grise
        else:
            piece_color = (30, 30, 30)  # Noir foncé
            outline_color = (255, 255, 255)  # Contour blanc
            shadow_color = (100, 100, 100)  # Ombre plus claire

        # Taille de police plus grande pour de meilleures pièces
        piece_font_size = int(self.square_size * 0.9)  # 90% de la case
        piece_font = pygame.font.Font(None, piece_font_size)

        # Rendu avec ombre pour l'effet 3D
        shadow_offset = 3

        # Ombre portée
        shadow_text = piece_font.render(symbol, True, shadow_color)
        shadow_rect = shadow_text.get_rect(
            center=(
                self.square_size // 2 + shadow_offset,
                self.square_size // 2 + shadow_offset,
            )
        )
        surface.blit(shadow_text, shadow_rect)

        # Contour épais pour la lisibilité
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx != 0 or dy != 0:  # Skip le centre
                    outline_text = piece_font.render(symbol, True, outline_color)
                    outline_rect = shadow_text.get_rect(
                        center=(self.square_size // 2 + dx, self.square_size // 2 + dy)
                    )
                    surface.blit(outline_text, outline_rect)

        # Pièce principale par-dessus
        piece_text = piece_font.render(symbol, True, piece_color)
        piece_rect = piece_text.get_rect(
            center=(self.square_size // 2, self.square_size // 2)
        )
        surface.blit(piece_text, piece_rect)

        return surface

    def _create_minimal_piece(self, piece: chess.Piece) -> pygame.Surface:
        """Crée une pièce avec style minimal (formes géométriques)."""
        surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)

        center = self.square_size // 2
        radius = self.square_size // 4

        # Couleur de base
        color = (
            self.colors["white_piece"]
            if piece.color == chess.WHITE
            else self.colors["black_piece"]
        )

        # Formes selon le type de pièce
        if piece.piece_type == chess.PAWN:
            pygame.draw.circle(surface, color, (center, center), radius)

        elif piece.piece_type == chess.ROOK:
            rect = pygame.Rect(center - radius, center - radius, radius * 2, radius * 2)
            pygame.draw.rect(surface, color, rect)

        elif piece.piece_type == chess.KNIGHT:
            # Triangle pour le cavalier
            points = [
                (center, center - radius),
                (center - radius, center + radius),
                (center + radius, center + radius),
            ]
            pygame.draw.polygon(surface, color, points)

        elif piece.piece_type == chess.BISHOP:
            # Losange pour le fou
            points = [
                (center, center - radius),
                (center + radius, center),
                (center, center + radius),
                (center - radius, center),
            ]
            pygame.draw.polygon(surface, color, points)

        elif piece.piece_type == chess.QUEEN:
            # Étoile pour la dame
            pygame.draw.circle(surface, color, (center, center), radius)
            for angle in range(0, 360, 45):
                x = center + int(radius * 1.2 * math.cos(math.radians(angle)))
                y = center + int(radius * 1.2 * math.sin(math.radians(angle)))
                pygame.draw.line(surface, color, (center, center), (x, y), 3)

        elif piece.piece_type == chess.KING:
            # Cercle avec croix pour le roi
            pygame.draw.circle(surface, color, (center, center), radius)
            pygame.draw.line(
                surface,
                color,
                (center, center - radius // 2),
                (center, center + radius // 2),
                3,
            )
            pygame.draw.line(
                surface,
                color,
                (center - radius // 2, center),
                (center + radius // 2, center),
                3,
            )

        # Bordure
        pygame.draw.circle(
            surface, self.colors["piece_outline"], (center, center), radius, 2
        )

        return surface

    def render_piece(
        self,
        screen: pygame.Surface,
        piece: chess.Piece,
        x: int,
        y: int,
        alpha: int = 255,
    ) -> None:
        """
        Rend une pièce à la position donnée.

        Args:
            screen: Surface de destination
            piece: Pièce à rendre
            x, y: Position de rendu
            alpha: Transparence (0-255)
        """
        piece_surface = self._get_piece_surface(piece)

        if alpha < 255:
            # Créer une surface temporaire avec alpha
            temp_surface = piece_surface.copy()
            temp_surface.set_alpha(alpha)
            screen.blit(temp_surface, (x, y))
        else:
            screen.blit(piece_surface, (x, y))

    def render_piece_with_shadow(
        self,
        screen: pygame.Surface,
        piece: chess.Piece,
        x: int,
        y: int,
        shadow_offset: int = 3,
    ) -> None:
        """
        Rend une pièce avec ombre portée.

        Args:
            screen: Surface de destination
            piece: Pièce à rendre
            x, y: Position de rendu
            shadow_offset: Décalage de l'ombre en pixels
        """
        # Ombre
        shadow_surface = self._get_piece_surface(piece).copy()
        shadow_surface.fill(
            self.colors["piece_shadow"], special_flags=pygame.BLEND_MULT
        )
        shadow_surface.set_alpha(128)
        screen.blit(shadow_surface, (x + shadow_offset, y + shadow_offset))

        # Pièce principale
        self.render_piece(screen, piece, x, y)

    def render_animated_piece(
        self,
        screen: pygame.Surface,
        piece: chess.Piece,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        progress: float,
    ) -> None:
        """
        Rend une pièce en cours d'animation.

        Args:
            screen: Surface de destination
            piece: Pièce à animer
            start_pos: Position de départ
            end_pos: Position d'arrivée
            progress: Progression de l'animation (0.0 à 1.0)
        """
        # Interpolation linéaire
        current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

        # Effet de "saut" pour rendre l'animation plus dynamique
        jump_height = 20 * math.sin(progress * math.pi)
        current_y -= jump_height

        # Rendu avec légère transparence pendant l'animation
        alpha = int(255 * (0.8 + 0.2 * (1 - abs(progress - 0.5) * 2)))
        self.render_piece(screen, piece, int(current_x), int(current_y), alpha)

    def render_ghost_piece(
        self, screen: pygame.Surface, piece: chess.Piece, x: int, y: int
    ) -> None:
        """
        Rend une pièce "fantôme" (transparente) pour prévisualiser un mouvement.

        Args:
            screen: Surface de destination
            piece: Pièce à rendre
            x, y: Position de rendu
        """
        self.render_piece(screen, piece, x, y, alpha=100)

    def update_colors(self, color_scheme: Dict[str, Tuple[int, int, int]]) -> None:
        """
        Met à jour le schéma de couleurs et invalide le cache.

        Args:
            color_scheme: Nouveau schéma de couleurs
        """
        self.colors.update(color_scheme)
        self._piece_cache.clear()
        self._precalculate_pieces()

    def set_style(self, style: str) -> None:
        """
        Change le style de rendu des pièces.

        Args:
            style: Nouveau style ('unicode', 'vector', 'minimal', 'image')
        """
        if style != self.style:
            self.style = style
            self._piece_cache.clear()
            self._precalculate_pieces()

    def reload_images(self) -> None:
        """Recharge les images personnalisées depuis le dossier assets."""
        if hasattr(self, "image_renderer"):
            self.image_renderer.reload_images()
            if self.style == "image":
                self._piece_cache.clear()
                self._precalculate_pieces()
                print("🔄 Images rechargées")

    def get_available_images(self) -> Dict[str, str]:
        """Retourne la liste des images disponibles."""
        if hasattr(self, "image_renderer"):
            return self.image_renderer.get_available_images()
        return {}

    def get_piece_at_pos(
        self, pos: Tuple[int, int], board_offset: Tuple[int, int] = (0, 0)
    ) -> Optional[Tuple[int, int]]:
        """
        Détermine quelle case correspond à une position écran.

        Args:
            pos: Position de la souris
            board_offset: Décalage du plateau sur l'écran

        Returns:
            Coordonnées de la case (file, rang) ou None
        """
        x, y = pos
        board_x, board_y = board_offset

        # Ajuster par rapport au plateau
        rel_x = x - board_x
        rel_y = y - board_y

        # Vérifier si dans les limites du plateau
        if 0 <= rel_x < self.square_size * 8 and 0 <= rel_y < self.square_size * 8:
            file = rel_x // self.square_size
            rank = 7 - (rel_y // self.square_size)  # Inverser le rang
            return (file, rank)

        return None
