"""
Rendu vectoriel avancé des pièces d'échecs.

Ce module fournit un rendu vectoriel des pièces pour une qualité
visuelle supérieure, avec des formes géométriques stylisées.
"""

import pygame
import math
from typing import Dict, Tuple, List
import chess


class VectorPieceRenderer:
    """
    Renderer vectoriel pour des pièces d'échecs de haute qualité.

    Utilise des formes géométriques pour créer des pièces
    visuellement attractives et facilement reconnaissables.
    """

    def __init__(self, square_size: int):
        """
        Initialise le renderer vectoriel.

        Args:
            square_size: Taille d'une case en pixels
        """
        self.square_size = square_size
        self.piece_size = int(square_size * 0.8)  # 80% de la case
        self.center = square_size // 2

        # Couleurs améliorées avec plus de contraste et d'élégance
        self.colors = {
            "white_fill": (250, 248, 240),  # Blanc cassé élégant
            "white_outline": (40, 40, 40),  # Gris très foncé
            "white_shadow": (180, 180, 180),  # Ombre douce
            "black_fill": (45, 45, 45),  # Noir élégant
            "black_outline": (20, 20, 20),  # Noir profond
            "black_shadow": (80, 80, 80),  # Ombre plus marquée
            "highlight": (220, 180, 0),  # Or plus chaud
            "accent": (180, 140, 60),  # Bronze pour les détails
        }

        # Cache des surfaces
        self._vector_cache: Dict[str, pygame.Surface] = {}

    def get_piece_surface(self, piece: chess.Piece) -> pygame.Surface:
        """Récupère la surface vectorielle d'une pièce (avec cache)."""
        cache_key = f"vector_{piece.piece_type}_{piece.color}"

        if cache_key not in self._vector_cache:
            self._vector_cache[cache_key] = self._create_vector_piece(piece)

        return self._vector_cache[cache_key]

    def _create_vector_piece(self, piece: chess.Piece) -> pygame.Surface:
        """Crée une pièce vectorielle."""
        surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)

        # Couleurs selon la pièce
        if piece.color == chess.WHITE:
            fill_color = self.colors["white_fill"]
            outline_color = self.colors["white_outline"]
            shadow_color = self.colors["white_shadow"]
        else:
            fill_color = self.colors["black_fill"]
            outline_color = self.colors["black_outline"]
            shadow_color = self.colors["black_shadow"]

        # Rendu selon le type de pièce
        if piece.piece_type == chess.PAWN:
            self._draw_pawn(surface, fill_color, outline_color, shadow_color)
        elif piece.piece_type == chess.ROOK:
            self._draw_rook(surface, fill_color, outline_color, shadow_color)
        elif piece.piece_type == chess.KNIGHT:
            self._draw_knight(surface, fill_color, outline_color, shadow_color)
        elif piece.piece_type == chess.BISHOP:
            self._draw_bishop(surface, fill_color, outline_color, shadow_color)
        elif piece.piece_type == chess.QUEEN:
            self._draw_queen(surface, fill_color, outline_color, shadow_color)
        elif piece.piece_type == chess.KING:
            self._draw_king(surface, fill_color, outline_color, shadow_color)

        return surface

    def _draw_pawn(
        self,
        surface: pygame.Surface,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        shadow: Tuple[int, int, int],
    ) -> None:
        """Dessine un pion élégant."""
        shadow_offset = 2

        # Base élargie et plus stable
        base_width = self.piece_size // 3
        base_height = self.piece_size // 8
        base_rect = pygame.Rect(
            self.center - base_width // 2,
            self.center + self.piece_size // 3,
            base_width,
            base_height,
        )

        # Ombre de la base
        shadow_rect = base_rect.move(shadow_offset, shadow_offset)
        pygame.draw.ellipse(surface, shadow, shadow_rect)

        # Base principale
        pygame.draw.ellipse(surface, fill, base_rect)
        pygame.draw.ellipse(surface, outline, base_rect, 2)

        # Corps élancé
        body_width = self.piece_size // 6
        body_height = self.piece_size // 2
        body_rect = pygame.Rect(
            self.center - body_width // 2,
            self.center - body_height // 4,
            body_width,
            body_height,
        )

        # Ombre du corps
        shadow_rect = body_rect.move(shadow_offset, shadow_offset)
        pygame.draw.ellipse(surface, shadow, shadow_rect)

        # Corps principal
        pygame.draw.ellipse(surface, fill, body_rect)
        pygame.draw.ellipse(surface, outline, body_rect, 2)

        # Tête sphérique plus grande
        head_radius = self.piece_size // 5
        head_center = (self.center, self.center - self.piece_size // 4)

        # Ombre de la tête
        shadow_center = (head_center[0] + shadow_offset, head_center[1] + shadow_offset)
        pygame.draw.circle(surface, shadow, shadow_center, head_radius + 1)

        # Tête principale
        pygame.draw.circle(surface, fill, head_center, head_radius)
        pygame.draw.circle(surface, outline, head_center, head_radius, 2)

        # Petit détail au sommet
        pygame.draw.circle(
            surface,
            self.colors["highlight"],
            (head_center[0], head_center[1] - head_radius // 3),
            2,
        )

    def _draw_rook(
        self,
        surface: pygame.Surface,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        shadow: Tuple[int, int, int],
    ) -> None:
        """Dessine une tour majestueuse."""
        shadow_offset = 2

        # Base large et solide
        base_width = self.piece_size // 2
        base_height = self.piece_size // 6
        base_rect = pygame.Rect(
            self.center - base_width // 2,
            self.center + self.piece_size // 4,
            base_width,
            base_height,
        )

        # Ombre de la base
        shadow_rect = base_rect.move(shadow_offset, shadow_offset)
        pygame.draw.rect(surface, shadow, shadow_rect)

        # Base avec coins arrondis
        pygame.draw.rect(surface, fill, base_rect, border_radius=4)
        pygame.draw.rect(surface, outline, base_rect, 2, border_radius=4)

        # Corps principal trapézoïdal
        tower_bottom = self.piece_size // 2
        tower_top = self.piece_size // 3
        tower_height = self.piece_size // 2

        # Points du trapèze
        points = [
            (self.center - tower_bottom // 2, self.center + tower_height // 4),
            (self.center + tower_bottom // 2, self.center + tower_height // 4),
            (self.center + tower_top // 2, self.center - tower_height // 4),
            (self.center - tower_top // 2, self.center - tower_height // 4),
        ]

        # Ombre du corps
        shadow_points = [(x + shadow_offset, y + shadow_offset) for x, y in points]
        pygame.draw.polygon(surface, shadow, shadow_points)

        # Corps principal
        pygame.draw.polygon(surface, fill, points)
        pygame.draw.polygon(surface, outline, points, 3)

        # Créneaux détaillés
        creneau_top = self.center - tower_height // 4
        creneau_width = tower_top // 6
        creneau_height = self.piece_size // 8

        for i in range(5):
            x_offset = self.center - tower_top // 2 + i * (tower_top // 4)
            if i % 2 == 0:  # Créneaux hauts
                creneau_rect = pygame.Rect(
                    x_offset - creneau_width // 2,
                    creneau_top - creneau_height,
                    creneau_width,
                    creneau_height,
                )
                # Ombre
                shadow_rect = creneau_rect.move(shadow_offset, shadow_offset)
                pygame.draw.rect(surface, shadow, shadow_rect)
                # Créneau
                pygame.draw.rect(surface, fill, creneau_rect)
                pygame.draw.rect(surface, outline, creneau_rect, 2)

        # Détails décoratifs
        center_line_y = self.center
        pygame.draw.line(
            surface,
            outline,
            (self.center - tower_top // 3, center_line_y),
            (self.center + tower_top // 3, center_line_y),
            2,
        )

    def _draw_knight(
        self,
        surface: pygame.Surface,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        shadow: Tuple[int, int, int],
    ) -> None:
        """Dessine un cavalier élégant."""
        shadow_offset = 2

        # Silhouette de cheval plus réaliste
        points = [
            # Base
            (self.center - self.piece_size // 3, self.center + self.piece_size // 3),
            (self.center + self.piece_size // 4, self.center + self.piece_size // 3),
            # Poitrail
            (self.center + self.piece_size // 5, self.center + self.piece_size // 6),
            # Encolure
            (self.center + self.piece_size // 4, self.center - self.piece_size // 8),
            # Oreilles
            (self.center + self.piece_size // 6, self.center - self.piece_size // 4),
            (self.center, self.center - self.piece_size // 3),
            # Chanfrein
            (self.center - self.piece_size // 8, self.center - self.piece_size // 4),
            # Museau
            (self.center - self.piece_size // 4, self.center - self.piece_size // 6),
            # Ganache
            (self.center - self.piece_size // 3, self.center),
            # Côté gauche
            (self.center - self.piece_size // 4, self.center + self.piece_size // 4),
        ]

        # Ombre
        shadow_points = [(x + shadow_offset, y + shadow_offset) for x, y in points]
        pygame.draw.polygon(surface, shadow, shadow_points)

        # Corps principal
        pygame.draw.polygon(surface, fill, points)
        pygame.draw.polygon(surface, outline, points, 3)

        # Crinière stylisée
        mane_points = [
            (self.center - self.piece_size // 8, self.center - self.piece_size // 4),
            (self.center - self.piece_size // 6, self.center - self.piece_size // 5),
            (self.center - self.piece_size // 12, self.center - self.piece_size // 6),
            (self.center, self.center - self.piece_size // 4),
        ]
        pygame.draw.lines(surface, outline, False, mane_points, 2)

        # Œil expressif
        eye_pos = (
            self.center - self.piece_size // 10,
            self.center - self.piece_size // 8,
        )
        pygame.draw.circle(surface, outline, eye_pos, 3)
        pygame.draw.circle(surface, (255, 255, 255), eye_pos, 2)
        pygame.draw.circle(surface, outline, eye_pos, 1)

        # Naseaux
        nostril_pos = (
            self.center - self.piece_size // 6,
            self.center - self.piece_size // 8,
        )
        pygame.draw.circle(surface, outline, nostril_pos, 1)

    def _draw_bishop(
        self,
        surface: pygame.Surface,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        shadow: Tuple[int, int, int],
    ) -> None:
        """Dessine un fou."""
        base_radius = self.piece_size // 6
        shadow_offset = 2

        # Base avec ombre
        pygame.draw.circle(
            surface,
            shadow,
            (
                self.center + shadow_offset,
                self.center + self.piece_size // 3 + shadow_offset,
            ),
            base_radius + 2,
        )
        pygame.draw.circle(
            surface,
            fill,
            (self.center, self.center + self.piece_size // 3),
            base_radius,
        )
        pygame.draw.circle(
            surface,
            outline,
            (self.center, self.center + self.piece_size // 3),
            base_radius,
            2,
        )

        # Corps (losange)
        points = [
            (self.center, self.center - self.piece_size // 3),  # Haut
            (self.center + self.piece_size // 4, self.center),  # Droite
            (self.center, self.center + self.piece_size // 6),  # Bas
            (self.center - self.piece_size // 4, self.center),  # Gauche
        ]

        # Ombre du losange
        shadow_points = [(x + shadow_offset, y + shadow_offset) for x, y in points]
        pygame.draw.polygon(surface, shadow, shadow_points)

        # Losange principal
        pygame.draw.polygon(surface, fill, points)
        pygame.draw.polygon(surface, outline, points, 3)

        # Fente caractéristique du fou
        fente_start = (self.center, self.center - self.piece_size // 4)
        fente_end = (self.center, self.center - self.piece_size // 6)
        pygame.draw.line(surface, outline, fente_start, fente_end, 3)

    def _draw_queen(
        self,
        surface: pygame.Surface,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        shadow: Tuple[int, int, int],
    ) -> None:
        """Dessine une dame majestueuse."""
        shadow_offset = 2

        # Base élégante avec gradation
        base_width = self.piece_size // 2
        base_height = self.piece_size // 8
        base_rect = pygame.Rect(
            self.center - base_width // 2,
            self.center + self.piece_size // 3,
            base_width,
            base_height,
        )

        # Ombre de la base
        shadow_rect = base_rect.move(shadow_offset, shadow_offset)
        pygame.draw.ellipse(surface, shadow, shadow_rect)

        # Base principale
        pygame.draw.ellipse(surface, fill, base_rect)
        pygame.draw.ellipse(surface, outline, base_rect, 2)

        # Corps élégant (silhouette féminine)
        body_width_bottom = self.piece_size // 3
        body_width_top = self.piece_size // 4
        body_height = self.piece_size // 2

        # Points pour la silhouette
        body_points = [
            (self.center - body_width_bottom // 2, self.center + self.piece_size // 3),
            (self.center + body_width_bottom // 2, self.center + self.piece_size // 3),
            (self.center + body_width_top // 2, self.center - self.piece_size // 6),
            (self.center - body_width_top // 2, self.center - self.piece_size // 6),
        ]

        # Ombre du corps
        shadow_points = [(x + shadow_offset, y + shadow_offset) for x, y in body_points]
        pygame.draw.polygon(surface, shadow, shadow_points)

        # Corps principal
        pygame.draw.polygon(surface, fill, body_points)
        pygame.draw.polygon(surface, outline, body_points, 2)

        # Couronne sophistiquée (9 pointes)
        crown_center_y = self.center - self.piece_size // 4
        crown_radius = self.piece_size // 3
        crown_points = []

        for i in range(18):  # 9 pointes + 9 creux
            angle = i * math.pi / 9
            if i % 2 == 0:  # Pointes
                radius = crown_radius
            else:  # Creux
                radius = crown_radius * 0.7

            x = self.center + radius * math.cos(angle - math.pi / 2)
            y = crown_center_y + radius * math.sin(angle - math.pi / 2) * 0.6  # Aplatir
            crown_points.append((x, y))

        # Ombre de la couronne
        shadow_crown = [(x + shadow_offset, y + shadow_offset) for x, y in crown_points]
        pygame.draw.polygon(surface, shadow, shadow_crown)

        # Couronne principale
        pygame.draw.polygon(surface, fill, crown_points)
        pygame.draw.polygon(surface, outline, crown_points, 2)

        # Bijou central de la couronne
        jewel_center = (self.center, crown_center_y - crown_radius // 3)
        pygame.draw.circle(surface, self.colors["highlight"], jewel_center, 5)
        pygame.draw.circle(surface, outline, jewel_center, 5, 2)

        # Petits bijoux sur les pointes principales
        for i in [0, 4, 8, 12, 16]:  # Pointes principales
            if i < len(crown_points):
                jewel_pos = crown_points[i]
                pygame.draw.circle(
                    surface,
                    self.colors["accent"],
                    (int(jewel_pos[0]), int(jewel_pos[1])),
                    2,
                )

    def _draw_king(
        self,
        surface: pygame.Surface,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        shadow: Tuple[int, int, int],
    ) -> None:
        """Dessine un roi."""
        base_radius = self.piece_size // 4
        shadow_offset = 2

        # Base avec ombre
        pygame.draw.circle(
            surface,
            shadow,
            (
                self.center + shadow_offset,
                self.center + self.piece_size // 4 + shadow_offset,
            ),
            base_radius + 2,
        )
        pygame.draw.circle(
            surface,
            fill,
            (self.center, self.center + self.piece_size // 4),
            base_radius,
        )
        pygame.draw.circle(
            surface,
            outline,
            (self.center, self.center + self.piece_size // 4),
            base_radius,
            2,
        )

        # Corps
        body_rect = pygame.Rect(
            self.center - base_radius // 2,
            self.center - self.piece_size // 8,
            base_radius,
            self.piece_size // 2,
        )
        shadow_rect = body_rect.move(shadow_offset, shadow_offset)
        pygame.draw.rect(surface, shadow, shadow_rect)
        pygame.draw.rect(surface, fill, body_rect)
        pygame.draw.rect(surface, outline, body_rect, 2)

        # Couronne royale (plus simple que la dame)
        crown_top = self.center - self.piece_size // 3
        crown_width = self.piece_size // 3
        crown_height = self.piece_size // 6

        crown_rect = pygame.Rect(
            self.center - crown_width // 2, crown_top, crown_width, crown_height
        )
        shadow_rect = crown_rect.move(shadow_offset, shadow_offset)
        pygame.draw.rect(surface, shadow, shadow_rect)
        pygame.draw.rect(surface, fill, crown_rect)
        pygame.draw.rect(surface, outline, crown_rect, 2)

        # Croix royale
        cross_size = crown_height // 2
        # Verticale
        pygame.draw.line(
            surface,
            self.colors["highlight"],
            (self.center, crown_top - cross_size // 2),
            (self.center, crown_top + cross_size // 2),
            3,
        )
        # Horizontale
        pygame.draw.line(
            surface,
            self.colors["highlight"],
            (self.center - cross_size // 2, crown_top),
            (self.center + cross_size // 2, crown_top),
            3,
        )

        # Bijoux sur la couronne
        for i in range(3):
            jewel_x = self.center - crown_width // 3 + i * crown_width // 3
            pygame.draw.circle(
                surface,
                self.colors["highlight"],
                (jewel_x, crown_top + crown_height // 2),
                2,
            )
