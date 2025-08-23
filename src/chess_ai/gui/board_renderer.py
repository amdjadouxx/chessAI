"""
Rendu du plateau d'échecs pour l'interface graphique.

Ce module gère le rendu visuel du plateau avec tous les
éléments graphiques (cases, highlighting, coordonnées).
"""

import pygame
import math
from typing import Tuple, Dict, Any


class BoardRenderer:
    """
    Gestionnaire de rendu pour le plateau d'échecs.

    Responsabilités :
    - Rendu des cases du plateau
    - Gestion des couleurs et thèmes
    - Highlighting des cases spéciales
    - Coordonnées et annotations
    """

    def __init__(self, square_size: int, colors: Dict[str, Tuple[int, ...]]):
        """
        Initialise le renderer de plateau.

        Args:
            square_size: Taille d'une case en pixels
            colors: Dictionnaire des couleurs du thème
        """
        self.square_size = square_size
        self.colors = colors

        # Surfaces pré-calculées pour les performances
        self._light_square = self._create_square_surface(colors["light_square"])
        self._dark_square = self._create_square_surface(colors["dark_square"])

        # Font pour les coordonnées
        pygame.font.init()
        self.coord_font = pygame.font.Font(None, 16)

    def _create_square_surface(self, color: Tuple[int, int, int]) -> pygame.Surface:
        """Crée une surface de case pré-rendue."""
        surface = pygame.Surface((self.square_size, self.square_size))
        surface.fill(color)

        # Effet de bordure subtil
        border_color = tuple(max(0, c - 20) for c in color)
        pygame.draw.rect(surface, border_color, surface.get_rect(), 1)

        return surface

    def render_board_base(
        self, screen: pygame.Surface, is_flipped: bool = False
    ) -> None:
        """
        Rend le plateau de base (cases alternées).

        Args:
            screen: Surface de rendu
            is_flipped: Si le plateau est retourné
        """
        for rank in range(8):
            for file in range(8):
                # Position écran
                display_file = 7 - file if is_flipped else file
                display_rank = rank if is_flipped else 7 - rank

                x = display_file * self.square_size
                y = display_rank * self.square_size

                # Couleur de la case
                is_light = (file + rank) % 2 == 1
                square_surface = self._light_square if is_light else self._dark_square

                screen.blit(square_surface, (x, y))

    def render_coordinates(
        self, screen: pygame.Surface, board_size: int, is_flipped: bool = False
    ) -> None:
        """
        Rend les coordonnées du plateau.

        Args:
            screen: Surface de rendu
            board_size: Taille totale du plateau
            is_flipped: Si le plateau est retourné
        """
        for i in range(8):
            # Files (a-h)
            file_char = chr(ord("a") + (7 - i if is_flipped else i))
            text = self.coord_font.render(file_char, True, self.colors["text"])
            x = i * self.square_size + self.square_size // 2 - text.get_width() // 2
            screen.blit(text, (x, board_size + 5))

            # Rangs (1-8)
            rank_char = str(i + 1 if is_flipped else 8 - i)
            text = self.coord_font.render(rank_char, True, self.colors["text"])
            y = i * self.square_size + self.square_size // 2 - text.get_height() // 2
            screen.blit(text, (board_size + 5, y))

    def render_highlight(
        self, screen: pygame.Surface, x: int, y: int, highlight_type: str = "selected"
    ) -> None:
        """
        Rend un highlighting sur une case.

        Args:
            screen: Surface de rendu
            x, y: Position de la case
            highlight_type: Type de highlighting
        """
        color_map = {
            "selected": self.colors.get("selected", (255, 255, 0, 100)),
            "legal_move": self.colors.get("legal_move", (0, 255, 0, 100)),
            "last_move": self.colors.get("last_move", (255, 255, 0, 150)),
            "check": self.colors.get("check", (255, 0, 0, 150)),
            "capture": self.colors.get("capture", (255, 0, 0, 100)),
        }

        color = color_map.get(highlight_type, color_map["selected"])

        # Surface transparente
        surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        surface.fill(color)
        screen.blit(surface, (x, y))

    def render_move_indicator(
        self, screen: pygame.Surface, x: int, y: int, is_capture: bool = False
    ) -> None:
        """
        Rend un indicateur de mouvement légal.

        Args:
            screen: Surface de rendu
            x, y: Position de la case
            is_capture: Si c'est une capture
        """
        center_x = x + self.square_size // 2
        center_y = y + self.square_size // 2

        if is_capture:
            # Cercle pour les captures
            radius = self.square_size // 3
            pygame.draw.circle(
                screen, self.colors["legal_move"][:3], (center_x, center_y), radius, 3
            )
        else:
            # Point pour les mouvements normaux
            radius = self.square_size // 8
            pygame.draw.circle(
                screen, self.colors["legal_move"][:3], (center_x, center_y), radius
            )

    def render_last_move_arrow(
        self, screen: pygame.Surface, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> None:
        """
        Rend une flèche pour le dernier mouvement.

        Args:
            screen: Surface de rendu
            from_pos: Position de départ
            to_pos: Position d'arrivée
        """
        from_x, from_y = from_pos
        to_x, to_y = to_pos

        # Centre des cases
        start = (from_x + self.square_size // 2, from_y + self.square_size // 2)
        end = (to_x + self.square_size // 2, to_y + self.square_size // 2)

        # Flèche simple
        pygame.draw.line(screen, self.colors["last_move"][:3], start, end, 5)

        # Pointe de flèche (simple triangle)
        import math

        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow_length = 15
        arrow_angle = math.pi / 6

        point1 = (
            end[0] - arrow_length * math.cos(angle - arrow_angle),
            end[1] - arrow_length * math.sin(angle - arrow_angle),
        )
        point2 = (
            end[0] - arrow_length * math.cos(angle + arrow_angle),
            end[1] - arrow_length * math.sin(angle + arrow_angle),
        )

        pygame.draw.polygon(screen, self.colors["last_move"][:3], [end, point1, point2])

    def render_check_indicator(
        self, screen: pygame.Surface, king_pos: Tuple[int, int]
    ) -> None:
        """
        Rend l'indicateur d'échec autour du roi.

        Args:
            screen: Surface de rendu
            king_pos: Position du roi en échec
        """
        x, y = king_pos

        # Effet pulsant (simple animation)
        import pygame.time

        time_ms = pygame.time.get_ticks()
        intensity = abs(math.sin(time_ms / 200)) * 100 + 155

        check_color = (*self.colors["check"][:3], int(intensity))

        # Bordure rouge pulsante
        surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        pygame.draw.rect(surface, check_color, surface.get_rect(), 5)
        screen.blit(surface, (x, y))
