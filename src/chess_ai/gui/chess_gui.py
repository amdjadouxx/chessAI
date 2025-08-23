"""
Interface graphique principale pour Chess AI.

Cette classe gère l'interface utilisateur complète avec Pygame,
incluant l'affichage du plateau, les interactions souris, et les animations.
"""

import pygame
import sys
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import chess

from ..core.environment import ChessEnvironment
from ..core.analyzer import ChessAnalyzer
from ..exceptions import ChessError
from .board_renderer import BoardRenderer
from .piece_renderer import PieceRenderer


@dataclass
class GameState:
    """État du jeu pour l'interface graphique."""

    selected_square: Optional[int] = None
    highlighted_moves: List[int] = None
    last_move: Optional[chess.Move] = None
    is_flipped: bool = False
    show_coordinates: bool = True
    show_legal_moves: bool = True
    animation_duration: int = 300  # ms

    def __post_init__(self):
        if self.highlighted_moves is None:
            self.highlighted_moves = []


class ChessGUI:
    """
    Interface graphique moderne pour Chess AI utilisant Pygame.

    Fonctionnalités :
    - Plateau interactif avec drag & drop
    - Animations fluides des mouvements
    - Highlighting des mouvements légaux
    - Analyse de position en temps réel
    - Perspective réversible
    - Historique visuel des coups
    """

    # Constantes d'interface
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    BOARD_SIZE = 640
    SQUARE_SIZE = BOARD_SIZE // 8
    SIDEBAR_WIDTH = WINDOW_WIDTH - BOARD_SIZE - 40

    # Couleurs (thème moderne)
    COLORS = {
        "light_square": (240, 217, 181),
        "dark_square": (181, 136, 99),
        "selected": (255, 255, 0, 100),
        "legal_move": (0, 255, 0, 100),
        "last_move": (255, 255, 0, 150),
        "check": (255, 0, 0, 150),
        "background": (49, 51, 56),
        "sidebar": (40, 42, 47),
        "text": (220, 220, 220),
        "button": (70, 130, 180),
        "button_hover": (100, 149, 237),
    }

    def __init__(self, environment: Optional[ChessEnvironment] = None):
        """
        Initialise l'interface graphique.

        Args:
            environment: Environnement Chess AI optionnel
        """
        # Initialisation Pygame
        pygame.init()
        pygame.display.set_caption("Chess AI - Interface Graphique Moderne")

        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Environnement de jeu
        self.environment = environment or ChessEnvironment(enable_logging=False)
        self.analyzer = ChessAnalyzer(self.environment.board)

        # État du jeu
        self.game_state = GameState()

        # Renderers
        self.board_renderer = BoardRenderer(self.SQUARE_SIZE, self.COLORS)
        self.piece_renderer = PieceRenderer(self.SQUARE_SIZE)

        # Interface
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)

        # Buttons
        self.buttons = self._create_buttons()

        # État d'animation
        self.animation_state = {
            "active": False,
            "start_time": 0,
            "from_square": None,
            "to_square": None,
            "piece": None,
        }

    def _create_buttons(self) -> List[Dict[str, Any]]:
        """Crée les boutons de l'interface."""
        button_width = 120
        button_height = 35  # Réduire la hauteur
        x_start = self.BOARD_SIZE + 20
        y_start = 380  # Descendre les boutons plus bas
        spacing = 50  # Réduire l'espacement

        buttons = [
            {
                "rect": pygame.Rect(x_start, y_start, button_width, button_height),
                "text": "Nouveau jeu",
                "action": self._new_game,
                "color": self.COLORS["button"],
            },
            {
                "rect": pygame.Rect(
                    x_start, y_start + spacing, button_width, button_height
                ),
                "text": "Annuler",
                "action": self._undo_move,
                "color": self.COLORS["button"],
            },
            {
                "rect": pygame.Rect(
                    x_start, y_start + 2 * spacing, button_width, button_height
                ),
                "text": "Retourner",
                "action": self._flip_board,
                "color": self.COLORS["button"],
            },
            {
                "rect": pygame.Rect(
                    x_start, y_start + 3 * spacing, button_width, button_height
                ),
                "text": "Style Pièces",
                "action": self._toggle_piece_style,
                "color": self.COLORS["button"],
            },
        ]

        return buttons

    def run(self) -> None:
        """Boucle principale de l'interface graphique."""
        running = True

        while running:
            dt = self.clock.tick(60)  # 60 FPS

            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Clic gauche
                        self._handle_mouse_click(event.pos)

                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_hover(event.pos)

                elif event.type == pygame.KEYDOWN:
                    self._handle_keyboard(event.key)

            # Mise à jour des animations
            self._update_animations(dt)

            # Rendu
            self._render()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def _handle_mouse_click(self, pos: Tuple[int, int]) -> None:
        """Gère les clics de souris."""
        x, y = pos

        # Vérifier les boutons
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                button["action"]()
                return

        # Vérifier le plateau
        if x < self.BOARD_SIZE and y < self.BOARD_SIZE:
            square = self._pos_to_square(pos)
            if square is not None:
                self._handle_square_click(square)

    def _handle_square_click(self, square: int) -> None:
        """Gère le clic sur une case du plateau."""
        try:
            # Si aucune case sélectionnée
            if self.game_state.selected_square is None:
                piece = self.environment.board.piece_at(square)
                if piece and piece.color == self.environment.board.turn:
                    self.game_state.selected_square = square
                    self.game_state.highlighted_moves = [
                        move.to_square
                        for move in self.environment.board.legal_moves
                        if move.from_square == square
                    ]

            # Si case déjà sélectionnée
            else:
                from_square = self.game_state.selected_square

                # Même case = désélection
                if from_square == square:
                    self._clear_selection()

                # Mouvement
                else:
                    try:
                        move = chess.Move(from_square, square)

                        # Vérifier promotion
                        piece = self.environment.board.piece_at(from_square)
                        if (
                            piece
                            and piece.piece_type == chess.PAWN
                            and (
                                chess.square_rank(square) == 7
                                or chess.square_rank(square) == 0
                            )
                        ):
                            move.promotion = chess.QUEEN  # Auto-promotion en dame

                        # Effectuer le mouvement avec animation
                        if move in self.environment.board.legal_moves:
                            self._animate_move(from_square, square, piece)
                            self.environment.make_move(move)
                            self.game_state.last_move = move

                        self._clear_selection()

                    except ChessError as e:
                        print(f"Mouvement invalide: {e}")
                        self._clear_selection()

        except Exception as e:
            print(f"Erreur lors du clic: {e}")
            self._clear_selection()

    def _animate_move(
        self, from_square: int, to_square: int, piece: chess.Piece
    ) -> None:
        """Lance l'animation d'un mouvement."""
        self.animation_state.update(
            {
                "active": True,
                "start_time": pygame.time.get_ticks(),
                "from_square": from_square,
                "to_square": to_square,
                "piece": piece,
            }
        )

    def _update_animations(self, dt: int) -> None:
        """Met à jour les animations en cours."""
        if not self.animation_state["active"]:
            return

        elapsed = pygame.time.get_ticks() - self.animation_state["start_time"]

        if elapsed >= self.game_state.animation_duration:
            self.animation_state["active"] = False

    def _pos_to_square(self, pos: Tuple[int, int]) -> Optional[int]:
        """Convertit une position écran en case d'échiquier."""
        x, y = pos

        if x < 0 or x >= self.BOARD_SIZE or y < 0 or y >= self.BOARD_SIZE:
            return None

        file = x // self.SQUARE_SIZE
        rank = 7 - (y // self.SQUARE_SIZE)

        if self.game_state.is_flipped:
            file = 7 - file
            rank = 7 - rank

        return chess.square(file, rank)

    def _square_to_pos(self, square: int) -> Tuple[int, int]:
        """Convertit une case d'échiquier en position écran."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if self.game_state.is_flipped:
            file = 7 - file
            rank = 7 - rank

        x = file * self.SQUARE_SIZE
        y = (7 - rank) * self.SQUARE_SIZE

        return x, y

    def _clear_selection(self) -> None:
        """Efface la sélection actuelle."""
        self.game_state.selected_square = None
        self.game_state.highlighted_moves.clear()

    def _render(self) -> None:
        """Effectue le rendu complet de l'interface."""
        # Fond
        self.screen.fill(self.COLORS["background"])

        # Plateau
        self._render_board()

        # Pièces (avec animation)
        self._render_pieces()

        # Interface latérale
        self._render_sidebar()

        # Overlays
        self._render_overlays()

    def _render_board(self) -> None:
        """Rend le plateau d'échecs."""
        for square in chess.SQUARES:
            x, y = self._square_to_pos(square)

            # Couleur de la case
            is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            color = (
                self.COLORS["light_square"] if is_light else self.COLORS["dark_square"]
            )

            # Case de base
            pygame.draw.rect(
                self.screen, color, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)
            )

            # Highlighting
            if square == self.game_state.selected_square:
                self._draw_highlight(x, y, self.COLORS["selected"])

            elif square in self.game_state.highlighted_moves:
                self._draw_highlight(x, y, self.COLORS["legal_move"])

            elif self.game_state.last_move and square in [
                self.game_state.last_move.from_square,
                self.game_state.last_move.to_square,
            ]:
                self._draw_highlight(x, y, self.COLORS["last_move"])

        # Coordonnées
        if self.game_state.show_coordinates:
            self._render_coordinates()

    def _render_pieces(self) -> None:
        """Rend les pièces sur le plateau."""
        for square in chess.SQUARES:
            piece = self.environment.board.piece_at(square)

            if piece:
                # Skip la pièce en animation
                if (
                    self.animation_state["active"]
                    and square == self.animation_state["from_square"]
                ):
                    continue

                x, y = self._square_to_pos(square)
                self.piece_renderer.render_piece(self.screen, piece, x, y)

        # Pièce en animation
        if self.animation_state["active"]:
            self._render_animated_piece()

    def _render_animated_piece(self) -> None:
        """Rend la pièce en cours d'animation."""
        if not self.animation_state["active"]:
            return

        elapsed = pygame.time.get_ticks() - self.animation_state["start_time"]
        progress = min(elapsed / self.game_state.animation_duration, 1.0)

        # Interpolation de position
        from_x, from_y = self._square_to_pos(self.animation_state["from_square"])
        to_x, to_y = self._square_to_pos(self.animation_state["to_square"])

        current_x = from_x + (to_x - from_x) * progress
        current_y = from_y + (to_y - from_y) * progress

        # Rendu de la pièce
        piece = self.animation_state["piece"]
        self.piece_renderer.render_piece(
            self.screen, piece, int(current_x), int(current_y)
        )

    def _render_coordinates(self) -> None:
        """Rend les coordonnées du plateau."""
        coord_font = pygame.font.Font(None, 16)

        for i in range(8):
            # Files (a-h)
            file_char = chr(ord("a") + (7 - i if self.game_state.is_flipped else i))
            text = coord_font.render(file_char, True, self.COLORS["text"])
            x = i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 - text.get_width() // 2
            self.screen.blit(text, (x, self.BOARD_SIZE + 5))

            # Rangs (1-8)
            rank_char = str(8 - i if self.game_state.is_flipped else i + 1)
            text = coord_font.render(rank_char, True, self.COLORS["text"])
            y = i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 - text.get_height() // 2
            self.screen.blit(text, (self.BOARD_SIZE + 5, y))

    def _render_sidebar(self) -> None:
        """Rend la barre latérale avec informations et contrôles."""
        sidebar_x = self.BOARD_SIZE + 20

        # Titre
        title = self.title_font.render("Chess AI", True, self.COLORS["text"])
        self.screen.blit(title, (sidebar_x, 20))

        # Boutons
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            # Couleur du bouton (hover effect)
            color = (
                self.COLORS["button_hover"]
                if button["rect"].collidepoint(mouse_pos)
                else button["color"]
            )

            pygame.draw.rect(self.screen, color, button["rect"])
            pygame.draw.rect(self.screen, self.COLORS["text"], button["rect"], 2)

            # Texte du bouton
            text = self.font.render(button["text"], True, self.COLORS["text"])
            text_rect = text.get_rect(center=button["rect"].center)
            self.screen.blit(text, text_rect)

        # Informations de jeu
        self._render_game_info(sidebar_x, 80)  # Commencer plus haut

    def _render_game_info(self, x: int, y: int) -> None:
        """Rend les informations de la partie."""
        info_lines = [
            f"Tour: {'Blancs' if self.environment.board.turn == chess.WHITE else 'Noirs'}",
            f"Coup #{self.environment.board.fullmove_number}",
            f"Mouvements légaux: {len(list(self.environment.board.legal_moves))}",
            "",
        ]

        # État de la partie
        if self.environment.board.is_checkmate():
            winner = "Noirs" if self.environment.board.turn == chess.WHITE else "Blancs"
            info_lines.append(f"Échec et mat! {winner} gagnent")
        elif self.environment.board.is_stalemate():
            info_lines.append("Pat - Match nul")
        elif self.environment.board.is_check():
            info_lines.append("Échec!")

        # Affichage des informations
        current_y = y
        for line in info_lines:
            if line:  # Skip empty lines
                text = self.font.render(line, True, self.COLORS["text"])
                self.screen.blit(text, (x, current_y))
                current_y += 22  # Espacement réduit
            else:
                current_y += 10  # Petit espace pour les lignes vides

    def _render_overlays(self) -> None:
        """Rend les overlays (échec, etc.)."""
        if self.environment.board.is_check():
            king_square = self.environment.board.king(self.environment.board.turn)
            if king_square:
                x, y = self._square_to_pos(king_square)
                self._draw_highlight(x, y, self.COLORS["check"])

    def _draw_highlight(self, x: int, y: int, color: Tuple[int, int, int, int]) -> None:
        """Dessine un highlighting sur une case."""
        surface = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
        surface.fill(color)
        self.screen.blit(surface, (x, y))

    def _handle_mouse_hover(self, pos: Tuple[int, int]) -> None:
        """Gère le survol de la souris (pour les effets hover)."""
        # Pour l'instant, juste utilisé pour les boutons
        pass

    def _handle_keyboard(self, key: int) -> None:
        """Gère les raccourcis clavier."""
        if key == pygame.K_n:  # N - Nouveau jeu
            self._new_game()
        elif key == pygame.K_u:  # U - Annuler
            self._undo_move()
        elif key == pygame.K_f:  # F - Flip
            self._flip_board()
        elif key == pygame.K_s:  # S - Style
            self._toggle_piece_style()
        elif key == pygame.K_r:  # R - Reload images
            self._reload_images()
        elif key == pygame.K_ESCAPE:  # ESC - Effacer sélection
            self._clear_selection()

    # Actions des boutons
    def _new_game(self) -> None:
        """Démarre une nouvelle partie."""
        self.environment.reset_board()
        self._clear_selection()
        self.game_state.last_move = None

    def _undo_move(self) -> None:
        """Annule le dernier mouvement."""
        try:
            if self.environment.undo_move():
                self._clear_selection()
                self.game_state.last_move = None
        except ChessError as e:
            print(f"Impossible d'annuler: {e}")

    def _flip_board(self) -> None:
        """Retourne le plateau."""
        self.game_state.is_flipped = not self.game_state.is_flipped
        self._clear_selection()

    def _toggle_piece_style(self) -> None:
        """Change le style des pièces."""
        styles = ["vector", "unicode", "minimal", "image"]
        current_index = styles.index(self.piece_renderer.style)
        new_index = (current_index + 1) % len(styles)
        new_style = styles[new_index]

        print(f"🎨 Changement style pièces: {self.piece_renderer.style} → {new_style}")
        self.piece_renderer.set_style(new_style)

        # Afficher les images disponibles si on passe en mode image
        if new_style == "image":
            available = self.piece_renderer.get_available_images()
            if available:
                print(f"🖼️  Images disponibles: {len(available)}")
                for filename in available.values():
                    print(f"   ✅ {filename}")
            else:
                print("⚠️  Aucune image trouvée dans assets/pieces/")
                print("   Utilisation du style vectoriel par défaut")

    def _reload_images(self) -> None:
        """Recharge les images personnalisées."""
        print("🔄 Rechargement des images...")
        self.piece_renderer.reload_images()


def main():
    """Point d'entrée pour l'interface graphique."""
    try:
        env = ChessEnvironment(enable_logging=False)
        gui = ChessGUI(env)
        gui.run()
    except Exception as e:
        print(f"Erreur lors du lancement de l'interface graphique: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
