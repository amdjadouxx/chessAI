"""
Interface 3D Simple pour Chess AI
=================================

Affichage pseudo-3D simple avec Pygame.
Interaction directe et visuelle.
"""

import pygame
import chess
import math
import sys
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Import des modules internes
from ..core.environment import ChessEnvironment


@dataclass
class Camera3D:
    """Configuration simple de la caméra 3D."""

    rotation_x: float = -30.0  # Vue légèrement inclinée vers le bas
    rotation_y: float = 0.0  # Rotation horizontale
    zoom: float = 1.0  # Facteur de zoom


class SimpleChessGUI3D:
    """Interface 3D simple pour jouer aux échecs."""

    def __init__(self):
        # Configuration de base
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 900
        self.BOARD_SIZE = 600
        self.CELL_SIZE = self.BOARD_SIZE // 8

        # Couleurs
        self.COLORS = {
            "light_square": (240, 217, 181),
            "dark_square": (181, 136, 99),
            "white_piece": (255, 255, 255),
            "black_piece": (50, 50, 50),
            "selected": (255, 255, 0, 128),
            "possible_move": (0, 255, 0, 128),
            "background": (100, 100, 100),
        }

        # État du jeu
        self.environment = ChessEnvironment()
        self.camera = Camera3D()

        # État de l'interface
        self.selected_square = None
        self.possible_moves = []
        self.legal_moves_from_selected = []
        self.dragging_camera = False
        self.last_mouse_pos = (0, 0)

        # Initialisation Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI - Vue 3D Simple")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        print("🎮 Interface 3D Simple initialisée !")
        print("🖱️  Contrôles :")
        print("   • Clic gauche : Sélectionner/Déplacer pièces")
        print("   • Clic droit + glisser : Rotation de la caméra")
        print("   • Molette : Zoom")
        print("   • R : Réinitialiser caméra")

    def get_3d_offset(self, file: int, rank: int) -> Tuple[int, int]:
        """Calcule l'offset 3D pour une case donnée."""
        # Facteur de profondeur simple
        depth_factor = 0.3

        # Calcul de l'offset basé sur la rotation
        x_offset = int(
            (rank - 3.5)
            * math.sin(math.radians(self.camera.rotation_y))
            * depth_factor
            * 10
        )
        y_offset = int(
            (rank - 3.5)
            * math.sin(math.radians(self.camera.rotation_x))
            * depth_factor
            * 10
        )

        return x_offset, y_offset

    def screen_to_board(self, screen_pos: Tuple[int, int]) -> Optional[int]:
        """Convertit une position écran en case d'échiquier."""
        x, y = screen_pos

        # Centrer sur l'échiquier
        board_start_x = (self.WINDOW_WIDTH - self.BOARD_SIZE) // 2
        board_start_y = (self.WINDOW_HEIGHT - self.BOARD_SIZE) // 2

        # Vérifier si dans les limites du plateau
        if (
            x < board_start_x
            or x >= board_start_x + self.BOARD_SIZE
            or y < board_start_y
            or y >= board_start_y + self.BOARD_SIZE
        ):
            return None

        # Convertir en coordonnées de case
        file = (x - board_start_x) // self.CELL_SIZE
        rank = 7 - ((y - board_start_y) // self.CELL_SIZE)

        # Vérifier les limites
        if 0 <= file <= 7 and 0 <= rank <= 7:
            return chess.square(file, rank)
        return None

    def draw_board(self):
        """Dessine l'échiquier avec effet 3D."""
        board_start_x = (self.WINDOW_WIDTH - self.BOARD_SIZE) // 2
        board_start_y = (self.WINDOW_HEIGHT - self.BOARD_SIZE) // 2

        for rank in range(8):
            for file in range(8):
                # Couleur de la case
                is_light = (rank + file) % 2 == 0
                color = (
                    self.COLORS["light_square"]
                    if is_light
                    else self.COLORS["dark_square"]
                )

                # Position de base
                x = board_start_x + file * self.CELL_SIZE
                y = board_start_y + (7 - rank) * self.CELL_SIZE

                # Effet 3D
                x_offset, y_offset = self.get_3d_offset(file, rank)

                # Dessiner la case avec l'offset 3D
                rect = pygame.Rect(
                    x + x_offset, y + y_offset, self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

                # Ombre pour l'effet de profondeur
                if x_offset != 0 or y_offset != 0:
                    shadow_rect = pygame.Rect(
                        x + 2, y + 2, self.CELL_SIZE, self.CELL_SIZE
                    )
                    shadow_color = tuple(max(0, c - 50) for c in color)
                    pygame.draw.rect(self.screen, shadow_color, shadow_rect)

    def draw_piece(self, piece: chess.Piece, square: int):
        """Dessine une pièce avec effet 3D."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        board_start_x = (self.WINDOW_WIDTH - self.BOARD_SIZE) // 2
        board_start_y = (self.WINDOW_HEIGHT - self.BOARD_SIZE) // 2

        x = board_start_x + file * self.CELL_SIZE
        y = board_start_y + (7 - rank) * self.CELL_SIZE

        # Effet 3D
        x_offset, y_offset = self.get_3d_offset(file, rank)
        center_x = x + self.CELL_SIZE // 2 + x_offset
        center_y = y + self.CELL_SIZE // 2 + y_offset

        # Couleur de la pièce
        color = (
            self.COLORS["white_piece"] if piece.color else self.COLORS["black_piece"]
        )

        # Symboles des pièces
        symbols = {
            chess.PAWN: "♟" if piece.color else "♟",
            chess.ROOK: "♜" if piece.color else "♜",
            chess.KNIGHT: "♞" if piece.color else "♞",
            chess.BISHOP: "♝" if piece.color else "♝",
            chess.QUEEN: "♛" if piece.color else "♛",
            chess.KING: "♚" if piece.color else "♚",
        }

        # Utiliser des cercles colorés avec des lettres
        piece_letters = {
            chess.PAWN: "P",
            chess.ROOK: "R",
            chess.KNIGHT: "N",
            chess.BISHOP: "B",
            chess.QUEEN: "Q",
            chess.KING: "K",
        }

        # Effet de hauteur pour le 3D
        height_offset = int(abs(x_offset + y_offset) * 0.5)

        # Ombre de la pièce
        pygame.draw.circle(
            self.screen,
            (0, 0, 0, 100),
            (center_x + 3, center_y + 3 - height_offset),
            self.CELL_SIZE // 3,
        )

        # Corps de la pièce
        pygame.draw.circle(
            self.screen,
            color,
            (center_x, center_y - height_offset),
            self.CELL_SIZE // 3,
        )

        # Bordure
        border_color = (0, 0, 0) if piece.color else (255, 255, 255)
        pygame.draw.circle(
            self.screen,
            border_color,
            (center_x, center_y - height_offset),
            self.CELL_SIZE // 3,
            2,
        )

        # Lettre de la pièce
        letter = piece_letters[piece.piece_type]
        text_color = (0, 0, 0) if piece.color else (255, 255, 255)
        text_surface = self.font.render(letter, True, text_color)
        text_rect = text_surface.get_rect(center=(center_x, center_y - height_offset))
        self.screen.blit(text_surface, text_rect)

    def draw_highlights(self):
        """Dessine les surlignages pour la case sélectionnée et les mouvements possibles."""
        if self.selected_square is None:
            return

        board_start_x = (self.WINDOW_WIDTH - self.BOARD_SIZE) // 2
        board_start_y = (self.WINDOW_HEIGHT - self.BOARD_SIZE) // 2

        # Surligner la case sélectionnée
        file = chess.square_file(self.selected_square)
        rank = chess.square_rank(self.selected_square)
        x = board_start_x + file * self.CELL_SIZE
        y = board_start_y + (7 - rank) * self.CELL_SIZE

        x_offset, y_offset = self.get_3d_offset(file, rank)
        highlight_surface = pygame.Surface(
            (self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA
        )
        highlight_surface.fill(self.COLORS["selected"])
        self.screen.blit(highlight_surface, (x + x_offset, y + y_offset))

        # Surligner les mouvements possibles
        for move_square in self.possible_moves:
            file = chess.square_file(move_square)
            rank = chess.square_rank(move_square)
            x = board_start_x + file * self.CELL_SIZE
            y = board_start_y + (7 - rank) * self.CELL_SIZE

            x_offset, y_offset = self.get_3d_offset(file, rank)
            move_surface = pygame.Surface(
                (self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA
            )
            move_surface.fill(self.COLORS["possible_move"])
            self.screen.blit(move_surface, (x + x_offset, y + y_offset))

    def draw_ui(self):
        """Dessine l'interface utilisateur."""
        # Informations sur le jeu
        turn_text = "Tour: " + ("Blancs" if self.environment.board.turn else "Noirs")
        text_surface = self.font.render(turn_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        # Contrôles
        controls = [
            "Contrôles:",
            "Clic gauche: Sélectionner/Jouer",
            "Clic droit + glisser: Rotation caméra",
            "Molette: Zoom",
            "R: Réinitialiser caméra",
        ]

        for i, control in enumerate(controls):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            text_surface = pygame.font.Font(None, 24).render(control, True, color)
            self.screen.blit(text_surface, (10, 50 + i * 25))

    def handle_click(self, pos: Tuple[int, int]):
        """Gère les clics de souris."""
        square = self.screen_to_board(pos)
        if square is None:
            return

        print(f"🎯 Clic sur {chess.square_name(square)}")

        # Si aucune case sélectionnée
        if self.selected_square is None:
            piece = self.environment.board.piece_at(square)
            if piece and piece.color == self.environment.board.turn:
                self.selected_square = square
                # Stocker les mouvements complets pour gérer le roque et les promotions
                self.legal_moves_from_selected = [
                    move
                    for move in self.environment.board.legal_moves
                    if move.from_square == square
                ]
                self.possible_moves = [
                    move.to_square for move in self.legal_moves_from_selected
                ]
                print(
                    f"✅ Pièce sélectionnée: {piece} ({len(self.possible_moves)} mouvements)"
                )
            else:
                print("❌ Pas de pièce valide à sélectionner")
        else:
            # Tentative de mouvement
            if square == self.selected_square:
                # Désélection
                self.selected_square = None
                self.possible_moves = []
                self.legal_moves_from_selected = []
                print("❌ Désélection")
            elif square in self.possible_moves:
                # Mouvement valide - trouver le mouvement complet
                target_move = None
                for move in self.legal_moves_from_selected:
                    if move.to_square == square:
                        target_move = move
                        break

                if target_move:
                    try:
                        # Détecter le type de mouvement avant de l'effectuer
                        is_castling = self.environment.board.is_castling(target_move)
                        is_en_passant = self.environment.board.is_en_passant(
                            target_move
                        )

                        if self.environment.make_move(target_move):
                            move_desc = f"{chess.square_name(self.selected_square)} → {chess.square_name(square)}"
                            # Ajouter info sur le type de mouvement
                            if target_move.promotion:
                                move_desc += f" (promotion: {chess.piece_name(target_move.promotion)})"
                            elif is_castling:
                                move_desc += " (roque)"
                            elif is_en_passant:
                                move_desc += " (en passant)"

                            print(f"✅ Mouvement: {move_desc}")
                            self.selected_square = None
                            self.possible_moves = []
                            self.legal_moves_from_selected = []
                        else:
                            print("❌ Mouvement invalide")
                    except Exception as e:
                        print(f"❌ Erreur: {e}")
                else:
                    print("❌ Mouvement non trouvé")
            else:
                # Nouvelle sélection
                piece = self.environment.board.piece_at(square)
                if piece and piece.color == self.environment.board.turn:
                    self.selected_square = square
                    self.legal_moves_from_selected = [
                        move
                        for move in self.environment.board.legal_moves
                        if move.from_square == square
                    ]
                    self.possible_moves = [
                        move.to_square for move in self.legal_moves_from_selected
                    ]
                    print(
                        f"✅ Nouvelle sélection: {piece} ({len(self.possible_moves)} mouvements)"
                    )
                else:
                    print("❌ Mouvement impossible")

    def handle_camera_rotation(self, rel: Tuple[int, int]):
        """Gère la rotation de la caméra."""
        dx, dy = rel
        self.camera.rotation_y += dx * 0.5
        self.camera.rotation_x = max(-60, min(60, self.camera.rotation_x + dy * 0.5))

    def reset_camera(self):
        """Remet la caméra à sa position par défaut."""
        self.camera.rotation_x = -30.0
        self.camera.rotation_y = 0.0
        self.camera.zoom = 1.0
        print("📷 Caméra réinitialisée")

    def run(self):
        """Boucle principale du jeu."""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Clic gauche
                        self.handle_click(event.pos)
                    elif event.button == 3:  # Clic droit
                        self.dragging_camera = True
                        self.last_mouse_pos = event.pos

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:  # Relâchement clic droit
                        self.dragging_camera = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_camera:
                        rel = (
                            event.pos[0] - self.last_mouse_pos[0],
                            event.pos[1] - self.last_mouse_pos[1],
                        )
                        self.handle_camera_rotation(rel)
                        self.last_mouse_pos = event.pos

                elif event.type == pygame.MOUSEWHEEL:
                    self.camera.zoom += event.y * 0.1
                    self.camera.zoom = max(0.5, min(2.0, self.camera.zoom))

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_camera()

            # Dessiner
            self.screen.fill(self.COLORS["background"])
            self.draw_board()
            self.draw_highlights()

            # Dessiner les pièces
            for square in chess.SQUARES:
                piece = self.environment.board.piece_at(square)
                if piece:
                    self.draw_piece(piece, square)

            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    """Fonction principale."""
    try:
        gui = SimpleChessGUI3D()
        gui.run()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    return True


if __name__ == "__main__":
    if not main():
        sys.exit(1)
