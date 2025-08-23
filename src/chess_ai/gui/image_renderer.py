"""
Renderer d'images pour les pièces d'échecs.

Ce module permet de charger et afficher des images personnalisées
pour les pièces d'échecs depuis le dossier assets/pieces.
"""

import pygame
import os
from typing import Optional, Dict
import chess


class ImagePieceRenderer:
    """
    Renderer pour afficher des images personnalisées des pièces.

    Charge automatiquement les images depuis assets/pieces/
    avec le format: {color}_{piece_type}.png
    """

    def __init__(self, square_size: int, assets_path: str = None):
        """
        Initialise le renderer d'images.

        Args:
            square_size: Taille d'une case en pixels
            assets_path: Chemin vers le dossier des assets
        """
        self.square_size = square_size

        # Déterminer le chemin des assets
        if assets_path is None:
            # Chemin relatif depuis le module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(current_dir))
            )
            assets_path = os.path.join(project_root, "assets", "pieces")

        self.assets_path = assets_path

        # Cache des images chargées
        self.piece_images: Dict[str, pygame.Surface] = {}

        # Mapping des types de pièces vers leurs noms
        self.piece_names = {
            chess.PAWN: "pawn",
            chess.ROOK: "rook",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }

        # Précharger les images disponibles
        self._load_images()

    def _load_images(self) -> None:
        """Charge toutes les images disponibles dans le dossier assets."""
        if not os.path.exists(self.assets_path):
            print(f"⚠️  Dossier assets non trouvé: {self.assets_path}")
            return

        print(f"🖼️  Chargement des images depuis: {self.assets_path}")
        loaded_count = 0

        for color in [chess.WHITE, chess.BLACK]:
            color_name = "white" if color == chess.WHITE else "black"

            for piece_type, piece_name in self.piece_names.items():
                filename = f"{color_name}_{piece_name}.png"
                filepath = os.path.join(self.assets_path, filename)

                if os.path.exists(filepath):
                    try:
                        # Charger l'image
                        image = pygame.image.load(filepath).convert_alpha()

                        # Redimensionner à la taille de la case
                        image = pygame.transform.smoothscale(
                            image, (self.square_size, self.square_size)
                        )

                        # Ajouter un contour selon la couleur de la pièce
                        if color == chess.BLACK:
                            # Contour noir pour les pièces noires
                            image = self._add_outline(image, (0, 0, 0), 2)
                        else:
                            # Contour blanc pour les pièces blanches
                            image = self._add_outline(image, (255, 255, 255), 2)

                        # Stocker dans le cache
                        key = f"{color}_{piece_type}"
                        self.piece_images[key] = image

                        loaded_count += 1
                        print(f"✅ {filename}")

                    except Exception as e:
                        print(f"❌ Erreur chargement {filename}: {e}")

        print(f"🎯 {loaded_count} images chargées")

    def get_piece_surface(self, piece: chess.Piece) -> Optional[pygame.Surface]:
        """
        Retourne la surface pour une pièce donnée.

        Args:
            piece: Pièce d'échecs

        Returns:
            Surface pygame ou None si l'image n'existe pas
        """
        key = f"{piece.color}_{piece.piece_type}"
        return self.piece_images.get(key)

    def has_piece_image(self, piece: chess.Piece) -> bool:
        """
        Vérifie si une image existe pour cette pièce.

        Args:
            piece: Pièce d'échecs

        Returns:
            True si l'image existe
        """
        key = f"{piece.color}_{piece.piece_type}"
        return key in self.piece_images

    def reload_images(self) -> None:
        """Recharge toutes les images depuis le dossier assets."""
        self.piece_images.clear()
        self._load_images()

    def get_available_images(self) -> Dict[str, str]:
        """
        Retourne la liste des images disponibles.

        Returns:
            Dictionnaire {key: filename} des images chargées
        """
        result = {}
        for key in self.piece_images.keys():
            # La clé est au format "True_1" ou "False_6" etc.
            # où le premier élément est la couleur (True=WHITE, False=BLACK)
            # et le second est le type de pièce
            parts = key.split("_")
            color = chess.WHITE if parts[0] == "True" else chess.BLACK
            piece_type = int(parts[1])

            color_name = "white" if color == chess.WHITE else "black"
            piece_name = self.piece_names[piece_type]
            filename = f"{color_name}_{piece_name}.png"

            result[key] = filename

        return result

    def _add_outline(
        self, surface: pygame.Surface, outline_color: tuple, thickness: int = 2
    ) -> pygame.Surface:
        """
        Ajoute un contour à une image de manière simple et efficace.

        Args:
            surface: Surface pygame à modifier
            outline_color: Couleur du contour (R, G, B)
            thickness: Épaisseur du contour en pixels

        Returns:
            Nouvelle surface avec contour
        """
        # Créer une nouvelle surface pour le résultat
        width, height = surface.get_size()
        new_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        new_surface.fill((0, 0, 0, 0))  # Transparent

        # Créer le masque de l'image originale
        mask = pygame.mask.from_surface(surface)

        # Dessiner le contour en décalant le masque
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx == 0 and dy == 0:
                    continue

                # Créer une surface pour ce décalage du contour
                outline_surface = mask.to_surface(
                    setcolor=outline_color + (255,),  # Ajouter alpha
                    unsetcolor=(0, 0, 0, 0),  # Transparent
                )

                # Dessiner le contour décalé
                new_surface.blit(outline_surface, (dx, dy))

        # Dessiner l'image originale par-dessus
        new_surface.blit(surface, (0, 0))

        return new_surface
