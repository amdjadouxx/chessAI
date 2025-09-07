#!/usr/bin/env python3
"""
Lanceur pour l'interface graphique Chess AI.

Ce script lance l'interface graphique moderne avec Pygame.
"""

import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    if __name__ == "__main__":
        use_3d = "--3d" in sys.argv

        # Seule l'interface 3D est disponible maintenant
        from chess_ai.gui.chess_gui_3d import main as main3d

        if use_3d or True:  # Force l'interface 3D
            print("🚀 Lancement de Chess AI - Interface 3D avec IA")
            print("=" * 50)
            print("Contrôles :")
            print("  • Clic gauche : Sélectionner/Déplacer pièces")
            print("  • Clic droit + glisser : Rotation caméra")
            print("  • Molette : Zoom")
            print("  • R : Réinitialiser caméra")
            print("  • H : Toggle suggestions IA")
            print("  • I : Jouer coup IA")
            print("=" * 50)
            sys.exit(main3d())
        else:
            print("❌ L'interface 2D n'est plus disponible")
            print("💡 Utilisez: python launch_gui.py --3d")
            sys.exit(1)
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Assurez-vous que pygame est installé:")
    print("   pip install pygame")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du lancement: {e}")
    sys.exit(1)
