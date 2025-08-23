#!/usr/bin/env python3
"""
Lanceur pour l'interface graphique Chess AI.

Ce script lance l'interface graphique moderne avec Pygame.
"""

import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from chess_ai.gui.chess_gui import main

    if __name__ == "__main__":
        print("🚀 Lancement de Chess AI - Interface Graphique")
        print("=" * 50)
        print("Contrôles :")
        print("  • Clic pour sélectionner/déplacer")
        print("  • N - Nouveau jeu")
        print("  • U - Annuler le coup")
        print("  • F - Retourner le plateau")
        print("  • S - Changer style des pièces")
        print("  • A - Analyser la position")
        print("  • ESC - Effacer la sélection")
        print("")
        print("Fonctionnalités :")
        print("  🎨 Pièces vectorielles de haute qualité")
        print("  📊 Analyse en temps réel")
        print("  🎞️ Animations fluides")
        print("  🔄 3 styles de pièces disponibles")
        print("=" * 50)

        sys.exit(main())

except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Assurez-vous que pygame est installé:")
    print("   pip install pygame")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du lancement: {e}")
    sys.exit(1)
