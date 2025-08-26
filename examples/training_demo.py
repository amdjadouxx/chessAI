"""
Exemple d'entraînement AlphaZero avec interface 3D
=================================================

Démonstration du système d'entraînement par auto-jeu
avec visualisation en temps réel dans l'interface 3D.
"""

import sys
import os

sys.path.append("src")

try:
    import torch
except ImportError:
    torch = None

from chess_ai.gui.chess_gui_3d import SimpleChessGUI3D
from chess_ai.ai.training import SelfPlayTrainer, TrainingConfig, quick_training_session


def demo_training_interface():
    """
    Démonstration de l'interface d'entraînement.

    Mode d'emploi :
    1. Lancer l'interface 3D
    2. Appuyer sur T pour activer le mode entraînement
    3. Appuyer sur S pour démarrer l'auto-jeu
    4. Observer les IA jouer l'une contre l'autre
    """
    print("🎮 Démonstration Interface d'Entraînement AlphaZero")
    print("=" * 60)
    print("Mode d'emploi :")
    print("1. 🚀 L'interface 3D va se lancer")
    print("2. 🔄 Appuyez sur 'T' pour activer le mode entraînement")
    print("3. ▶️  Appuyez sur 'S' pour démarrer l'auto-jeu")
    print("4. 👀 Observez les IA jouer l'une contre l'autre")
    print("5. ⏸️  Appuyez sur 'Espace' pour pause/reprendre")
    print("6. 🔄 Appuyez sur 'T' pour revenir au mode normal")
    print("\n🎯 Les modèles s'améliorent automatiquement à chaque itération!")
    print("\nAppuyez sur Entrée pour lancer l'interface...")
    input()

    try:
        # Lancer l'interface 3D avec support d'entraînement
        gui = SimpleChessGUI3D()
        gui.run()

    except Exception as e:
        print(f"❌ Erreur : {e}")
        print("💡 Assurez-vous que PyTorch est installé : pip install torch")


def demo_training_backend_only():
    """
    Démonstration d'entraînement en arrière-plan (sans interface).
    Plus rapide pour l'entraînement réel.
    """
    print("🧠 Démonstration Entraînement Backend AlphaZero")
    print("=" * 60)

    print("⚡ Mode rapide : 5 itérations avec 20 parties chacune")
    print("🎯 Les modèles seront sauvegardés automatiquement")
    print("\nAppuyez sur Entrée pour commencer l'entraînement...")
    input()

    try:
        # Session d'entraînement rapide
        trainer = quick_training_session(
            num_iterations=5,
            num_games=20,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print("✅ Entraînement terminé !")
        print(f"📊 {len(trainer.training_history)} itérations complétées")
        print("💾 Modèles sauvegardés dans le dossier 'models/'")

    except Exception as e:
        print(f"❌ Erreur d'entraînement : {e}")


def demo_continue_existing_training():
    """
    Démonstration de la continuation d'un entraînement existant.
    """
    print("🔄 Continuation d'Entraînement Existant")
    print("=" * 60)

    # Lister les modèles disponibles
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]

        if model_files:
            print("📦 Modèles disponibles :")
            for i, model_file in enumerate(model_files, 1):
                print(f"  {i}. {model_file}")

            try:
                choice = int(input("\nChoisissez un modèle (numéro) : ")) - 1
                if 0 <= choice < len(model_files):
                    model_path = os.path.join(models_dir, model_files[choice])

                    print(
                        f"🔄 Continuation de l'entraînement depuis : {model_files[choice]}"
                    )
                    print("➕ 3 itérations supplémentaires...")

                    from chess_ai.ai.training import continue_training

                    trainer = continue_training(model_path, num_iterations=3)

                    print("✅ Entraînement supplémentaire terminé !")

                else:
                    print("❌ Choix invalide")

            except (ValueError, KeyboardInterrupt):
                print("❌ Opération annulée")
        else:
            print("📂 Aucun modèle trouvé dans le dossier 'models/'")
            print("💡 Lancez d'abord demo_training_backend_only()")
    else:
        print("📂 Dossier 'models/' non trouvé")
        print("💡 Lancez d'abord demo_training_backend_only()")


def demo_custom_training():
    """
    Démonstration d'entraînement avec configuration personnalisée.
    """
    print("⚙️  Entraînement Personnalisé AlphaZero")
    print("=" * 60)

    # Configuration personnalisée
    config = TrainingConfig(
        # MCTS
        mcts_simulations=600,  # Plus de simulations = meilleure qualité
        c_puct=1.6,  # Facteur d'exploration
        # Entraînement
        games_per_iteration=50,  # Plus de parties par itération
        epochs_per_iteration=8,  # Plus d'époques d'entraînement
        batch_size=32,  # Batch plus grand
        learning_rate=0.0005,  # Learning rate réduit
        # Sauvegarde
        save_interval=3,  # Sauvegarder tous les 3 itérations
    )

    print("📋 Configuration personnalisée :")
    print(f"  🎯 MCTS : {config.mcts_simulations} simulations")
    print(f"  🎮 Parties par itération : {config.games_per_iteration}")
    print(f"  🧠 Époques d'entraînement : {config.epochs_per_iteration}")
    print(f"  📦 Taille de batch : {config.batch_size}")
    print(f"  📈 Learning rate : {config.learning_rate}")

    print("\n🚀 Démarrage de l'entraînement personnalisé...")
    print("⏱️  Cela peut prendre plusieurs minutes...")

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔥 Device utilisé : {device}")

        trainer = SelfPlayTrainer(config, device=device)
        trainer.train(num_iterations=5, verbose=True)

        print("✅ Entraînement personnalisé terminé !")

    except Exception as e:
        print(f"❌ Erreur : {e}")


def main():
    """Menu principal pour les démonstrations d'entraînement."""

    print("🤖 ENTRAÎNEMENT ALPHAZERO - Menu Principal")
    print("=" * 60)
    print("Choisissez une démonstration :")
    print()
    print("1. 🎮 Interface 3D avec entraînement visuel")
    print("2. ⚡ Entraînement rapide (backend seul)")
    print("3. 🔄 Continuer un entraînement existant")
    print("4. ⚙️  Entraînement avec configuration personnalisée")
    print("5. ❌ Quitter")
    print()

    try:
        choice = input("Votre choix (1-5) : ").strip()

        if choice == "1":
            demo_training_interface()
        elif choice == "2":
            demo_training_backend_only()
        elif choice == "3":
            demo_continue_existing_training()
        elif choice == "4":
            demo_custom_training()
        elif choice == "5":
            print("👋 Au revoir !")
        else:
            print("❌ Choix invalide")

    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
    except Exception as e:
        print(f"❌ Erreur : {e}")


if __name__ == "__main__":
    # Vérifications préliminaires
    try:
        import torch

        print(f"✅ PyTorch disponible : {torch.__version__}")
        print(f"🔥 CUDA disponible : {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch non installé. Installez avec : pip install torch")
        sys.exit(1)

    try:
        import pygame

        print(f"✅ Pygame disponible : {pygame.version.ver}")
    except ImportError:
        print("❌ Pygame non installé. Installez avec : pip install pygame")
        sys.exit(1)

    try:
        import chess

        print(f"✅ python-chess disponible")
    except ImportError:
        print("❌ python-chess non installé. Installez avec : pip install python-chess")
        sys.exit(1)

    print()
    main()
