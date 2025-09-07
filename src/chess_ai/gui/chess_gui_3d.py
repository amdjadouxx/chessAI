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

# Import de l'évaluateur de référence
try:
    from ..ai.reference_evaluator import get_reference_evaluator

    REFERENCE_AVAILABLE = True
except ImportError:
    REFERENCE_AVAILABLE = False
    print("⚠️  Évaluateur de référence non disponible")

# Import de l'entraînement hybride
try:
    from ..ai.hybrid_training import HybridSelfPlayTrainer, HybridTrainingConfig

    HYBRID_TRAINING_AVAILABLE = True
except ImportError:
    HYBRID_TRAINING_AVAILABLE = False
    print("⚠️  Entraînement hybride non disponible")

# Import de l'IA (optionnel)
try:
    from .ai_integration import AlphaZeroPlayer

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  Module IA non disponible (PyTorch requis)")

# Import du module d'entraînement (optionnel)
try:
    from ..ai.training import SelfPlayTrainer, TrainingConfig
    import torch

    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("⚠️  Module d'entraînement non disponible")

# Import de l'IA hybride (nouvelle)
try:
    from ..ai.hybrid_engine import HybridAI, AIMode, GameContext, create_hybrid_ai

    HYBRID_AI_AVAILABLE = True
except ImportError:
    HYBRID_AI_AVAILABLE = False
    print("⚠️  IA hybride non disponible")


@dataclass
class Camera3D:
    """Configuration simple de la caméra 3D."""

    rotation_x: float = -30.0  # Vue légèrement inclinée vers le bas
    rotation_y: float = 0.0  # Rotation horizontale
    zoom: float = 1.0  # Facteur de zoom


class SimpleChessGUI3D:
    """Interface 3D simple pour jouer aux échecs avec mode d'entraînement."""

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
            "ai_suggestion": (0, 150, 255, 100),  # Bleu pour suggestions IA
            "background": (100, 100, 100),
            # Couleurs pour la barre d'évaluation
            "eval_white": (240, 240, 240),
            "eval_black": (60, 60, 60),
            "eval_border": (0, 0, 0),
            "eval_text": (255, 255, 255),
            # Couleurs pour la barre de référence (Stockfish)
            "ref_border": (0, 150, 255),
            "ref_black": (30, 30, 80),
            "ref_white": (200, 220, 255),
            # Couleurs pour la barre IA
            "ai_border": (255, 150, 0),
            "ai_black": (80, 40, 0),
            "ai_white": (255, 220, 180),
        }

        # État du jeu
        self.environment = ChessEnvironment()
        self.camera = Camera3D()

        # IA (optionnel)
        self.ai_player = None
        self.ai_enabled = False
        self.show_ai_hints = False
        self.ai_analysis = None

        # Barres d'évaluation
        self.current_evaluation = 0.0  # -1.0 (noir) à +1.0 (blanc) - IA
        self.evaluation_history = []  # Historique des évaluations IA
        self.reference_evaluation = 0.0  # Évaluation de référence (Stockfish)
        self.reference_history = []  # Historique référence
        self.show_evaluation_bar = True
        self.eval_bar_width = 40
        self.eval_bar_height = 400

        # Évaluateur de référence
        self.reference_evaluator = None
        if REFERENCE_AVAILABLE:
            try:
                self.reference_evaluator = get_reference_evaluator()
            except Exception as e:
                print(f"⚠️  Impossible d'initialiser l'évaluateur de référence : {e}")

        # Mode IA vs IA
        self.ai_vs_ai_mode = False
        self.ai_player_white = None
        self.ai_player_black = None
        self.auto_play_delay = 2000  # 2 secondes entre les coups
        self.last_move_time = 0
        self.game_paused = False

        # Mode IA continue contre joueur
        self.ai_auto_play = False  # IA joue automatiquement ses coups
        self.ai_delay = 1500  # Délai avant que l'IA joue (ms)
        self.waiting_for_ai = False  # En attente du coup IA
        self.ai_move_time = 0  # Temps du dernier coup pour l'IA
        self.ai_color = None  # Couleur de l'IA (None = auto-détection)
        self.player_color = None  # Couleur du joueur (None = auto-détection)

        # Entraînement pendant le jeu
        self.learning_from_games = (
            True  # 🚀 ACTIVÉ PAR DÉFAUT ! Apprendre des parties contre le joueur
        )
        self.current_game_moves = []  # Mouvements de la partie actuelle
        self.game_start_time = None  # Temps de début de partie
        self.collected_games = []  # Parties collectées pour l'entraînement

        # Mode d'entraînement
        self.training_mode = False
        self.trainer = None
        self.hybrid_trainer = None  # Nouveau : trainer hybride
        self.use_hybrid_training = True  # Par défaut, utiliser l'hybride
        self.use_hybrid_trainer = True  # Alias pour compatibilité
        self.training_iteration = 0
        self.training_games_played = 0
        self.current_training_game = None
        self.training_auto_play = False
        self.training_speed = 1000  # ms entre les coups en mode entraînement

        # Entraînement automatique jusqu'à convergence
        self.auto_training_active = False
        self.auto_training_iteration = 0
        self.auto_training_max_iterations = 100  # Augmenté de 20 à 100
        self.auto_training_target_diff = 0.25
        self.auto_training_target_corr = 0.7
        self.auto_training_paused = False
        self.auto_training_speed = 100  # Accéléré de 500 à 100ms
        self.auto_training_game_count = 0

        if AI_AVAILABLE:
            try:
                self.ai_player = AlphaZeroPlayer()
                self.ai_enabled = True
                print("🤖 IA AlphaZero activée")

                # Créer deux IA différentes pour le mode vs IA
                self.ai_player_white = AlphaZeroPlayer(
                    use_mcts=True, mcts_simulations=200, c_puct=1.4
                )
                self.ai_player_black = AlphaZeroPlayer(
                    use_mcts=True, mcts_simulations=300, c_puct=1.6
                )
                print("🤖 Mode IA vs IA disponible (touche A)")

                # Initialiser l'entraîneur si disponible
                if TRAINING_AVAILABLE:
                    self.setup_trainer()

            except Exception as e:
                print(f"⚠️  IA non disponible : {e}")

        # Initialiser l'IA hybride (nouvelle)
        self.hybrid_ai = None
        self.hybrid_ai_enabled = False
        if HYBRID_AI_AVAILABLE:
            try:
                self.hybrid_ai = create_hybrid_ai()
                self.hybrid_ai_enabled = True
                self.current_ai_mode = AIMode.ADAPTIVE
                print("🚀 IA Hybride activée (Stockfish + Neural + MCTS)")
                print("   • Mode adaptatif intelligent")
                print("   • Force immédiate de niveau maître")
            except Exception as e:
                print(f"⚠️  IA Hybride non disponible : {e}")

        # Préférence IA : utiliser hybride si disponible, sinon standard
        self.use_hybrid_ai = self.hybrid_ai_enabled

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
        if self.ai_enabled:
            print("   • H : Suggestions IA")
            print("   • A : Analyse position")
            print("   • I : Coup IA")
            if TRAINING_AVAILABLE:
                print("   • T : Mode entraînement")
                print("   • S : Démarrer/Arrêter auto-jeu")

    def setup_trainer(self):
        """Initialise l'entraîneur pour le mode d'entraînement."""
        try:
            # Configuration d'entraînement adaptée à l'interface
            config = TrainingConfig(
                games_per_iteration=5,  # Réduit pour tests plus rapides
                mcts_simulations=100,  # Simulations réduites pour la vitesse
                epochs_per_iteration=3,  # Moins d'époques
                batch_size=8,  # Batch plus petit
                save_interval=1,  # 🚀 SAUVEGARDER APRÈS CHAQUE ITÉRATION !
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Entraîneur classique (AlphaZero)
            self.trainer = SelfPlayTrainer(config, device=device)

            # Entraîneur hybride (Stockfish + AlphaZero) avec ADAPTATION AUTOMATIQUE
            if HYBRID_TRAINING_AVAILABLE:
                hybrid_config = HybridTrainingConfig(
                    games_per_iteration=5,  # Réduit pour tests plus rapides
                    mcts_simulations=50,  # Moins car Stockfish compense
                    epochs_per_iteration=3,
                    batch_size=8,
                    save_interval=1,  # 🚀 SAUVEGARDER APRÈS CHAQUE ITÉRATION !
                    use_stockfish_guidance=True,
                    stockfish_depth=15,  # 🚀 Niveau club fort (~1800-2000 ELO)
                    # 🎲 VARIATION DE PROFONDEUR pour éviter la répétitivité
                    vary_stockfish_depth=True,  # Activer variation
                    depth_range=(10, 20),  # Facile à difficile
                    depth_variation_mode="adaptive",  # Mode intelligent
                    # 🚀 ADAPTATION AUTOMATIQUE PAR DÉFAUT !
                    adaptive_training=True,  # Adaptation activée
                    neural_confidence_threshold=0.8,  # Seuil de confiance initial
                    # 🎯 ÉVALUATIONS STOCKFISH PAR DÉFAUT !
                    use_stockfish_values=True,  # Toujours utiliser Stockfish pour les valeurs
                    stockfish_eval_depth=15,  # 🚀 Même force pour évaluations
                )
                self.hybrid_trainer = HybridSelfPlayTrainer(
                    hybrid_config, device=device
                )
                print(f"🏋️  Entraîneur HYBRIDE initialisé sur {device}")
                print(f"   ⚡ Force immédiate grâce à Stockfish")
                print(f"   🎯 Évaluation précise des positions par Stockfish")
                print(f"   📈 Neural Network apprend les VRAIES valeurs !")
                print(f"   🎲 Variation automatique de profondeur Stockfish (10-20)")
                print(f"   🔄 ADAPTATION AUTOMATIQUE activée par défaut")
                print(f"   🎓 ENTRAÎNEMENT APRÈS CHAQUE PARTIE activé")
            else:
                self.hybrid_trainer = None
                print(f"⚠️  Entraîneur hybride non disponible")

            print(f"🏋️  Entraîneur classique initialisé sur {device}")
            print("   • Mode entraînement disponible")
            print(f"   • Hybride: {'✅' if self.hybrid_trainer else '❌'}")

            # 🚀 INFORMATION UTILISATEUR SUR LES FONCTIONNALITÉS AUTOMATIQUES
            if self.hybrid_trainer:
                print()
                print("🎓 APPRENTISSAGE CONTINU ACTIVÉ PAR DÉFAUT !")
                print("=" * 50)
                print("✅ L'IA apprendra automatiquement de toutes vos parties")
                print("✅ Adaptation automatique du ratio Stockfish/Neural Network")
                print("✅ Entraînement immédiat après chaque partie")
                print("✅ Évaluations précises avec Stockfish")
                print()
                print("📈 Plus vous jouez, plus l'IA s'améliore !")
                print("🔄 Le système s'adapte selon les performances de l'IA")
                print()

            # 🚀 IMPORTANT : Définir le trainer principal pour l'interface
            # Utiliser l'hybride si disponible, sinon le classique
            if self.hybrid_trainer:
                self.trainer = self.hybrid_trainer  # Le trainer principal = hybride
                print("🎯 Trainer principal: HYBRIDE (Stockfish + Neural Network)")
            else:
                print("🎯 Trainer principal: CLASSIQUE (Neural Network seul)")

        except Exception as e:
            print(f"⚠️  Entraîneur non disponible : {e}")
            self.trainer = None
            self.hybrid_trainer = None

    def toggle_training_mode(self):
        """Active/désactive le mode d'entraînement."""
        # Utiliser le trainer hybride si disponible, sinon le classique
        current_trainer = self.hybrid_trainer if self.hybrid_trainer else self.trainer

        if not current_trainer:
            print("❌ Mode entraînement non disponible")
            return

        self.training_mode = not self.training_mode

        if self.training_mode:
            trainer_type = (
                "HYBRIDE (Stockfish+NN)" if self.hybrid_trainer else "CLASSIQUE"
            )
            print(f"🏋️  MODE ENTRAÎNEMENT {trainer_type} ACTIVÉ")
            print("🚀 Démarrage automatique de l'entraînement...")
            # Démarrer automatiquement l'entraînement
            self.start_training_iteration()
        else:
            print("🎮 Mode normal activé")
            self.training_auto_play = False

    def start_training_iteration(self):
        """Démarre une itération d'entraînement avec visualisation."""
        # Utiliser le trainer hybride si disponible, sinon le classique
        current_trainer = self.hybrid_trainer if self.hybrid_trainer else self.trainer

        if not current_trainer or not self.training_mode:
            return

        trainer_type = "hybride" if self.hybrid_trainer else "classique"
        print(
            f"\n🚀 Démarrage itération d'entraînement {trainer_type} #{self.training_iteration + 1}"
        )

        # Réinitialiser pour une nouvelle partie d'entraînement
        self.environment.board = chess.Board()
        self.training_auto_play = True
        self.training_games_played = 0
        self.last_move_time = pygame.time.get_ticks()

        # Utiliser les IA du trainer pour l'auto-jeu
        self.ai_player_white = AlphaZeroPlayer(
            use_mcts=True,
            mcts_simulations=current_trainer.config.mcts_simulations,
            c_puct=current_trainer.config.c_puct,
        )
        self.ai_player_black = AlphaZeroPlayer(
            use_mcts=True,
            mcts_simulations=current_trainer.config.mcts_simulations,
            c_puct=current_trainer.config.c_puct,
        )

        print(f"🎯 Objectif: {self.trainer.config.games_per_iteration} parties")

    def start_automatic_training(self):
        """Lance l'entraînement automatique jusqu'à convergence avec visualisation."""
        # 🚀 UTILISER L'ENTRAÎNEUR HYBRIDE PAR DÉFAUT si disponible
        current_trainer = self.hybrid_trainer if self.hybrid_trainer else self.trainer

        if not current_trainer:
            print("❌ Entraîneur non disponible")
            return

        trainer_type = (
            "HYBRIDE avec adaptation automatique"
            if self.hybrid_trainer
            else "CLASSIQUE"
        )

        print(f"\n🎯 ENTRAÎNEMENT AUTOMATIQUE {trainer_type}")
        print("=" * 60)
        print("L'IA va s'entraîner automatiquement jusqu'à atteindre")
        print("des performances satisfaisantes par rapport à Stockfish!")

        if self.hybrid_trainer:
            print()
            print("🔄 FONCTIONNALITÉS ACTIVÉES PAR DÉFAUT:")
            print("  • Adaptation automatique du ratio Stockfish/NN")
            print("  • Entraînement immédiat après chaque partie")
            print("  • Évaluations Stockfish précises")
            print("  • Apprentissage des meilleures politiques Stockfish")

        print()
        print("🎮 CONTRÔLES PENDANT L'ENTRAÎNEMENT:")
        print("  • ÉCHAP : Arrêter l'entraînement")
        print("  • ESPACE : Pause/Reprendre")
        print("  • + : Accélérer (moins d'attente)")
        print("  • - : Ralentir (plus d'attente)")
        print()

        # Configuration pour l'entraînement automatique
        self.auto_training_active = True
        self.auto_training_iteration = 0
        self.auto_training_max_iterations = 30
        self.auto_training_target_diff = 0.25
        self.auto_training_target_corr = 0.7
        self.auto_training_paused = False
        self.auto_training_speed = 500  # ms entre les coups
        self.auto_training_game_count = 0

        # Utiliser le bon trainer
        self.trainer = current_trainer

        # Démarrer le mode d'entraînement
        self.training_mode = True

        print(f"🎯 Objectifs:")
        print(f"  • Écart moyen < {self.auto_training_target_diff}")
        print(f"  • Corrélation > {self.auto_training_target_corr}")
        print(f"  • Maximum {self.auto_training_max_iterations} itérations")

        if self.hybrid_trainer:
            config = self.hybrid_trainer.hybrid_config
            print(f"  • Profondeur Stockfish: {config.stockfish_depth}")
            print(f"  • Seuil confiance NN: {config.neural_confidence_threshold:.2f}")

        print()
        print("🚀 Démarrage de l'entraînement automatique...")

    def handle_training_auto_play(self):
        """Gère l'auto-jeu en mode entraînement."""
        current_time = pygame.time.get_ticks()

        # Vérifier si on doit redémarrer automatiquement après une itération
        if (
            self.training_mode
            and not self.training_auto_play
            and hasattr(self, "last_move_time")
            and current_time > self.last_move_time
        ):
            print(
                f"🚀 Redémarrage automatique de l'itération {self.training_iteration + 1}"
            )
            self.start_training_iteration()
            return

        if not self.training_auto_play or not self.trainer:
            return

        # Vérifier si il est temps pour le prochain coup
        if current_time - self.last_move_time < self.training_speed:
            return

        board = self.environment.board

        # Vérifier si la partie est terminée
        if board.is_game_over():
            self.finish_training_game()
            return

        # Obtenir le joueur courant
        if board.turn == chess.WHITE:
            current_ai = self.ai_player_white
            player_name = "Blanc"
        else:
            current_ai = self.ai_player_black
            player_name = "Noir"

        try:
            # Obtenir le coup de l'IA
            move = current_ai.select_move(board, temperature=0.3)

            if move and move in board.legal_moves:
                board.push(move)
                self.last_move_time = current_time
                print(f"🤖 {player_name}: {move}")

                # Reset des sélections
                self.selected_square = None
                self.possible_moves = []

                # Mettre à jour l'évaluation après le coup
                self.update_evaluation()

        except Exception as e:
            print(f"❌ Erreur IA {player_name}: {e}")
            self.training_auto_play = False

    def finish_training_game(self):
        """Termine une partie d'entraînement et commence la suivante."""
        result = self.environment.board.result()
        self.training_games_played += 1

        print(f"🏁 Partie {self.training_games_played}: {result}")

        # 🚀 ENTRAÎNEMENT IMMÉDIAT APRÈS CHAQUE PARTIE (PAR DÉFAUT) !
        current_trainer = self.hybrid_trainer if self.hybrid_trainer else self.trainer

        if current_trainer:
            try:
                # Créer les données de la partie qui vient de se terminer
                from ..ai.training import GameData
                from ..ai.hybrid_training import HybridGameData
                from ..ai.network import encode_board

                # Reconstruire les données de la partie
                board = chess.Board()
                positions = []
                board_positions = []
                moves = []

                # Rejouer la partie pour extraire les positions
                for move_san in self.environment.board.move_stack:
                    positions.append(encode_board(board))
                    board_positions.append(board.copy())
                    moves.append(move_san.uci())
                    board.push(move_san)

                if positions:  # Seulement si on a des positions
                    print(f"🧠 Entraînement immédiat sur la partie terminée...")

                    # 🚀 UTILISER L'ENTRAÎNEMENT HYBRIDE PAR DÉFAUT si disponible
                    if self.hybrid_trainer and hasattr(
                        self.hybrid_trainer, "train_single_game_hybrid"
                    ):
                        print(f"   📊 Mode HYBRIDE avec adaptation automatique...")

                        # Calculer les valeurs et politiques Stockfish
                        values = self.hybrid_trainer._calculate_stockfish_values(
                            board_positions, result
                        )

                        # Créer des politiques dummy (seront recalculées par Stockfish)
                        policies = []
                        stockfish_policies = []

                        for board_pos in board_positions:
                            # Politique uniforme dummy pour MCTS
                            legal_moves = list(board_pos.legal_moves)
                            if legal_moves:
                                uniform_prob = 1.0 / len(legal_moves)
                                dummy_policy = {
                                    move: uniform_prob for move in legal_moves
                                }
                                policies.append(dummy_policy)

                                # Politique Stockfish concentrée sur le meilleur coup
                                try:
                                    best_move = self.hybrid_trainer.reference_evaluator.get_best_move(
                                        board_pos
                                    )
                                    if best_move and best_move in legal_moves:
                                        stockfish_policy = {}
                                        for move in legal_moves:
                                            if move == best_move:
                                                stockfish_policy[move] = 0.8
                                            else:
                                                stockfish_policy[move] = 0.2 / (
                                                    len(legal_moves) - 1
                                                )
                                        stockfish_policies.append(stockfish_policy)
                                    else:
                                        stockfish_policies.append(dummy_policy)
                                except:
                                    stockfish_policies.append(dummy_policy)
                            else:
                                policies.append({})
                                stockfish_policies.append({})

                        # Créer les données de partie hybride
                        game_data = HybridGameData(
                            positions=positions,
                            policies=policies,
                            values=values,
                            result=result,
                            moves=moves,
                            game_length=len(moves),
                            stockfish_policies=stockfish_policies,
                        )

                        # Entraîner le réseau avec adaptation automatique
                        training_metrics = self.hybrid_trainer.train_single_game_hybrid(
                            game_data, verbose=True
                        )
                        print(f"   🔄 Adaptation automatique appliquée")

                    elif self.trainer and hasattr(self.trainer, "train_single_game"):
                        print(f"   📊 Mode CLASSIQUE...")

                        # Entraînement classique avec évaluations Stockfish
                        values = self.trainer._calculate_stockfish_values(
                            board_positions, result
                        )

                        # Politiques dummy
                        policies = []
                        for board_pos in board_positions:
                            legal_moves = list(board_pos.legal_moves)
                            if legal_moves:
                                uniform_prob = 1.0 / len(legal_moves)
                                dummy_policy = {
                                    move: uniform_prob for move in legal_moves
                                }
                                policies.append(dummy_policy)
                            else:
                                policies.append({})

                        game_data = GameData(
                            positions=positions,
                            policies=policies,
                            values=values,
                            result=result,
                            moves=moves,
                            game_length=len(moves),
                        )

                        training_metrics = self.trainer.train_single_game(
                            game_data, verbose=True
                        )

                    print(
                        f"✅ Entraînement terminé ! Loss: {training_metrics.get('total_loss', 0):.4f}"
                    )

            except Exception as e:
                print(f"⚠️ Erreur lors de l'entraînement automatique: {e}")
                import traceback

                traceback.print_exc()

        # Vérifier si on a terminé toutes les parties pour cette itération
        if self.training_games_played >= current_trainer.config.games_per_iteration:
            self.complete_training_iteration()
        else:
            # Commencer une nouvelle partie
            self.environment.board = chess.Board()
            self.last_move_time = (
                pygame.time.get_ticks() + 1000
            )  # Pause de 1s entre les parties

    def complete_training_iteration(self):
        """Complète une itération d'entraînement."""
        print(f"\n✅ Itération {self.training_iteration + 1} terminée!")
        print(f"📊 {self.training_games_played} parties jouées")

        # En mode interface, on ne fait pas l'entraînement complet
        # (trop lourd), mais on simule la progression
        self.training_iteration += 1

        # Continuer automatiquement l'entraînement si on est en mode training
        if self.training_mode:
            print(
                f"🔄 Démarrage automatique de l'itération {self.training_iteration + 1}"
            )
            # Petite pause de 2 secondes puis redémarrage automatique
            self.training_auto_play = False
            self.last_move_time = pygame.time.get_ticks() + 2000  # 2s de pause
            # On va redémarrer dans handle_training
        else:
            self.training_auto_play = False
            print(f"🔄 Prêt pour l'itération {self.training_iteration + 1}")
            print("   • Appuyez sur S pour continuer l'entraînement")

    def handle_automatic_training(self):
        """Gère l'entraînement automatique jusqu'à convergence."""
        current_time = pygame.time.get_ticks()

        # Contrôler la vitesse d'affichage
        if hasattr(self, "last_auto_training_update"):
            if current_time - self.last_auto_training_update < self.auto_training_speed:
                return

        self.last_auto_training_update = current_time

        # Si pas de partie en cours, en démarrer une nouvelle
        if (
            not hasattr(self, "auto_training_game_active")
            or not self.auto_training_game_active
        ):
            self.start_auto_training_game()
            return

        # Jouer le prochain coup
        board = self.environment.board

        if board.is_game_over():
            self.finish_auto_training_game()
            return

        # Obtenir le coup de l'IA
        try:
            move = self.get_ai_move(board, time_budget=1.0)
            if move and move in board.legal_moves:
                board.push(move)
                self.update_evaluation()

                # Afficher le coup
                move_str = f"{'Blanc' if not board.turn else 'Noir'}: {move}"
                print(f"🎯 Auto-training: {move_str}")

        except Exception as e:
            print(f"⚠️ Erreur pendant l'auto-training: {e}")
            self.auto_training_active = False

    def start_auto_training_game(self):
        """Démarre une nouvelle partie d'entraînement automatique."""
        self.environment.board = chess.Board()
        self.auto_training_game_active = True
        self.auto_training_game_count += 1

        print(
            f"\n🎮 Partie {self.auto_training_game_count} - Itération {self.auto_training_iteration + 1}"
        )

    def finish_auto_training_game(self):
        """Termine une partie d'entraînement automatique."""
        result = self.environment.board.result()
        self.auto_training_game_active = False

        print(f"🏁 Partie {self.auto_training_game_count} terminée: {result}")

        # Vérifier si on a assez de parties pour cette itération
        games_per_iteration = 1  # 🚀 RAPIDE : Entraînement après chaque partie !

        if self.auto_training_game_count >= games_per_iteration:
            self.complete_auto_training_iteration()
        else:
            # Pause très courte entre les parties pour accélérer
            self.last_auto_training_update = (
                pygame.time.get_ticks() + 200
            )  # 200ms au lieu de 1000ms

    def complete_auto_training_iteration(self):
        """Complète une itération d'entraînement automatique."""
        self.auto_training_iteration += 1
        self.auto_training_game_count = 0

        print(f"\n✅ Itération {self.auto_training_iteration} terminée!")

        # Effectuer l'entraînement réel avec le trainer approprié
        current_trainer = (
            self.hybrid_trainer if self.use_hybrid_trainer else self.trainer
        )

        if current_trainer:
            try:
                print(
                    f"🧠 Entraînement du réseau avec trainer {'hybride' if self.use_hybrid_trainer else 'standard'}..."
                )

                # 🎯 VRAIE COLLECTE : Jouer une partie avec le trainer pour obtenir les données
                print("📊 Génération d'une partie d'entraînement...")
                game_data = current_trainer.play_game_hybrid(verbose=False)
                games_data = [game_data]  # Liste avec une partie

                if hasattr(current_trainer, "train_network"):
                    # Entraîner avec les vraies données de partie
                    current_trainer.train_network(games_data)
                    print("📈 Réseau entraîné avec succès!")
                else:
                    print("⚠️ Pas de méthode d'entraînement disponible")

            except Exception as e:
                print(f"⚠️ Erreur pendant l'entraînement: {e}")

        # Vérifier si on continue
        if self.auto_training_iteration >= self.auto_training_max_iterations:
            print("🎉 ENTRAÎNEMENT AUTOMATIQUE TERMINÉ!")
            print("🏆 Nombre maximum d'itérations atteint")
            self.auto_training_active = False
            self.training_mode = False
        else:
            progress = (
                self.auto_training_iteration / self.auto_training_max_iterations
            ) * 100
            print(
                f"🔄 Démarrage itération {self.auto_training_iteration + 1} (Progrès: {progress:.1f}%)"
            )
            # Petite pause entre les itérations
            self.last_auto_training_update = (
                pygame.time.get_ticks() + 1000
            )  # 1s au lieu de 2s

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
        board_start_x = (self.WINDOW_WIDTH - self.BOARD_SIZE) // 2
        board_start_y = (self.WINDOW_HEIGHT - self.BOARD_SIZE) // 2

        # Surligner la case sélectionnée
        if self.selected_square is not None:
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

        # Afficher les suggestions IA
        if self.show_ai_hints and self.ai_analysis:
            for i, (move, prob) in enumerate(self.ai_analysis["top_moves"][:3]):
                file = chess.square_file(move.to_square)
                rank = chess.square_rank(move.to_square)
                x = board_start_x + file * self.CELL_SIZE
                y = board_start_y + (7 - rank) * self.CELL_SIZE

                x_offset, y_offset = self.get_3d_offset(file, rank)
                ai_surface = pygame.Surface(
                    (self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA
                )
                # Intensité basée sur la probabilité
                alpha = int(50 + prob * 100)
                color = (*self.COLORS["ai_suggestion"][:3], alpha)
                ai_surface.fill(color)
                self.screen.blit(ai_surface, (x + x_offset, y + y_offset))

    def draw_coordinates(self):
        """Dessine les coordonnées A-H et 1-8 autour du plateau."""
        board_start_x = (self.WINDOW_WIDTH - self.BOARD_SIZE) // 2
        board_start_y = (self.WINDOW_HEIGHT - self.BOARD_SIZE) // 2

        # Font plus lisible pour les coordonnées
        coord_font = pygame.font.Font(None, 28)

        # Couleur des coordonnées - blanc avec bordure noire pour la visibilité
        coord_color = (255, 255, 255)
        border_color = (0, 0, 0)

        # Dessiner les lettres A-H (colonnes)
        for file in range(8):
            letter = chr(ord("A") + file)
            x = board_start_x + file * self.CELL_SIZE + self.CELL_SIZE // 2

            # En haut du plateau
            y_top = board_start_y - 25
            # Bordure noire
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        border_surface = coord_font.render(letter, True, border_color)
                        border_rect = border_surface.get_rect(
                            center=(x + dx, y_top + dy)
                        )
                        self.screen.blit(border_surface, border_rect)
            # Texte blanc
            text_surface = coord_font.render(letter, True, coord_color)
            text_rect = text_surface.get_rect(center=(x, y_top))
            self.screen.blit(text_surface, text_rect)

            # En bas du plateau
            y_bottom = board_start_y + self.BOARD_SIZE + 15
            # Bordure noire
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        border_surface = coord_font.render(letter, True, border_color)
                        border_rect = border_surface.get_rect(
                            center=(x + dx, y_bottom + dy)
                        )
                        self.screen.blit(border_surface, border_rect)
            # Texte blanc
            text_surface = coord_font.render(letter, True, coord_color)
            text_rect = text_surface.get_rect(center=(x, y_bottom))
            self.screen.blit(text_surface, text_rect)

        # Dessiner les chiffres 1-8 (rangées)
        for rank in range(8):
            number = str(8 - rank)  # Inversion car rank 0 = 8ème rangée
            y = board_start_y + rank * self.CELL_SIZE + self.CELL_SIZE // 2

            # À gauche du plateau
            x_left = board_start_x - 25
            # Bordure noire
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        border_surface = coord_font.render(number, True, border_color)
                        border_rect = border_surface.get_rect(
                            center=(x_left + dx, y + dy)
                        )
                        self.screen.blit(border_surface, border_rect)
            # Texte blanc
            text_surface = coord_font.render(number, True, coord_color)
            text_rect = text_surface.get_rect(center=(x_left, y))
            self.screen.blit(text_surface, text_rect)

            # À droite du plateau
            x_right = board_start_x + self.BOARD_SIZE + 20
            # Bordure noire
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        border_surface = coord_font.render(number, True, border_color)
                        border_rect = border_surface.get_rect(
                            center=(x_right + dx, y + dy)
                        )
                        self.screen.blit(border_surface, border_rect)
            # Texte blanc
            text_surface = coord_font.render(number, True, coord_color)
            text_rect = text_surface.get_rect(center=(x_right, y))
            self.screen.blit(text_surface, text_rect)

    def get_ai_move(
        self,
        board: chess.Board,
        time_budget: float = 15.0,  # 🚀 Plus de force par défaut
    ) -> Optional[chess.Move]:
        """
        Obtient un coup de l'IA (trainer hybride ou standard selon configuration).

        Args:
            board: Position actuelle
            time_budget: Budget de temps en secondes

        Returns:
            Meilleur coup selon l'IA, ou None si erreur
        """
        # Utiliser le trainer hybride si activé
        if self.use_hybrid_trainer and self.hybrid_trainer:
            try:
                # Utiliser MCTS correctement : run puis select_move
                move_distribution = self.hybrid_trainer.mcts.run(
                    board, num_simulations=200
                )
                move = self.hybrid_trainer.mcts.select_move(
                    move_distribution, temperature=0.1
                )
                if move:
                    print(f"🚀 Trainer Hybride: {move}")
                return move
            except Exception as e:
                print(f"⚠️ Erreur Trainer Hybride, fallback sur trainer standard: {e}")

        # Utiliser le trainer standard
        if self.trainer:
            try:
                # Utiliser MCTS correctement : run puis select_move
                move_distribution = self.trainer.mcts.run(board, num_simulations=200)
                move = self.trainer.mcts.select_move(move_distribution, temperature=0.1)
                if move:
                    print(f"🤖 Trainer Standard: {move}")
                return move
            except Exception as e:
                print(f"⚠️ Erreur Trainer Standard: {e}")

        # Fallback sur l'ancien système d'IA si disponible
        if self.use_hybrid_ai and self.hybrid_ai_enabled:
            try:
                # Créer le contexte de jeu
                context = GameContext(
                    time_left=time_budget,
                    move_number=len(board.move_stack) + 1,
                    is_critical=board.is_check() or len(list(board.legal_moves)) < 5,
                )

                # Obtenir la décision de l'IA hybride
                decision = self.hybrid_ai.get_move(
                    board, context, mode=self.current_ai_mode
                )

                # Afficher les informations de la décision
                print(
                    f"� IA Hybride Legacy: {decision.move} (eval: {decision.evaluation:.3f}, "
                    f"confiance: {decision.confidence:.3f}, méthode: {decision.method_used})"
                )

                return decision.move

            except Exception as e:
                print(f"⚠️ Erreur IA Hybride Legacy: {e}")

        # Dernier fallback sur l'IA standard
        if self.ai_enabled and self.ai_player:
            try:
                move = self.ai_player.get_move(board)
                if move:
                    print(f"🔄 IA Standard Legacy: {move}")
                return move
            except Exception as e:
                print(f"⚠️ Erreur IA Standard Legacy: {e}")

        return None

    def get_ai_analysis(self, board: chess.Board) -> Optional[dict]:
        """
        Obtient une analyse de position de l'IA.

        Args:
            board: Position à analyser

        Returns:
            Dictionnaire d'analyse ou None
        """
        if self.use_hybrid_ai and self.hybrid_ai_enabled:
            try:
                # Analyse avec IA hybride
                context = GameContext(
                    time_left=15.0,  # 🚀 Plus de temps = Force supérieure (depth 15)
                    move_number=len(board.move_stack) + 1,
                )

                decision = self.hybrid_ai.get_move(
                    board, context, mode=AIMode.HYBRID_DEEP
                )

                return {
                    "best_move": decision.move,
                    "evaluation": decision.evaluation,
                    "confidence": decision.confidence,
                    "method": decision.method_used,
                    "principal_variation": decision.principal_variation,
                    "alternatives": decision.alternative_moves,
                }

            except Exception as e:
                print(f"⚠️ Erreur analyse IA Hybride: {e}")

        # Fallback sur IA standard
        if self.ai_enabled and self.ai_player:
            try:
                return self.ai_player.get_move_analysis(board)
            except Exception as e:
                print(f"⚠️ Erreur analyse IA Standard: {e}")

        return None

    def toggle_ai_mode(self):
        """Change le mode de l'IA hybride."""
        if not self.hybrid_ai_enabled:
            return

        modes = list(AIMode)
        current_index = modes.index(self.current_ai_mode)
        self.current_ai_mode = modes[(current_index + 1) % len(modes)]

        print(f"🔄 Mode IA: {self.current_ai_mode.value}")

    def toggle_ai_continuous(self):
        """Active/désactive l'IA continue."""
        self.ai_auto_play = not self.ai_auto_play
        if self.ai_auto_play:
            print("🤖 IA CONTINUE ACTIVÉE - L'IA jouera automatiquement ses coups")
            print("   • L'IA joue automatiquement après vos coups")
            print(
                "   • Délai configurable avec +/- (actuellement {}ms)".format(
                    self.ai_delay
                )
            )
            # Initialiser le système de collecte de partie si pas déjà fait
            if not self.current_game_moves:
                self.start_new_game_collection()

            # Si on ne connaît pas encore les couleurs, essayer de détecter
            if self.ai_color is None:
                # Si c'est le début de partie et que l'IA a déjà joué, elle a les blancs
                if len(self.environment.board.move_stack) == 1:
                    self.ai_color = chess.WHITE
                    self.player_color = chess.BLACK
                    print("🎯 Détection: IA = Blancs, Joueur = Noirs")
                # Sinon, par défaut l'IA prend la couleur du tour actuel
                elif len(self.environment.board.move_stack) > 0:
                    self.ai_color = self.environment.board.turn
                    self.player_color = not self.environment.board.turn
                    ai_name = "Blancs" if self.ai_color else "Noirs"
                    player_name = "Blancs" if self.player_color else "Noirs"
                    print(f"🎯 Détection: IA = {ai_name}, Joueur = {player_name}")

            # Si c'est déjà le tour de l'IA, la déclencher
            if (
                self.ai_color is not None
                and self.environment.board.turn == self.ai_color
                and not self.environment.board.is_game_over()
                and not self.waiting_for_ai
            ):
                self.waiting_for_ai = True
                self.ai_move_time = pygame.time.get_ticks() + self.ai_delay
                print(f"⏳ L'IA va jouer dans {self.ai_delay}ms...")
        else:
            print("🎮 IA MANUELLE - Appuyez sur 'I' pour chaque coup IA")
            self.waiting_for_ai = False

    def toggle_learning_mode(self):
        """Active/désactive l'apprentissage pendant le jeu."""
        self.learning_from_games = not self.learning_from_games
        if self.learning_from_games:
            print("🧠 APPRENTISSAGE ACTIVÉ - L'IA apprend de vos parties")
            print("   • Les parties sont collectées pour l'entraînement")
            print("   • L'IA s'améliore au fil des parties")
            if not self.current_game_moves:
                self.start_new_game_collection()
        else:
            print("🎯 MODE NORMAL - Pas d'apprentissage")

    def start_new_game_collection(self):
        """Démarre la collecte d'une nouvelle partie."""
        self.current_game_moves = []
        self.game_start_time = pygame.time.get_ticks()

        # Message adapté selon l'entraîneur disponible
        trainer_type = (
            "HYBRIDE avec adaptation automatique"
            if self.hybrid_trainer
            else "CLASSIQUE"
        )
        print(f"📝 Nouvelle partie - L'IA apprendra avec l'entraîneur {trainer_type}")
        if self.hybrid_trainer:
            print(f"   🔄 Adaptation automatique active")
            print(f"   🎯 Évaluations Stockfish précises")

    def add_move_to_collection(self, move, board_before_move):
        """Ajoute un mouvement à la collection de la partie actuelle."""
        if self.learning_from_games:
            move_data = {
                "move": move,
                "board_fen": board_before_move.fen(),
                "timestamp": pygame.time.get_ticks() - self.game_start_time,
                "player": "human" if board_before_move.turn else "ai",
            }
            self.current_game_moves.append(move_data)

    def finish_game_collection(self, result):
        """Termine la collecte de la partie et la sauvegarde pour l'entraînement."""
        if self.learning_from_games and self.current_game_moves:
            game_data = {
                "moves": self.current_game_moves,
                "result": result,
                "duration": pygame.time.get_ticks() - self.game_start_time,
                "timestamp": pygame.time.get_ticks(),
            }
            self.collected_games.append(game_data)

            print(
                f"📚 Partie sauvegardée ({len(self.current_game_moves)} coups, résultat: {result})"
            )
            print(f"   Total parties collectées: {len(self.collected_games)}")

            # 🚀 NOUVEAU : Entraîner après CHAQUE partie (pas seulement après 5)
            if (
                self.trainer and len(self.current_game_moves) > 5
            ):  # Parties assez longues
                print("🎓 Entraînement immédiat sur la partie terminée...")
                self.run_micro_training()

            # Redémarrer la collecte pour la prochaine partie
            self.start_new_game_collection()

    def run_micro_training(self):
        """Lance un micro-entraînement avec les parties collectées."""
        # 🚀 UTILISER L'ENTRAÎNEUR HYBRIDE PAR DÉFAUT si disponible
        current_trainer = self.hybrid_trainer if self.hybrid_trainer else self.trainer

        if not current_trainer or len(self.collected_games) < 2:
            return

        try:
            trainer_type = (
                "HYBRIDE avec adaptation automatique"
                if self.hybrid_trainer
                else "CLASSIQUE"
            )
            print(
                f"🏋️ MICRO-ENTRAÎNEMENT {trainer_type} - Apprentissage des dernières parties..."
            )

            # 🚀 VRAI ENTRAÎNEMENT avec les parties collectées !
            from ..ai.training import GameData
            from ..ai.hybrid_training import HybridGameData
            from ..ai.network import encode_board

            games_trained = 0
            total_loss = 0.0

            for game_data in self.collected_games[
                -3:
            ]:  # Prendre les 3 dernières parties
                try:
                    # Reconstruire la partie depuis les coups stockés
                    board = chess.Board()
                    positions = []
                    board_positions = []
                    moves = []

                    for move_info in game_data["moves"]:
                        if "move" in move_info:
                            move = move_info["move"]
                            positions.append(encode_board(board))
                            board_positions.append(board.copy())
                            moves.append(move.uci())
                            board.push(move)

                    if len(positions) > 5:  # Seulement si la partie est assez longue
                        result = game_data["result"]

                        # 🚀 UTILISER L'ENTRAÎNEUR HYBRIDE EN PRIORITÉ
                        if self.hybrid_trainer and hasattr(
                            self.hybrid_trainer, "train_single_game_hybrid"
                        ):
                            # Calculer valeurs Stockfish
                            values = self.hybrid_trainer._calculate_stockfish_values(
                                board_positions, result
                            )

                            # Politiques simplifiées
                            policies = []
                            stockfish_policies = []

                            for board_pos in board_positions:
                                legal_moves = list(board_pos.legal_moves)
                                if legal_moves:
                                    uniform_prob = 1.0 / len(legal_moves)
                                    dummy_policy = {
                                        move: uniform_prob for move in legal_moves
                                    }
                                    policies.append(dummy_policy)

                                    # Politique Stockfish
                                    try:
                                        best_move = self.hybrid_trainer.reference_evaluator.get_best_move(
                                            board_pos
                                        )
                                        if best_move and best_move in legal_moves:
                                            sf_policy = {}
                                            for move in legal_moves:
                                                if move == best_move:
                                                    sf_policy[move] = 0.8
                                                else:
                                                    sf_policy[move] = 0.2 / (
                                                        len(legal_moves) - 1
                                                    )
                                            stockfish_policies.append(sf_policy)
                                        else:
                                            stockfish_policies.append(dummy_policy)
                                    except:
                                        stockfish_policies.append(dummy_policy)
                                else:
                                    policies.append({})
                                    stockfish_policies.append({})

                            hybrid_game_data = HybridGameData(
                                positions=positions,
                                policies=policies,
                                values=values,
                                result=result,
                                moves=moves,
                                game_length=len(moves),
                                stockfish_policies=stockfish_policies,
                            )

                            metrics = self.hybrid_trainer.train_single_game_hybrid(
                                hybrid_game_data, verbose=False
                            )
                            print(
                                f"  ✅ Partie {games_trained + 1} entrainée HYBRIDE (Loss: {metrics.get('total_loss', 0):.4f})"
                            )

                        elif current_trainer and hasattr(
                            current_trainer, "train_single_game"
                        ):
                            # Entraînement classique en fallback
                            values = current_trainer._calculate_stockfish_values(
                                board_positions, result
                            )

                            policies = []
                            for board_pos in board_positions:
                                legal_moves = list(board_pos.legal_moves)
                                if legal_moves:
                                    uniform_prob = 1.0 / len(legal_moves)
                                    dummy_policy = {
                                        move: uniform_prob for move in legal_moves
                                    }
                                    policies.append(dummy_policy)
                                else:
                                    policies.append({})

                            classic_game_data = GameData(
                                positions=positions,
                                policies=policies,
                                values=values,
                                result=result,
                                moves=moves,
                                game_length=len(moves),
                            )

                            metrics = current_trainer.train_single_game(
                                classic_game_data, verbose=False
                            )
                            print(
                                f"  ✅ Partie {games_trained + 1} entrainée CLASSIQUE (Loss: {metrics.get('total_loss', 0):.4f})"
                            )

                        total_loss += metrics.get("total_loss", 0)
                        games_trained += 1

                except Exception as e:
                    print(f"  ⚠️ Erreur partie {game_data.get('result', '?')}: {e}")
                    continue

            if games_trained > 0:
                avg_loss = total_loss / games_trained
                print(
                    f"✨ Micro-entraînement terminé! {games_trained} parties entrainées"
                )
                print(f"   Loss moyenne: {avg_loss:.4f}")
                if self.hybrid_trainer:
                    print(f"   🔄 Adaptation automatique appliquée")
                print(f"   L'IA s'améliore en continu ! 🚀")
            else:
                print("⚠️ Aucune partie n'a pu être entrainée")

            # Garder seulement les 10 dernières parties pour éviter l'accumulation
            if len(self.collected_games) > 10:
                self.collected_games = self.collected_games[-10:]

        except Exception as e:
            print(f"⚠️ Erreur pendant le micro-entraînement: {e}")
            import traceback

            traceback.print_exc()

    def handle_ai_continuous_play(self):
        """Gère l'IA continue contre le joueur."""
        current_time = pygame.time.get_ticks()
        board = self.environment.board

        # Si les couleurs ne sont pas encore détectées, ne pas agir
        if self.ai_color is None:
            return

        # Vérifier si c'est au tour de l'IA et qu'on attend son coup
        if (
            board.turn == self.ai_color  # Tour de l'IA
            and not self.waiting_for_ai
            and not board.is_game_over()
            and not self.training_mode
        ):  # Pas en mode entraînement

            # Démarrer l'attente du coup IA
            self.waiting_for_ai = True
            self.ai_move_time = current_time + self.ai_delay
            ai_color_name = "Blancs" if self.ai_color else "Noirs"
            print(f"⏳ L'IA ({ai_color_name}) réfléchit... (délai: {self.ai_delay}ms)")

        # Exécuter le coup IA si le délai est écoulé
        elif (
            self.waiting_for_ai
            and current_time >= self.ai_move_time
            and not board.is_game_over()
        ):

            self.execute_ai_move()
            self.waiting_for_ai = False

    def execute_ai_move(self):
        """Exécute un coup de l'IA."""
        board = self.environment.board

        try:
            # Sauvegarder l'état avant le coup pour la collecte
            board_before = board.copy()

            # Obtenir le coup de l'IA
            move = self.get_ai_move(board, time_budget=2.0)

            if move and move in board.legal_moves:
                # Ajouter le coup à la collection
                self.add_move_to_collection(move, board_before)

                # Jouer le coup
                board.push(move)

                print(f"🤖 IA: {move}")

                # Reset des sélections
                self.selected_square = None
                self.possible_moves = []

                # Mettre à jour l'évaluation
                self.update_evaluation()

                # Vérifier si la partie est terminée
                if board.is_game_over():
                    result = board.result()
                    print(f"🏁 Partie terminée: {result}")
                    self.finish_game_collection(result)

            else:
                print("❌ L'IA n'a pas pu trouver de coup valide")

        except Exception as e:
            print(f"❌ Erreur lors du coup IA: {e}")
            self.waiting_for_ai = False

    def update_evaluation(self):
        """Met à jour les évaluations de la position actuelle."""
        # 1. Évaluation de référence (Stockfish)
        if self.reference_evaluator:
            try:
                self.reference_evaluation = self.reference_evaluator.evaluate_position(
                    self.environment.board
                )
                self.reference_history.append(self.reference_evaluation)
                if len(self.reference_history) > 100:
                    self.reference_history = self.reference_history[-100:]
            except Exception as e:
                print(f"⚠️  Erreur évaluation référence : {e}")

        # 2. Évaluation de l'IA (si disponible)
        if self.ai_player:
            try:
                if hasattr(self.ai_player, "evaluate_position"):
                    value, _ = self.ai_player.evaluate_position(self.environment.board)
                    self.current_evaluation = max(-1.0, min(1.0, value))
                elif hasattr(self.ai_player, "get_move_analysis"):
                    analysis = self.ai_player.get_move_analysis(self.environment.board)
                    self.current_evaluation = max(
                        -1.0, min(1.0, analysis.get("evaluation", 0.0))
                    )

                # Ajouter à l'historique
                self.evaluation_history.append(self.current_evaluation)
                if len(self.evaluation_history) > 100:
                    self.evaluation_history = self.evaluation_history[-100:]

            except Exception as e:
                # En cas d'erreur, garder la dernière évaluation
                pass

    def draw_evaluation_bar(self):
        """Dessine les barres d'évaluation sur le côté droit."""
        if not self.show_evaluation_bar:
            return

        # Position des barres (côté droit)
        board_start_x = (self.WINDOW_WIDTH - self.BOARD_SIZE) // 2
        bars_start_x = board_start_x + self.BOARD_SIZE + 60
        bar_y = (self.WINDOW_HEIGHT - self.eval_bar_height) // 2

        # Dessiner barre de référence (gauche)
        self._draw_single_evaluation_bar(
            bars_start_x,
            bar_y,
            self.reference_evaluation,
            self.reference_history,
            "Référence",
            "Stockfish" if self.reference_evaluator else "Basique",
            "ref",
        )

        # Dessiner barre IA (droite)
        ai_bar_x = bars_start_x + self.eval_bar_width + 20
        self._draw_single_evaluation_bar(
            ai_bar_x,
            bar_y,
            self.current_evaluation,
            self.evaluation_history,
            "IA AlphaZero",
            "En apprentissage",
            "ai",
        )

        # Légende générale
        legend_x = bars_start_x + self.eval_bar_width
        legend_y = bar_y + self.eval_bar_height + 50

        # Afficher la différence
        diff = abs(self.reference_evaluation - self.current_evaluation)
        diff_text = f"Écart: {diff:.2f}"
        diff_color = (
            (255, 100, 100)
            if diff > 0.5
            else (255, 255, 100) if diff > 0.2 else (100, 255, 100)
        )

        diff_surface = pygame.font.Font(None, 20).render(diff_text, True, diff_color)
        diff_rect = diff_surface.get_rect(center=(legend_x, legend_y))
        self.screen.blit(diff_surface, diff_rect)

    def _draw_single_evaluation_bar(
        self, x, y, evaluation, history, title, subtitle, color_prefix
    ):
        """Dessine une barre d'évaluation individuelle."""
        # Couleurs selon le préfixe
        border_color = self.COLORS[f"{color_prefix}_border"]
        black_color = self.COLORS[f"{color_prefix}_black"]
        white_color = self.COLORS[f"{color_prefix}_white"]

        # Fond de la barre
        bar_rect = pygame.Rect(x, y, self.eval_bar_width, self.eval_bar_height)
        pygame.draw.rect(self.screen, border_color, bar_rect, 2)

        # Calculer la hauteur des sections blanc/noir
        normalized_eval = (evaluation + 1.0) / 2.0  # 0 à 1
        white_height = int(normalized_eval * self.eval_bar_height)
        black_height = self.eval_bar_height - white_height

        # Dessiner la section noire (en haut)
        if black_height > 0:
            black_rect = pygame.Rect(
                x + 2, y + 2, self.eval_bar_width - 4, black_height - 2
            )
            pygame.draw.rect(self.screen, black_color, black_rect)

        # Dessiner la section blanche (en bas)
        if white_height > 0:
            white_rect = pygame.Rect(
                x + 2, y + black_height, self.eval_bar_width - 4, white_height - 2
            )
            pygame.draw.rect(self.screen, white_color, white_rect)

        # Ligne de milieu (égalité)
        middle_y = y + self.eval_bar_height // 2
        pygame.draw.line(
            self.screen,
            (255, 255, 0),
            (x, middle_y),
            (x + self.eval_bar_width, middle_y),
            2,
        )

        # Titre
        title_surface = pygame.font.Font(None, 20).render(
            title, True, self.COLORS["eval_text"]
        )
        title_rect = title_surface.get_rect(
            center=(x + self.eval_bar_width // 2, y - 40)
        )
        self.screen.blit(title_surface, title_rect)

        # Sous-titre
        subtitle_surface = pygame.font.Font(None, 16).render(
            subtitle, True, (200, 200, 200)
        )
        subtitle_rect = subtitle_surface.get_rect(
            center=(x + self.eval_bar_width // 2, y - 25)
        )
        self.screen.blit(subtitle_surface, subtitle_rect)

        # Valeur numérique
        eval_text = f"{evaluation:+.2f}"
        eval_surface = pygame.font.Font(None, 18).render(
            eval_text, True, self.COLORS["eval_text"]
        )
        eval_rect = eval_surface.get_rect(center=(x + self.eval_bar_width // 2, y - 8))
        self.screen.blit(eval_surface, eval_rect)

        # Mini-graphique historique (en dessous)
        if len(history) > 1:
            history_y = y + self.eval_bar_height + 10
            history_height = 30

            # Fond du mini-graphique
            history_rect = pygame.Rect(
                x, history_y, self.eval_bar_width, history_height
            )
            pygame.draw.rect(self.screen, (40, 40, 40), history_rect)
            pygame.draw.rect(self.screen, border_color, history_rect, 1)

            # Ligne de milieu
            middle_y = history_y + history_height // 2
            pygame.draw.line(
                self.screen,
                (100, 100, 100),
                (x, middle_y),
                (x + self.eval_bar_width, middle_y),
                1,
            )

            # Dessiner la courbe
            points = []
            recent_history = history[-20:]  # 20 derniers points
            for i, eval_val in enumerate(recent_history):
                point_x = x + (i * self.eval_bar_width) // len(recent_history)
                normalized = (eval_val + 1.0) / 2.0  # 0 à 1
                point_y = history_y + history_height - int(normalized * history_height)
                points.append((point_x, point_y))

            if len(points) > 1:
                color = (100, 255, 150) if color_prefix == "ref" else (255, 200, 100)
                pygame.draw.lines(self.screen, color, False, points, 2)

    def draw_ui(self):
        """Dessine l'interface utilisateur."""
        # Informations sur le jeu
        turn_text = "Tour: " + ("Blancs" if self.environment.board.turn else "Noirs")
        text_surface = self.font.render(turn_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        # Mode d'entraînement et IA continue
        if self.training_mode:
            if self.auto_training_active:
                mode_text = "ENTRAÎNEMENT AUTOMATIQUE"
                mode_color = (255, 215, 0)  # Or
            else:
                mode_text = "MODE ENTRAÎNEMENT"
                mode_color = (255, 255, 0)  # Jaune

            mode_surface = self.font.render(mode_text, True, mode_color)
            self.screen.blit(mode_surface, (10, 40))

            # Informations d'entraînement
            if self.auto_training_active:
                training_info = [
                    f"Itération: {self.auto_training_iteration + 1}/{self.auto_training_max_iterations}",
                    f"Partie: {self.auto_training_game_count}/5",
                    f"Status: {'En pause' if self.auto_training_paused else 'Actif'}",
                    f"Vitesse: {self.auto_training_speed}ms",
                ]

                for i, info in enumerate(training_info):
                    color = (
                        (255, 255, 255)
                        if not self.auto_training_paused
                        else (255, 255, 0)
                    )
                    info_surface = pygame.font.Font(None, 24).render(info, True, color)
                    self.screen.blit(info_surface, (10, 70 + i * 25))
            else:
                training_info = [
                    f"Itération: {self.training_iteration + 1}",
                    f"Parties: {self.training_games_played}/{self.trainer.config.games_per_iteration if self.trainer else 0}",
                    f"Status: {'Auto-jeu' if self.training_auto_play else 'En pause'}",
                ]

                for i, info in enumerate(training_info):
                    color = (0, 255, 0) if self.training_auto_play else (255, 255, 255)
                    info_surface = pygame.font.Font(None, 24).render(info, True, color)
                    self.screen.blit(info_surface, (10, 70 + i * 25))

        # Affichage des modes IA continue et apprentissage
        if self.ai_auto_play or self.learning_from_games:
            y_offset = 70 if not self.training_mode else 170

            if self.ai_auto_play:
                ai_text = "🤖 IA CONTINUE ACTIVÉE"
                ai_color = (0, 255, 0)  # Vert
                if self.waiting_for_ai:
                    ai_text += " (⏳ réflexion...)"
                    ai_color = (255, 255, 0)  # Jaune pendant la réflexion
                ai_surface = self.font.render(ai_text, True, ai_color)
                self.screen.blit(ai_surface, (10, y_offset))
                y_offset += 30

                # Délai configuré
                delay_text = f"Délai IA: {self.ai_delay}ms (+/- pour ajuster)"
                delay_surface = pygame.font.Font(None, 20).render(
                    delay_text, True, (200, 200, 200)
                )
                self.screen.blit(delay_surface, (10, y_offset))
                y_offset += 25

            if self.learning_from_games:
                learn_text = (
                    f"🧠 APPRENTISSAGE ACTIF ({len(self.collected_games)} parties)"
                )
                learn_color = (100, 255, 100)  # Vert clair
                learn_surface = self.font.render(learn_text, True, learn_color)
                self.screen.blit(learn_surface, (10, y_offset))
                y_offset += 30

                if self.current_game_moves:
                    moves_text = (
                        f"Partie actuelle: {len(self.current_game_moves)} coups"
                    )
                    moves_surface = pygame.font.Font(None, 20).render(
                        moves_text, True, (200, 200, 200)
                    )
                    self.screen.blit(moves_surface, (10, y_offset))

        # Contrôles
        controls = [
            "Contrôles:",
            "Clic gauche: Sélectionner/Jouer",
            "Clic droit + glisser: Rotation caméra",
            "Molette: Zoom",
            "R: Réinitialiser caméra",
            "H: Toggle hints IA",
            "I: Jouer coup IA",
            "C: IA continue ON/OFF",
            "L: Apprentissage ON/OFF",
            "M: Changer mode IA",
            "E: Toggle barre d'évaluation",
        ]

        # Ajouter contrôles d'entraînement
        if TRAINING_AVAILABLE and self.trainer:
            controls.extend(
                [
                    "T: Mode entraînement",
                    "Y: Entraînement automatique",
                    "S: Démarrer/Arrêter auto-jeu",
                    "Espace: Pause/Reprendre",
                    "+/-: Vitesse entraînement",
                    "Échap: Arrêter entraînement auto",
                ]
            )

        start_y = 160 if self.training_mode else 50
        for i, control in enumerate(controls):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            if control.startswith("H:") and self.show_ai_hints:
                color = (0, 255, 0)  # Vert si hints actifs
            elif control.startswith("T:") and self.training_mode:
                color = (255, 255, 0)  # Jaune si mode entraînement actif
            text_surface = pygame.font.Font(None, 24).render(control, True, color)
            self.screen.blit(text_surface, (10, start_y + i * 25))

        # Affichage du mode IA hybride
        if self.hybrid_ai:
            mode_y = start_y + len(controls) * 25 + 10
            mode_text = f"Mode IA: {self.hybrid_ai.current_mode.value}"
            mode_color = (100, 255, 100)  # Vert clair
            mode_surface = self.font.render(mode_text, True, mode_color)
            self.screen.blit(mode_surface, (10, mode_y))

        # Afficher les suggestions IA si actives
        if self.show_ai_hints and self.ai_analysis:
            y_pos = (
                start_y + len(controls) * 25 + 50
            )  # +50 pour laisser place au mode IA
            title = self.font.render("Suggestions IA:", True, (255, 255, 255))
            self.screen.blit(title, (10, y_pos))

            for i, (move, prob) in enumerate(self.ai_analysis["top_moves"][:3]):
                move_text = f"{i+1}. {move} ({prob:.1%})"
                surface = pygame.font.Font(None, 24).render(
                    move_text, True, self.COLORS["ai_suggestion"][:3]
                )
                self.screen.blit(surface, (10, y_pos + 25 + i * 20))

        # État du jeu
        if self.environment.board.is_checkmate():
            winner = "Noirs" if self.environment.board.turn else "Blancs"
            game_text = self.font.render(
                f"Échec et mat! {winner} gagnent!", True, (255, 0, 0)
            )
            self.screen.blit(game_text, (10, self.WINDOW_HEIGHT - 30))
        elif self.environment.board.is_stalemate():
            game_text = self.font.render("Pat! Match nul.", True, (255, 255, 0))
            self.screen.blit(game_text, (10, self.WINDOW_HEIGHT - 30))
        elif self.environment.board.is_check():
            game_text = self.font.render("Échec!", True, (255, 100, 100))
            self.screen.blit(game_text, (10, self.WINDOW_HEIGHT - 30))

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
                        # Sauvegarder l'état avant le coup pour la collecte
                        board_before = self.environment.board.copy()

                        # Détecter le type de mouvement avant de l'effectuer
                        is_castling = self.environment.board.is_castling(target_move)
                        is_en_passant = self.environment.board.is_en_passant(
                            target_move
                        )

                        if self.environment.make_move(target_move):
                            # Ajouter le coup à la collection si l'apprentissage est activé
                            self.add_move_to_collection(target_move, board_before)

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

                            # Mettre à jour l'évaluation après le coup
                            self.update_evaluation()
                            self.legal_moves_from_selected = []

                            # Vérifier si la partie est terminée
                            if self.environment.board.is_game_over():
                                result = self.environment.board.result()
                                print(f"🏁 Partie terminée: {result}")
                                self.finish_game_collection(result)
                            else:
                                # Détecter automatiquement les couleurs des joueurs
                                if self.player_color is None:
                                    # Le joueur vient de jouer, donc c'est sa couleur
                                    self.player_color = (
                                        not self.environment.board.turn
                                    )  # Couleur opposée au tour actuel
                                    self.ai_color = (
                                        self.environment.board.turn
                                    )  # Tour actuel = IA
                                    print(
                                        f"🎯 Détection: Joueur = {'Blancs' if self.player_color else 'Noirs'}, IA = {'Blancs' if self.ai_color else 'Noirs'}"
                                    )

                                # Déclencher l'IA si mode continu activé et c'est son tour
                                if (
                                    self.ai_auto_play
                                    and self.environment.board.turn == self.ai_color
                                    and not self.training_mode
                                    and not self.waiting_for_ai
                                ):
                                    self.waiting_for_ai = True
                                    self.ai_move_time = (
                                        pygame.time.get_ticks() + self.ai_delay
                                    )
                                    print(f"⏳ L'IA va jouer dans {self.ai_delay}ms...")
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
                    elif event.key == pygame.K_h:
                        # Toggle pour afficher/masquer les hints IA
                        self.show_ai_hints = not self.show_ai_hints
                        if self.show_ai_hints and not self.ai_analysis:
                            # Analyser la position
                            self.ai_analysis = self.get_ai_analysis(
                                self.environment.board
                            )
                    elif event.key == pygame.K_i:
                        # Jouer le meilleur coup IA
                        if not self.environment.board.is_game_over():
                            move = self.get_ai_move(
                                self.environment.board, time_budget=3.0
                            )
                            if move and move in self.environment.board.legal_moves:
                                self.environment.board.push(move)
                                self.selected_square = None
                                self.possible_moves = []
                                self.ai_analysis = None  # Reset analysis après un coup
                                self.update_evaluation()  # Mettre à jour l'évaluation
                    elif event.key == pygame.K_m:
                        # Changer le mode de l'IA hybride
                        self.toggle_ai_mode()
                    elif event.key == pygame.K_t:
                        # Toggle mode d'entraînement
                        if TRAINING_AVAILABLE and self.trainer:
                            self.toggle_training_mode()
                    elif event.key == pygame.K_y:
                        # Lancer l'entraînement automatique jusqu'à convergence
                        if (
                            TRAINING_AVAILABLE
                            and self.trainer
                            and not self.auto_training_active
                        ):
                            self.start_automatic_training()
                        elif self.auto_training_active:
                            print("🛑 Arrêt de l'entraînement automatique")
                            self.auto_training_active = False
                            self.training_mode = False
                    elif event.key == pygame.K_e:
                        # Toggle barre d'évaluation
                        self.show_evaluation_bar = not self.show_evaluation_bar
                        status = "activée" if self.show_evaluation_bar else "désactivée"
                        print(f"📊 Barre d'évaluation {status}")
                    elif event.key == pygame.K_s:
                        # Démarrer/Arrêter auto-jeu d'entraînement
                        if self.training_mode:
                            if not self.training_auto_play:
                                self.start_training_iteration()
                            else:
                                self.training_auto_play = False
                                print("⏸️  Auto-jeu en pause")
                    elif event.key == pygame.K_SPACE:
                        # Pause/Reprendre en mode entraînement
                        if self.training_mode or self.auto_training_active:
                            if self.auto_training_active:
                                self.auto_training_paused = (
                                    not self.auto_training_paused
                                )
                                status = (
                                    "repris"
                                    if not self.auto_training_paused
                                    else "en pause"
                                )
                                print(f"⏯️  Entraînement automatique {status}")
                            else:
                                self.training_auto_play = not self.training_auto_play
                                status = (
                                    "repris" if self.training_auto_play else "en pause"
                                )
                                print(f"⏯️  Auto-jeu {status}")
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Accélérer l'entraînement OU réduire le délai IA
                        if self.auto_training_active:
                            self.auto_training_speed = max(
                                100, self.auto_training_speed - 100
                            )
                            print(
                                f"⚡ Vitesse d'entraînement: {self.auto_training_speed}ms"
                            )
                        elif self.ai_auto_play:
                            self.ai_delay = max(500, self.ai_delay - 200)
                            print(f"⚡ Délai IA réduit: {self.ai_delay}ms")
                    elif event.key == pygame.K_MINUS:
                        # Ralentir l'entraînement OU augmenter le délai IA
                        if self.auto_training_active:
                            self.auto_training_speed = min(
                                2000, self.auto_training_speed + 100
                            )
                            print(
                                f"🐌 Vitesse d'entraînement: {self.auto_training_speed}ms"
                            )
                        elif self.ai_auto_play:
                            self.ai_delay = min(5000, self.ai_delay + 200)
                            print(f"🐌 Délai IA augmenté: {self.ai_delay}ms")
                    elif event.key == pygame.K_ESCAPE:
                        # Arrêter l'entraînement automatique
                        if self.auto_training_active:
                            print("🛑 Arrêt de l'entraînement automatique demandé")
                            print("💾 Sauvegarde d'urgence du modèle...")

                            # 🚀 SAUVEGARDE D'URGENCE !
                            current_trainer = (
                                self.hybrid_trainer
                                if self.use_hybrid_trainer
                                else self.trainer
                            )
                            if current_trainer and hasattr(
                                current_trainer, "save_model"
                            ):
                                try:
                                    current_trainer.save_model(
                                        f"hybrid_model_emergency_iter_{self.auto_training_iteration}.pt"
                                    )
                                    print("✅ Modèle sauvegardé avec succès!")
                                except Exception as e:
                                    print(f"⚠️ Erreur de sauvegarde: {e}")

                            self.auto_training_active = False
                            self.training_mode = False
                    elif event.key == pygame.K_c:
                        # Toggle IA continue
                        self.toggle_ai_continuous()
                    elif event.key == pygame.K_l:
                        # Toggle apprentissage pendant le jeu
                        self.toggle_learning_mode()

            # Gestion de l'auto-jeu d'entraînement
            if self.training_mode and not self.auto_training_active:
                self.handle_training_auto_play()

            # Gestion de l'entraînement automatique
            if self.auto_training_active and not self.auto_training_paused:
                self.handle_automatic_training()

            # Gestion de l'IA continue contre le joueur
            if self.ai_auto_play and not self.training_mode:
                self.handle_ai_continuous_play()

            # Dessiner
            self.screen.fill(self.COLORS["background"])
            self.draw_board()
            self.draw_coordinates()
            self.draw_highlights()

            # Dessiner les pièces
            for square in chess.SQUARES:
                piece = self.environment.board.piece_at(square)
                if piece:
                    self.draw_piece(piece, square)

            self.draw_evaluation_bar()
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
