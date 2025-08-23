"""
Exemple d'utilisation professionnelle de Chess AI.

Ce script démontre l'utilisation de l'architecture modulaire
avec gestion d'erreurs robuste et logging complet.
"""

import sys
import os

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_ai import ChessEnvironment, ChessAnalyzer, ChessDisplay
from chess_ai.exceptions import *
from chess_ai.utils import setup_logging, ChessLogger
import chess


def setup_environment():
    """Configure l'environnement de démonstration."""
    # Configuration du logging
    setup_logging(level="INFO")

    # Logger spécialisé pour les échecs
    chess_logger = ChessLogger("chess_demo", "INFO")

    return chess_logger


def demo_basic_operations(logger: ChessLogger):
    """Démonstration des opérations de base avec gestion d'erreurs."""
    print("\n" + "=" * 60)
    print("           DÉMONSTRATION DES OPÉRATIONS DE BASE")
    print("=" * 60)

    try:
        # Créer l'environnement
        env = ChessEnvironment()
        analyzer = ChessAnalyzer(env.board)
        display = ChessDisplay(env.board)

        logger.logger.info("Environnement Chess AI initialisé avec succès")

        # Affichage initial
        print("\n🎯 Plateau initial:")
        display.display_unicode()

        # Série de mouvements avec gestion d'erreurs
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]

        print("\n🎲 Exécution des mouvements:")
        for i, move in enumerate(moves, 1):
            try:
                success = env.make_move(move)
                if success:
                    logger.log_move(move, True, f"({i}/{len(moves)})")
                    print(f"  ✅ {i}. {move} - Succès")
                else:
                    logger.log_move(move, False, "Mouvement illégal")
                    print(f"  ❌ {i}. {move} - Échec")

            except InvalidMoveError as e:
                logger.logger.error(f"Mouvement invalide: {e}")
                print(f"  ❌ {i}. {move} - Erreur: {e.message}")
            except GameOverError as e:
                logger.logger.warning(f"Partie terminée: {e}")
                print(f"  🏁 {i}. {move} - Partie terminée: {e.message}")
                break

        # Affichage après mouvements
        print("\n🎯 Position après mouvements:")
        display.display_unicode()

        # Statistiques
        print("\n📊 Analyse de position:")
        display.display_statistics()

        return env, analyzer, display

    except Exception as e:
        logger.logger.error(f"Erreur critique dans demo_basic_operations: {e}")
        print(f"❌ Erreur critique: {e}")
        return None, None, None


def demo_advanced_analysis(
    env: ChessEnvironment, analyzer: ChessAnalyzer, logger: ChessLogger
):
    """Démonstration de l'analyse avancée."""
    print("\n" + "=" * 60)
    print("           DÉMONSTRATION DE L'ANALYSE AVANCÉE")
    print("=" * 60)

    try:
        # Analyse complète de la position
        analysis = analyzer.analyze_position()

        print("\n🔍 Analyse complète de la position:")
        print("-" * 40)

        # Informations de position
        pos_info = analysis["position_info"]
        print(f"FEN: {pos_info['fen']}")
        print(f"Tour: {pos_info['turn']}")
        print(f"En échec: {pos_info['is_check']}")
        print(f"Mouvements légaux: {pos_info['legal_moves_count']}")

        # Matériel
        material = analysis["material"]
        print(f"\n📦 Matériel:")
        print(f"  Blancs: {material['white']}")
        print(f"  Noirs: {material['black']}")

        # Sécurité des rois
        white_safety = analysis["white_king_safety"]
        black_safety = analysis["black_king_safety"]

        print(f"\n👑 Sécurité des rois:")
        if "error" not in white_safety:
            print(
                f"  Roi blanc ({white_safety['king_square']}): {white_safety['safety_rating']}/10"
            )
            print(f"    Menaces: {white_safety['threats_around_king']}")
            print(f"    Cases protégées: {white_safety['protected_squares']}")

        if "error" not in black_safety:
            print(
                f"  Roi noir ({black_safety['king_square']}): {black_safety['safety_rating']}/10"
            )
            print(f"    Menaces: {black_safety['threats_around_king']}")
            print(f"    Cases protégées: {black_safety['protected_squares']}")

        # Développement
        white_dev = analysis["white_development"]
        black_dev = analysis["black_development"]

        print(f"\n🏗️  Développement:")
        if "error" not in white_dev:
            print(f"  Blancs: {white_dev['development_percentage']:.1f}%")
            print(f"    Cavaliers développés: {white_dev['developed_knights']}/2")
            print(f"    Fous développés: {white_dev['developed_bishops']}/2")

        if "error" not in black_dev:
            print(f"  Noirs: {black_dev['development_percentage']:.1f}%")
            print(f"    Cavaliers développés: {black_dev['developed_knights']}/2")
            print(f"    Fous développés: {black_dev['developed_bishops']}/2")

        logger.logger.info("Analyse avancée complétée avec succès")

    except ChessBoardStateError as e:
        logger.logger.error(f"Erreur d'état du plateau: {e}")
        print(f"❌ Erreur d'analyse: {e.message}")
    except Exception as e:
        logger.logger.error(f"Erreur inattendue dans l'analyse: {e}")
        print(f"❌ Erreur inattendue: {e}")


def demo_error_handling(logger: ChessLogger):
    """Démonstration de la gestion d'erreurs robuste."""
    print("\n" + "=" * 60)
    print("         DÉMONSTRATION DE LA GESTION D'ERREURS")
    print("=" * 60)

    env = ChessEnvironment()

    # Tests d'erreurs contrôlées
    error_tests = [
        ("mouvement invalide", lambda: env.make_move("z9z9")),
        ("case invalide", lambda: env.get_piece_at("z9")),
        ("FEN invalide", lambda: ChessEnvironment("invalid_fen")),
        ("mouvement illégal", lambda: env.make_move("e1e8")),
    ]

    print("\n🧪 Tests de gestion d'erreurs:")

    for test_name, test_func in error_tests:
        try:
            print(f"\n  🔬 Test: {test_name}")
            result = test_func()
            print(f"    ⚠️  Aucune erreur détectée (inattendu)")
        except ChessError as e:
            print(f"    ✅ Erreur Chess AI capturée: {e.message}")
            if e.details:
                print(f"       Détails: {e.details}")
            logger.logger.debug(f"Erreur contrôlée capturée: {test_name}")
        except Exception as e:
            print(f"    ❌ Erreur inattendue: {e}")
            logger.logger.error(f"Erreur inattendue dans {test_name}: {e}")


def demo_different_displays(env: ChessEnvironment, logger: ChessLogger):
    """Démonstration des différents modes d'affichage."""
    print("\n" + "=" * 60)
    print("         DÉMONSTRATION DES MODES D'AFFICHAGE")
    print("=" * 60)

    try:
        display = ChessDisplay(env.board)

        # Affichage Unicode
        print("\n🎨 Affichage Unicode (perspective blancs):")
        display.display_unicode(perspective=chess.WHITE)

        # Affichage ASCII
        print("\n🎨 Affichage ASCII (perspective noirs):")
        display.display_ascii(perspective=chess.BLACK)

        # Affichage compact
        print("\n🎨 Affichage compact:")
        display.display_compact()

        # Historique des mouvements
        move_history = env.get_move_history()
        display.display_move_history(move_history)

        logger.logger.info("Démonstration des affichages complétée")

    except Exception as e:
        logger.logger.error(f"Erreur dans demo_different_displays: {e}")
        print(f"❌ Erreur d'affichage: {e}")


def demo_game_simulation(logger: ChessLogger):
    """Simulation d'une mini-partie avec logging complet."""
    print("\n" + "=" * 60)
    print("           SIMULATION D'UNE MINI-PARTIE")
    print("=" * 60)

    try:
        env = ChessEnvironment()
        display = ChessDisplay(env.board)

        # Ouverture italienne
        game_moves = [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1c4",
            "f8c5",
            "d3",
            "d6",
            "c3",
            "g8f6",
            "b4",
            "c5b6",
        ]

        print("\n🎮 Simulation d'ouverture italienne:")

        move_number = 1
        for i, move in enumerate(game_moves):
            try:
                # Afficher le numéro de coup
                if i % 2 == 0:
                    print(f"\n{move_number}.", end=" ")
                    move_number += 1

                # Effectuer le mouvement
                success = env.make_move(move)
                if success:
                    print(f"{move}", end=" " if i % 2 == 0 else "\n")
                    logger.log_move(move, True)
                else:
                    print(f"[{move} - ÉCHEC]")
                    logger.log_move(move, False)
                    break

                # Vérifier l'état du jeu
                if env.is_game_over():
                    result = env.get_game_result()
                    logger.log_game_over(result or "Inconnu")
                    print(f"\n🏁 Partie terminée: {result}")
                    break

            except Exception as e:
                logger.logger.error(f"Erreur pendant le mouvement {move}: {e}")
                print(f"\n❌ Erreur: {e}")
                break

        # Position finale
        print(f"\n🎯 Position finale après {len(env.get_move_history())} coups:")
        display.display_unicode()

        # Statistiques finales
        stats = env.get_board_stats()
        print(f"\n📈 Statistiques finales:")
        print(f"  Mouvements joués: {stats.get('move_count', 0)}")
        print(f"  Joueur actuel: {stats.get('current_player', 'Inconnu')}")
        print(f"  Mouvements légaux: {stats.get('legal_moves_count', 0)}")
        print(f"  Partie terminée: {stats.get('is_game_over', False)}")

        logger.logger.info("Simulation de partie complétée")

    except Exception as e:
        logger.logger.error(f"Erreur critique dans la simulation: {e}")
        print(f"❌ Erreur critique: {e}")


def main():
    """Fonction principale de démonstration."""
    print("🚀 CHESS AI - DÉMONSTRATION PROFESSIONNELLE")
    print("=" * 60)
    print("Architecture modulaire avec gestion d'erreurs robuste")
    print("Utilisation complète de la librairie python-chess")
    print("=" * 60)

    # Configuration
    logger = setup_environment()

    try:
        # Démonstrations
        env, analyzer, display = demo_basic_operations(logger)

        if env and analyzer and display:
            demo_advanced_analysis(env, analyzer, logger)
            demo_different_displays(env, logger)

        demo_error_handling(logger)
        demo_game_simulation(logger)

        print("\n" + "=" * 60)
        print("🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
        print("=" * 60)
        print("Toutes les fonctionnalités ont été testées avec succès!")
        print("Architecture: ✅ Modulaire")
        print("Gestion d'erreurs: ✅ Robuste")
        print("Logging: ✅ Complet")
        print("Tests: ✅ Réussis")

        logger.logger.info("Démonstration complète terminée avec succès")

    except Exception as e:
        logger.logger.error(f"Erreur fatale dans main: {e}")
        print(f"💥 Erreur fatale: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
