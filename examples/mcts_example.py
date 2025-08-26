"""
Exemple d'utilisation de MCTS avec AlphaZero pour les échecs
===========================================================

Ce script démontre comment utiliser le module MCTS avec le réseau de neurones
pour faire jouer l'IA contre elle-même ou contre un joueur humain.
"""

import chess
import torch
import time
import sys
import os

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from chess_ai.ai import ChessNet, MCTS


class AlphaZeroMCTSPlayer:
    """
    Joueur d'échecs utilisant MCTS + réseau de neurones AlphaZero.
    """

    def __init__(
        self,
        model_path: str = None,
        num_simulations: int = 800,
        c_puct: float = 1.4,
        temperature: float = 1.0,
    ):
        """
        Initialise le joueur AlphaZero.

        Args:
            model_path: Chemin vers le modèle entraîné (None = modèle aléatoire)
            num_simulations: Nombre de simulations MCTS par coup
            c_puct: Constante d'exploration PUCT
            temperature: Température pour la sélection de coups
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_simulations = num_simulations
        self.temperature = temperature

        # Charger ou créer le réseau
        self.network = ChessNet().to(self.device)
        if model_path:
            try:
                self.network.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                print(f"Modèle chargé depuis : {model_path}")
            except Exception as e:
                print(f"Erreur lors du chargement : {e}")
                print("Utilisation d'un modèle aléatoire")
        else:
            print("Utilisation d'un modèle aléatoire (non entraîné)")

        self.network.eval()

        # Créer l'algorithme MCTS
        self.mcts = MCTS(self.network, c_puct=c_puct, device=self.device)

    def get_move(self, board: chess.Board, verbose: bool = True) -> chess.Move:
        """
        Sélectionne le meilleur coup pour la position donnée.

        Args:
            board: Position d'échecs actuelle
            verbose: Afficher les informations de debug

        Returns:
            Coup sélectionné
        """
        if verbose:
            print(f"\nRéflexion pour {'Blancs' if board.turn else 'Noirs'}...")
            print(f"Simulations MCTS : {self.num_simulations}")

        start_time = time.time()

        # Effectuer les simulations MCTS
        move_distribution = self.mcts.run(board, self.num_simulations)

        # Sélectionner le coup
        selected_move = self.mcts.select_move(move_distribution, self.temperature)

        elapsed = time.time() - start_time

        if verbose:
            print(f"Temps de réflexion : {elapsed:.2f}s")
            print(f"Coup choisi : {selected_move}")

            # Afficher le top 5 des coups considérés
            print("\nTop 5 des coups analysés :")
            sorted_moves = sorted(
                move_distribution.items(), key=lambda x: x[1], reverse=True
            )
            for i, (move, prob) in enumerate(sorted_moves[:5]):
                print(f"  {i+1}. {move} : {prob:.4f}")

            # Statistiques MCTS
            stats = self.mcts.get_action_stats()
            if "total_visits" in stats:
                print(f"\nVisites totales : {stats['total_visits']}")
                print(f"Q-value : {stats['q_value']:.4f}")

        return selected_move

    def reset(self):
        """Réinitialise l'arbre MCTS pour une nouvelle partie."""
        self.mcts.reset()


def play_game(
    player1: AlphaZeroMCTSPlayer,
    player2: AlphaZeroMCTSPlayer = None,
    max_moves: int = 100,
    verbose: bool = True,
):
    """
    Fait jouer une partie entre deux joueurs IA ou IA vs humain.

    Args:
        player1: Joueur des blancs
        player2: Joueur des noirs (None = humain)
        max_moves: Nombre maximum de coups
        verbose: Afficher les détails de la partie

    Returns:
        Résultat de la partie
    """
    board = chess.Board()
    move_count = 0

    # Réinitialiser les joueurs
    player1.reset()
    if player2:
        player2.reset()

    if verbose:
        print("=" * 50)
        print("NOUVELLE PARTIE D'ÉCHECS")
        print("=" * 50)
        print(f"Blancs : {'IA MCTS' if player1 else 'Humain'}")
        print(f"Noirs : {'IA MCTS' if player2 else 'Humain'}")
        print()
        print(board)
        print()

    while not board.is_game_over() and move_count < max_moves:
        current_player = player1 if board.turn == chess.WHITE else player2

        if current_player:  # IA
            move = current_player.get_move(board, verbose)
        else:  # Joueur humain
            print(f"\nCoups légaux : {[str(m) for m in board.legal_moves]}")
            while True:
                try:
                    move_str = input(
                        f"Votre coup ({'Blancs' if board.turn else 'Noirs'}) : "
                    )
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        break
                    else:
                        print("Coup illégal, essayez à nouveau.")
                except:
                    print("Format incorrect. Utilisez la notation UCI (ex: e2e4)")

        # Jouer le coup
        board.push(move)
        move_count += 1

        if verbose:
            print(f"\nCoup {move_count}: {move}")
            print(board)
            print()

    # Résultat de la partie
    result = board.result()
    if verbose:
        print("=" * 50)
        print("FIN DE PARTIE")
        print("=" * 50)
        print(f"Résultat : {result}")
        if result == "1-0":
            print("Victoire des Blancs")
        elif result == "0-1":
            print("Victoire des Noirs")
        else:
            print("Match nul")
        print(f"Nombre de coups : {move_count}")

    return result


def benchmark_mcts(num_positions: int = 10, simulations_list: list = [100, 400, 800]):
    """
    Benchmark des performances MCTS avec différents nombres de simulations.

    Args:
        num_positions: Nombre de positions à tester
        simulations_list: Liste des nombres de simulations à tester
    """
    print("=" * 60)
    print("BENCHMARK MCTS")
    print("=" * 60)

    # Créer un réseau de test
    network = ChessNet()

    results = {}

    for num_sims in simulations_list:
        print(f"\nTest avec {num_sims} simulations...")
        mcts = MCTS(network, c_puct=1.4)

        total_time = 0
        positions_tested = 0

        for i in range(num_positions):
            # Créer une position aléatoire
            board = chess.Board()
            # Jouer quelques coups aléatoires
            for _ in range(min(10, len(list(board.legal_moves)))):
                if board.is_game_over():
                    break
                moves = list(board.legal_moves)
                if moves:
                    board.push(moves[0])  # Premier coup légal

            if board.is_game_over():
                continue

            # Mesurer le temps MCTS
            start_time = time.time()
            move_dist = mcts.run(board, num_sims)
            elapsed = time.time() - start_time

            total_time += elapsed
            positions_tested += 1
            mcts.reset()

        if positions_tested > 0:
            avg_time = total_time / positions_tested
            results[num_sims] = avg_time
            print(f"  Temps moyen : {avg_time:.3f}s par position")
            print(f"  Positions testées : {positions_tested}")

    print("\n" + "=" * 60)
    print("RÉSUMÉ DU BENCHMARK")
    print("=" * 60)
    for num_sims, avg_time in results.items():
        print(f"{num_sims:4d} simulations : {avg_time:.3f}s/position")


if __name__ == "__main__":
    print("Exemple d'utilisation de MCTS AlphaZero pour les échecs")
    print("=" * 60)

    # Test simple avec un réseau non entraîné
    print("\n1. Test d'une position simple...")

    # Créer un joueur IA
    ai_player = AlphaZeroMCTSPlayer(
        model_path=None,  # Pas de modèle pré-entraîné
        num_simulations=200,
        c_puct=1.4,
        temperature=1.0,
    )

    # Position de test
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    print(f"Position FEN : {board.fen()}")
    print(board)

    # Obtenir un coup de l'IA
    move = ai_player.get_move(board, verbose=True)
    print(f"\nL'IA joue : {move}")

    # Test d'une partie IA vs IA
    print("\n" + "=" * 60)
    print("2. Partie courte IA vs IA...")

    player_white = AlphaZeroMCTSPlayer(num_simulations=100, temperature=0.5)
    player_black = AlphaZeroMCTSPlayer(num_simulations=100, temperature=0.5)

    result = play_game(player_white, player_black, max_moves=20, verbose=True)

    # Benchmark optionnel
    print("\n" + "=" * 60)
    print("3. Benchmark des performances (optionnel)")
    response = input("Lancer le benchmark ? (y/N): ")
    if response.lower() == "y":
        benchmark_mcts(num_positions=5, simulations_list=[50, 100, 200])

    print("\nExemple terminé !")
