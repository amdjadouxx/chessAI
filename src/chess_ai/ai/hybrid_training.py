"""
Entraînement hybride avec Stockfish + Neural Network
==================================================

Combine la force de Stockfish avec la vitesse du Neural Network
pour un entraînement plus efficace et des résultats immédiats.
"""

import os
import time
import chess
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .training import SelfPlayTrainer, TrainingConfig, GameData
from .network import encode_board
from .reference_evaluator import get_reference_evaluator


@dataclass
class HybridTrainingConfig(TrainingConfig):
    """Configuration pour l'entraînement hybride."""

    # Modes d'entraînement
    use_stockfish_guidance: bool = True
    stockfish_depth: int = 8
    hybrid_ratio: float = 0.7  # 70% Stockfish, 30% Neural Net

    # Adaptation dynamique
    neural_confidence_threshold: float = 0.8
    fallback_to_stockfish: bool = True

    # Performance
    stockfish_timeout: float = 2.0  # Max 2s par position
    parallel_evaluation: bool = True


class HybridSelfPlayTrainer(SelfPlayTrainer):
    """
    Entraîneur hybride qui combine Stockfish et Neural Network.

    Avantages :
    - Force immédiate grâce à Stockfish
    - Vitesse progressive grâce au Neural Network
    - Adaptation intelligente selon le contexte
    """

    def __init__(
        self,
        config: HybridTrainingConfig = None,
        device: str = "cpu",
        pretrained_model: str = None,
    ):
        """Initialise l'entraîneur hybride."""

        # Initialiser la base AlphaZero
        base_config = config or HybridTrainingConfig()
        super().__init__(base_config, device, pretrained_model)

        self.hybrid_config = base_config

        # Initialiser l'évaluateur de référence (Stockfish)
        self.reference_evaluator = get_reference_evaluator()
        print(f"🤖 Évaluateur hybride initialisé")

        # Statistiques
        self.stockfish_usage = 0
        self.neural_usage = 0
        self.hybrid_decisions = []

    def play_game_hybrid(self, verbose: bool = False) -> GameData:
        """
        Joue une partie avec l'IA hybride.

        Args:
            verbose: Afficher les détails

        Returns:
            Données de la partie
        """
        board = chess.Board()
        positions = []
        policies = []
        moves_played = []
        decision_log = []

        move_count = 0

        if verbose:
            print(f"\n🎮 Partie hybride (Stockfish + Neural Net)")

        while not board.is_game_over() and move_count < self.config.max_game_length:
            # Encoder la position
            position_tensor = encode_board(board)
            positions.append(position_tensor)

            # Obtenir la température
            temperature = self.get_temperature(move_count)

            # Décision hybride intelligente
            time_budget = self._calculate_time_budget(move_count, board)
            ai_mode = self._select_ai_mode(board, time_budget, move_count)

            # Obtenir le coup selon le mode choisi
            if ai_mode == "stockfish":
                move_distribution = self._get_stockfish_policy(board, time_budget)
                self.stockfish_usage += 1
                decision_type = "SF"
            elif ai_mode == "neural":
                move_distribution = self.mcts.run(board, self.config.mcts_simulations)
                self.neural_usage += 1
                decision_type = "NN"
            else:  # hybrid
                move_distribution = self._get_hybrid_policy(board, time_budget)
                decision_type = "HY"

            policies.append(move_distribution.copy())
            decision_log.append(decision_type)

            # Sélectionner et jouer le coup
            selected_move = self._select_move_from_policy(
                move_distribution, temperature
            )

            if verbose and move_count < 10:
                sorted_moves = sorted(
                    move_distribution.items(), key=lambda x: x[1], reverse=True
                )
                top_moves = sorted_moves[:3]
                print(
                    f"  {move_count+1:2d}. {selected_move} ({decision_type}) T={temperature:.1f}"
                )
                for i, (move, prob) in enumerate(top_moves):
                    print(f"      {i+1}. {move}: {prob:.3f}")

            board.push(selected_move)
            moves_played.append(selected_move.uci())
            move_count += 1

        # Résultat et valeurs
        result = board.result()
        values = self._calculate_values(result, len(positions))

        if verbose:
            decisions_summary = {
                "Stockfish": decision_log.count("SF"),
                "Neural": decision_log.count("NN"),
                "Hybrid": decision_log.count("HY"),
            }
            print(f"🏁 Partie terminée: {result} en {move_count} coups")
            print(f"📊 Décisions: {decisions_summary}")

        self.hybrid_decisions.extend(decision_log)

        return GameData(
            positions=positions,
            policies=policies,
            values=values,
            result=result,
            moves=moves_played,
            game_length=move_count,
        )

    def _calculate_time_budget(self, move_number: int, board: chess.Board) -> float:
        """Calcule le budget temps selon la phase de jeu."""

        # Plus de temps en début et fin de partie
        if move_number < 10:  # Ouverture
            return self.hybrid_config.stockfish_timeout * 1.5
        elif move_number > 40:  # Finale
            return self.hybrid_config.stockfish_timeout * 2.0
        else:  # Milieu
            return self.hybrid_config.stockfish_timeout

    def _select_ai_mode(
        self, board: chess.Board, time_budget: float, move_number: int
    ) -> str:
        """Sélectionne le mode IA optimal."""

        # Positions critiques → Stockfish
        if self._is_critical_position(board):
            return "stockfish"

        # Peu de temps → Neural Net
        if time_budget < 0.5:
            return "neural"

        # Phase d'ouverture → Stockfish (plus fiable)
        if move_number < 15:
            return "stockfish"

        # Évaluer la confiance du Neural Net
        neural_confidence = self._evaluate_neural_confidence(board)

        if neural_confidence > self.hybrid_config.neural_confidence_threshold:
            return "neural"  # Neural Net confiant
        elif neural_confidence > 0.5:
            return "hybrid"  # Combiner les deux
        else:
            return "stockfish"  # Fallback Stockfish

    def _is_critical_position(self, board: chess.Board) -> bool:
        """Détecte si la position est critique."""

        # Échec, mat en 1, gain/perte de matériel imminent
        if board.is_check():
            return True

        # Peu de matériel (finale)
        piece_count = len(board.piece_map())
        if piece_count <= 10:
            return True

        # Coups forcés (1 seul coup légal)
        legal_moves = list(board.legal_moves)
        if len(legal_moves) <= 2:
            return True

        return False

    def _evaluate_neural_confidence(self, board: chess.Board) -> float:
        """Évalue la confiance du Neural Network."""

        try:
            # Obtenir les prédictions du réseau
            position_tensor = encode_board(board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, value = self.network(position_tensor)

            # Calculer l'entropie de la policy (plus c'est bas, plus c'est confiant)
            probs = torch.softmax(policy_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))

            # Normaliser l'entropie (0 = très confiant, 1 = très incertain)
            max_entropy = np.log(len(list(board.legal_moves)))
            confidence = max(0, 1 - entropy.item() / max_entropy)

            return confidence

        except Exception:
            return 0.0  # Pas confiant en cas d'erreur

    def _get_stockfish_policy(
        self, board: chess.Board, time_budget: float
    ) -> Dict[chess.Move, float]:
        """Obtient la policy de Stockfish."""

        try:
            # Obtenir l'évaluation de base
            base_eval = self.reference_evaluator.evaluate_position(board)

            # Générer les policies pour tous les coups légaux
            legal_moves = list(board.legal_moves)
            move_scores = {}

            for move in legal_moves[:10]:  # Limiter aux 10 meilleurs pour la vitesse
                # Jouer le coup temporairement
                board.push(move)
                move_eval = self.reference_evaluator.evaluate_position(board)
                board.pop()

                # Score relatif (gain/perte par rapport à la position actuelle)
                if board.turn == chess.WHITE:
                    score_diff = move_eval - base_eval
                else:
                    score_diff = base_eval - move_eval

                move_scores[move] = max(
                    0, score_diff + 1.0
                )  # Éviter les scores négatifs

            # Normaliser en probabilités
            total_score = sum(move_scores.values())
            if total_score > 0:
                return {
                    move: score / total_score for move, score in move_scores.items()
                }
            else:
                # Fallback : distribution uniforme
                uniform_prob = 1.0 / len(legal_moves)
                return {move: uniform_prob for move in legal_moves}

        except Exception as e:
            print(f"⚠️ Erreur Stockfish policy: {e}")
            # Fallback : distribution uniforme
            legal_moves = list(board.legal_moves)
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 1.0
            return {move: uniform_prob for move in legal_moves}

    def _get_hybrid_policy(
        self, board: chess.Board, time_budget: float
    ) -> Dict[chess.Move, float]:
        """Combine Stockfish et Neural Network."""

        # Obtenir les deux policies
        stockfish_policy = self._get_stockfish_policy(board, time_budget * 0.7)
        neural_policy = self.mcts.run(board, int(self.config.mcts_simulations * 0.5))

        # Combiner selon le ratio configuré
        ratio = self.hybrid_config.hybrid_ratio
        combined_policy = {}

        all_moves = set(stockfish_policy.keys()) | set(neural_policy.keys())

        for move in all_moves:
            sf_prob = stockfish_policy.get(move, 0.0)
            nn_prob = neural_policy.get(move, 0.0)

            combined_policy[move] = ratio * sf_prob + (1 - ratio) * nn_prob

        # Normaliser
        total = sum(combined_policy.values())
        if total > 0:
            combined_policy = {
                move: prob / total for move, prob in combined_policy.items()
            }

        return combined_policy

    def _select_move_from_policy(
        self, policy: Dict[chess.Move, float], temperature: float
    ) -> chess.Move:
        """Sélectionne un coup selon la policy et la température."""

        if not policy:
            # Fallback : coup aléatoire
            return np.random.choice(
                list(policy.keys()) if policy else [chess.Move.null()]
            )

        moves = list(policy.keys())
        probs = list(policy.values())

        if temperature == 0.0:
            # Déterministe : meilleur coup
            best_idx = np.argmax(probs)
            return moves[best_idx]

        # Stochastique avec température
        if temperature != 1.0:
            probs = np.array(probs) ** (1.0 / temperature)
            probs = probs / np.sum(probs)

        return np.random.choice(moves, p=probs)

    def generate_hybrid_data(
        self, num_games: int, verbose: bool = True
    ) -> List[GameData]:
        """Génère des données par auto-jeu hybride."""

        games_data = []
        start_time = time.time()

        if verbose:
            print(f"\n🎯 Génération hybride de {num_games} parties...")

        for game_idx in range(num_games):
            game_data = self.play_game_hybrid(verbose=(game_idx < 3))
            games_data.append(game_data)

            if verbose and (game_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (game_idx + 1)
                eta = avg_time * (num_games - game_idx - 1)
                print(f"  📊 {game_idx + 1}/{num_games} parties (ETA: {eta:.1f}s)")

        # Statistiques finales
        if verbose:
            total_decisions = len(self.hybrid_decisions)
            sf_count = self.hybrid_decisions.count("SF")
            nn_count = self.hybrid_decisions.count("NN")
            hy_count = self.hybrid_decisions.count("HY")

            print(f"\n📈 Statistiques hybrides:")
            print(
                f"  🤖 Stockfish: {sf_count}/{total_decisions} ({sf_count/total_decisions*100:.1f}%)"
            )
            print(
                f"  🧠 Neural Net: {nn_count}/{total_decisions} ({nn_count/total_decisions*100:.1f}%)"
            )
            print(
                f"  ⚖️ Hybride: {hy_count}/{total_decisions} ({hy_count/total_decisions*100:.1f}%)"
            )

        return games_data

    def train_hybrid(self, num_iterations: int, verbose: bool = True):
        """Lance l'entraînement hybride."""

        if verbose:
            print(f"\n🚀 ENTRAÎNEMENT HYBRIDE - {num_iterations} itérations")
            print("=" * 60)

        for iteration in range(num_iterations):
            if verbose:
                print(f"\n🔄 Itération {iteration + 1}/{num_iterations}")
                print("-" * 40)

            # Générer des données hybrides
            games_data = self.generate_hybrid_data(
                self.config.games_per_iteration, verbose=verbose
            )

            # Entraîner le réseau sur ces données de qualité
            training_metrics = self.train_network(games_data, verbose=verbose)

            # Sauvegarder et loguer
            self.iteration += 1
            self.training_history.append(
                {
                    "iteration": self.iteration,
                    "games_played": len(games_data),
                    "training_loss": training_metrics.get("total_loss", 0),
                    "stockfish_usage": self.stockfish_usage,
                    "neural_usage": self.neural_usage,
                }
            )

            if iteration % self.config.save_interval == 0:
                self.save_model(f"hybrid_model_iter_{self.iteration:04d}.pt")

            if verbose:
                print(f"✅ Itération {iteration + 1} terminée")
                print(f"📊 Loss: {training_metrics.get('total_loss', 0):.4f}")


def create_hybrid_trainer(config: HybridTrainingConfig = None) -> HybridSelfPlayTrainer:
    """Crée un entraîneur hybride avec configuration par défaut."""

    if config is None:
        config = HybridTrainingConfig(
            mcts_simulations=200,  # Réduit car Stockfish compense
            games_per_iteration=20,
            epochs_per_iteration=5,
            use_stockfish_guidance=True,
            stockfish_depth=8,
            hybrid_ratio=0.7,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HybridSelfPlayTrainer(config, device)


if __name__ == "__main__":
    # Test de l'entraîneur hybride
    print("🧪 Test de l'entraîneur hybride")

    trainer = create_hybrid_trainer()

    # Jouer une partie test
    game_data = trainer.play_game_hybrid(verbose=True)
    print(f"\n✅ Partie test terminée: {game_data.result}")

    # Test d'entraînement court
    print(f"\n🎯 Test d'entraînement hybride (1 itération)")
    trainer.train_hybrid(1, verbose=True)

    print(f"\n🎉 Test terminé avec succès !")
