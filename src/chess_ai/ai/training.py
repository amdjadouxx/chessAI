"""
Module d'entraînement par auto-jeu pour AlphaZero
================================================

Implémente l'entraînement continu des modèles par parties IA vs IA
avec sauvegarde des données et amélioration des réseaux.
"""

import os
import time
import json
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle

from .network import ChessNet, encode_board, decode_policy
from .mcts import MCTS


@dataclass
class GameData:
    """Données d'une partie pour l'entraînement."""

    positions: List[torch.Tensor]  # Positions encodées
    policies: List[Dict[chess.Move, float]]  # Distributions MCTS
    values: List[float]  # Valeurs finales
    result: str  # Résultat de la partie ("1-0", "0-1", "1/2-1/2")
    moves: List[str]  # Coups joués en notation UCI
    game_length: int  # Nombre de coups


@dataclass
class TrainingConfig:
    """Configuration pour l'entraînement."""

    # MCTS
    mcts_simulations: int = 400
    c_puct: float = 1.4
    temperature_schedule: Dict[int, float] = None  # {move_number: temperature}

    # Entraînement
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 10

    # Auto-jeu
    games_per_iteration: int = 100
    max_game_length: int = 200
    save_interval: int = 10  # Sauvegarder tous les N itérations

    # Chemins
    models_dir: str = "models"
    data_dir: str = "training_data"
    logs_dir: str = "logs"

    def __post_init__(self):
        if self.temperature_schedule is None:
            # Température décroissante au cours de la partie
            self.temperature_schedule = {
                0: 1.0,  # Début créatif
                10: 0.8,  # Milieu plus stable
                20: 0.5,  # Fin plus déterministe
                30: 0.1,
            }


class SelfPlayTrainer:
    """
    Entraîneur par auto-jeu pour AlphaZero.

    Cycle d'entraînement :
    1. Génération de parties par auto-jeu
    2. Entraînement du réseau sur les données
    3. Évaluation et mise à jour du modèle
    """

    def __init__(
        self,
        config: TrainingConfig = None,
        device: str = "cpu",
        pretrained_model: str = None,
    ):
        """
        Initialise l'entraîneur.

        Args:
            config: Configuration d'entraînement
            device: Device de calcul ("cpu" ou "cuda")
            pretrained_model: Chemin vers un modèle pré-entraîné (optionnel)
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(device)

        # Créer les dossiers nécessaires
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)

        # Initialiser le réseau
        self.network = ChessNet().to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Charger un modèle pré-entraîné si spécifié
        if pretrained_model and os.path.exists(pretrained_model):
            self.load_pretrained(pretrained_model)

        # Initialiser MCTS
        self.mcts = MCTS(
            self.network, c_puct=self.config.c_puct, device=str(self.device)
        )

        # Historique d'entraînement
        self.iteration = 0
        self.training_history = []
        self.game_database = []

        print(f"🤖 SelfPlayTrainer initialisé sur {self.device}")
        print(f"📁 Modèles: {self.config.models_dir}")
        print(f"📊 Données: {self.config.data_dir}")

    def get_temperature(self, move_number: int) -> float:
        """
        Obtient la température selon le planning.

        Args:
            move_number: Numéro du coup

        Returns:
            Température à utiliser
        """
        schedule = self.config.temperature_schedule

        # Trouver la température appropriée
        for threshold in sorted(schedule.keys(), reverse=True):
            if move_number >= threshold:
                return schedule[threshold]

        return 1.0  # Défaut

    def play_game(self, verbose: bool = False) -> GameData:
        """
        Joue une partie complète en auto-jeu.

        Args:
            verbose: Afficher les détails

        Returns:
            Données de la partie
        """
        board = chess.Board()
        positions = []
        policies = []
        moves_played = []

        move_count = 0

        if verbose:
            print(
                f"\n🎮 Nouvelle partie auto-jeu (MCTS: {self.config.mcts_simulations} sim)"
            )

        # Réinitialiser MCTS
        self.mcts.reset()

        while not board.is_game_over() and move_count < self.config.max_game_length:
            # Encoder la position
            position_tensor = encode_board(board)
            positions.append(position_tensor)

            # Obtenir la température pour ce coup
            temperature = self.get_temperature(move_count)

            # Recherche MCTS
            move_distribution = self.mcts.run(board, self.config.mcts_simulations)
            policies.append(move_distribution.copy())

            # Sélectionner le coup
            selected_move = self.mcts.select_move(move_distribution, temperature)

            if verbose and move_count < 10:  # Afficher les premiers coups
                sorted_moves = sorted(
                    move_distribution.items(), key=lambda x: x[1], reverse=True
                )
                top_move = sorted_moves[0] if sorted_moves else (selected_move, 0.0)
                print(
                    f"  {move_count+1:2d}. {selected_move} (T={temperature:.1f}, P={top_move[1]:.3f})"
                )

            # Jouer le coup
            board.push(selected_move)
            moves_played.append(selected_move.uci())
            move_count += 1

        # Résultat de la partie
        result = board.result()

        # Calculer les valeurs finales (rewards)
        values = self._calculate_values(result, len(positions))

        if verbose:
            print(f"🏁 Partie terminée: {result} en {move_count} coups")

        return GameData(
            positions=positions,
            policies=policies,
            values=values,
            result=result,
            moves=moves_played,
            game_length=move_count,
        )

    def _calculate_values(self, result: str, num_positions: int) -> List[float]:
        """
        Calcule les valeurs finales pour chaque position.

        Args:
            result: Résultat de la partie
            num_positions: Nombre de positions

        Returns:
            Liste des valeurs pour chaque position
        """
        # Valeur selon le résultat
        if result == "1-0":  # Blancs gagnent
            final_value = 1.0
        elif result == "0-1":  # Noirs gagnent
            final_value = -1.0
        else:  # Match nul
            final_value = 0.0

        values = []
        for i in range(num_positions):
            # Alterner la valeur selon le joueur (blancs = pair, noirs = impair)
            if i % 2 == 0:  # Position des blancs
                values.append(final_value)
            else:  # Position des noirs
                values.append(-final_value)

        return values

    def generate_self_play_data(
        self, num_games: int, verbose: bool = True
    ) -> List[GameData]:
        """
        Génère des données par auto-jeu.

        Args:
            num_games: Nombre de parties à jouer
            verbose: Afficher les progrès

        Returns:
            Liste des données de parties
        """
        games_data = []
        start_time = time.time()

        if verbose:
            print(f"\n🎯 Génération de {num_games} parties par auto-jeu...")

        for game_idx in range(num_games):
            game_data = self.play_game(
                verbose=(game_idx < 3)
            )  # Détails pour les 3 premières
            games_data.append(game_data)

            if verbose and (game_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (game_idx + 1)
                eta = avg_time * (num_games - game_idx - 1)
                print(f"  📊 {game_idx + 1}/{num_games} parties (ETA: {eta:.1f}s)")

        elapsed = time.time() - start_time

        if verbose:
            # Statistiques
            results = [game.result for game in games_data]
            white_wins = results.count("1-0")
            black_wins = results.count("0-1")
            draws = results.count("1/2-1/2")
            avg_length = np.mean([game.game_length for game in games_data])

            print(f"\n📈 Statistiques auto-jeu:")
            print(f"  ⏱️  Temps total: {elapsed:.1f}s ({elapsed/num_games:.2f}s/partie)")
            print(
                f"  🏆 Victoires blancs: {white_wins} ({white_wins/num_games*100:.1f}%)"
            )
            print(
                f"  🏆 Victoires noirs: {black_wins} ({black_wins/num_games*100:.1f}%)"
            )
            print(f"  🤝 Matchs nuls: {draws} ({draws/num_games*100:.1f}%)")
            print(f"  📏 Longueur moyenne: {avg_length:.1f} coups")

        return games_data

    def prepare_training_data(
        self, games_data: List[GameData]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prépare les données pour l'entraînement.

        Args:
            games_data: Données des parties

        Returns:
            Tuple (positions, target_policies, target_values)
        """
        all_positions = []
        all_policies = []
        all_values = []

        for game in games_data:
            for pos, policy, value in zip(game.positions, game.policies, game.values):
                all_positions.append(pos)
                all_values.append(value)

                # Convertir la politique en tenseur
                # Créer un vecteur de probabilités pour tous les coups possibles
                policy_vector = torch.zeros(4672)  # Taille de l'espace d'action

                for move, prob in policy.items():
                    try:
                        from .network import move_to_index

                        idx = move_to_index(move)
                        if 0 <= idx < 4672:
                            policy_vector[idx] = prob
                    except:
                        continue  # Ignorer les coups non valides

                all_policies.append(policy_vector)

        # Convertir en tenseurs
        positions_tensor = torch.stack(all_positions)
        policies_tensor = torch.stack(all_policies)
        values_tensor = torch.tensor(all_values, dtype=torch.float32)

        return positions_tensor, policies_tensor, values_tensor

    def train_network(
        self, games_data: List[GameData], verbose: bool = True
    ) -> Dict[str, float]:
        """
        Entraîne le réseau sur les données d'auto-jeu.

        Args:
            games_data: Données des parties
            verbose: Afficher les progrès

        Returns:
            Métriques d'entraînement
        """
        if verbose:
            print(f"\n🧠 Entraînement du réseau...")

        # Préparer les données
        positions, target_policies, target_values = self.prepare_training_data(
            games_data
        )
        positions = positions.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device).unsqueeze(1)

        if verbose:
            print(f"  📊 {len(positions)} exemples d'entraînement")

        # DataLoader
        dataset = torch.utils.data.TensorDataset(
            positions, target_policies, target_values
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Entraînement
        self.network.train()
        total_loss = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        num_batches = 0

        for epoch in range(self.config.epochs_per_iteration):
            epoch_loss = 0.0

            for batch_positions, batch_policies, batch_values in dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                pred_policies, pred_values = self.network(batch_positions)

                # Loss
                policy_loss = nn.CrossEntropyLoss()(pred_policies, batch_policies)
                value_loss = nn.MSELoss()(pred_values, batch_values)
                total_batch_loss = policy_loss + value_loss

                # Backward pass
                total_batch_loss.backward()
                self.optimizer.step()

                epoch_loss += total_batch_loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                num_batches += 1

            if verbose and epoch % 2 == 0:
                print(
                    f"    Époque {epoch+1}/{self.config.epochs_per_iteration}: Loss = {epoch_loss/len(dataloader):.4f}"
                )

        avg_total_loss = (policy_loss_total + value_loss_total) / num_batches
        avg_policy_loss = policy_loss_total / num_batches
        avg_value_loss = value_loss_total / num_batches

        if verbose:
            print(f"  ✅ Entraînement terminé:")
            print(f"    Loss totale: {avg_total_loss:.4f}")
            print(f"    Loss politique: {avg_policy_loss:.4f}")
            print(f"    Loss valeur: {avg_value_loss:.4f}")

        return {
            "total_loss": avg_total_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
        }

    def save_model(self, filename: Optional[str] = None):
        """Sauvegarde le modèle."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alphazero_iter_{self.iteration:04d}_{timestamp}.pt"

        filepath = os.path.join(self.config.models_dir, filename)

        checkpoint = {
            "iteration": self.iteration,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_history": self.training_history,
        }

        torch.save(checkpoint, filepath)
        print(f"💾 Modèle sauvegardé: {filepath}")

        return filepath

    def load_model(self, filepath: str):
        """Charge un modèle."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint.get("iteration", 0)
        self.training_history = checkpoint.get("training_history", [])

        print(f"📥 Modèle chargé: {filepath} (itération {self.iteration})")

    def load_pretrained(self, filepath: str):
        """
        Charge un modèle pré-entraîné supervisé.

        Args:
            filepath: Chemin vers le modèle pré-entraîné
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # Charger les poids du réseau
            self.network.load_state_dict(checkpoint["model_state_dict"])

            # Vérifier si c'est un modèle pré-entraîné supervisé
            is_supervised = checkpoint.get("supervised_pretrained", False)

            if is_supervised:
                print(f"🧠 Modèle pré-entraîné chargé: {filepath}")
                print("✅ Le réseau a été pré-entraîné avec Stockfish")
                print("🚀 L'entraînement AlphaZero sera plus efficace")
            else:
                print(f"📥 Modèle standard chargé: {filepath}")

            # Réinitialiser l'optimiseur pour l'auto-jeu
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        except Exception as e:
            print(f"⚠️  Erreur lors du chargement du modèle pré-entraîné: {e}")
            print("🔄 Utilisation d'un modèle vierge")

    def save_training_data(self, games_data: List[GameData]):
        """Sauvegarde les données d'entraînement."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_iter_{self.iteration:04d}_{timestamp}.pkl"
        filepath = os.path.join(self.config.data_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(games_data, f)

        print(f"💾 Données sauvegardées: {filepath}")

    def training_iteration(self, verbose: bool = True) -> Dict[str, any]:
        """
        Effectue une itération complète d'entraînement.

        Returns:
            Résultats de l'itération
        """
        iteration_start = time.time()

        if verbose:
            print(f"\n" + "=" * 60)
            print(f"🚀 ITÉRATION D'ENTRAÎNEMENT #{self.iteration + 1}")
            print(f"=" * 60)

        # 1. Génération des données par auto-jeu
        games_data = self.generate_self_play_data(
            self.config.games_per_iteration, verbose=verbose
        )

        # 2. Entraînement du réseau
        training_metrics = self.train_network(games_data, verbose=verbose)

        # 3. Mise à jour de l'historique
        self.iteration += 1
        iteration_data = {
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "games_played": len(games_data),
            "training_metrics": training_metrics,
            "avg_game_length": np.mean([g.game_length for g in games_data]),
            "results_distribution": {
                "white_wins": sum(1 for g in games_data if g.result == "1-0"),
                "black_wins": sum(1 for g in games_data if g.result == "0-1"),
                "draws": sum(1 for g in games_data if g.result == "1/2-1/2"),
            },
        }

        self.training_history.append(iteration_data)

        # 4. Sauvegardes
        if self.iteration % self.config.save_interval == 0:
            self.save_model()
            self.save_training_data(games_data)

        # Mettre à jour MCTS avec le nouveau modèle
        self.mcts = MCTS(
            self.network, c_puct=self.config.c_puct, device=str(self.device)
        )

        elapsed = time.time() - iteration_start

        if verbose:
            print(f"\n⏱️  Itération {self.iteration} terminée en {elapsed:.1f}s")
            print(f"📊 Loss totale: {training_metrics['total_loss']:.4f}")

        return iteration_data

    def train(self, num_iterations: int, verbose: bool = True):
        """
        Lance l'entraînement pour plusieurs itérations.

        Args:
            num_iterations: Nombre d'itérations d'entraînement
            verbose: Afficher les détails
        """
        print(f"\n🎯 DÉBUT D'ENTRAÎNEMENT ALPHAZERO")
        print(f"Itérations prévues: {num_iterations}")
        print(f"Parties par itération: {self.config.games_per_iteration}")
        print(f"Simulations MCTS: {self.config.mcts_simulations}")
        print(f"Device: {self.device}")

        training_start = time.time()

        try:
            for i in range(num_iterations):
                iteration_data = self.training_iteration(verbose=verbose)

                if verbose:
                    remaining = num_iterations - (i + 1)
                    if remaining > 0:
                        elapsed = time.time() - training_start
                        eta = elapsed / (i + 1) * remaining
                        print(
                            f"📅 Itérations restantes: {remaining} (ETA: {eta/60:.1f}min)"
                        )

        except KeyboardInterrupt:
            print(f"\n⚠️  Entraînement interrompu par l'utilisateur")

        finally:
            # Sauvegarde finale
            final_model_path = self.save_model("alphazero_final.pt")

            total_elapsed = time.time() - training_start
            print(f"\n🏁 ENTRAÎNEMENT TERMINÉ")
            print(f"⏱️  Temps total: {total_elapsed/60:.1f} minutes")
            print(f"🔄 Itérations complétées: {self.iteration}")
            print(f"💾 Modèle final: {final_model_path}")


# Fonctions utilitaires pour l'interface
def quick_training_session(
    num_iterations: int = 5, num_games: int = 50, device: str = "cpu"
):
    """
    Session d'entraînement rapide pour tests.

    Args:
        num_iterations: Nombre d'itérations
        num_games: Parties par itération
        device: Device de calcul
    """
    config = TrainingConfig(
        games_per_iteration=num_games,
        mcts_simulations=200,  # Réduit pour vitesse
        epochs_per_iteration=5,
        batch_size=16,
    )

    trainer = SelfPlayTrainer(config, device=device)
    trainer.train(num_iterations)

    return trainer


def continue_training(model_path: str, num_iterations: int = 10, device: str = "cpu"):
    """
    Continue l'entraînement depuis un modèle existant.

    Args:
        model_path: Chemin vers le modèle à charger
        num_iterations: Itérations supplémentaires
        device: Device de calcul
    """
    trainer = SelfPlayTrainer(device=device)
    trainer.load_model(model_path)
    trainer.train(num_iterations)

    return trainer


if __name__ == "__main__":
    # Test rapide
    print("🧪 Test du module d'entraînement")

    # Configuration de test
    config = TrainingConfig(
        games_per_iteration=5, mcts_simulations=50, epochs_per_iteration=2, batch_size=4
    )

    # Entraîneur de test
    trainer = SelfPlayTrainer(config, device="cpu")

    # Une itération de test
    result = trainer.training_iteration(verbose=True)

    print(f"\n✅ Test réussi !")
    print(f"Résultat: {result}")
