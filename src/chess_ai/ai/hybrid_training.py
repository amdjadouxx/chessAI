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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .training import SelfPlayTrainer, TrainingConfig, GameData
from .network import encode_board
from .reference_evaluator import get_reference_evaluator
from .stockfish_mcts import create_stockfish_mcts


@dataclass
class HybridGameData(GameData):
    """Données d'une partie hybride avec politiques Stockfish."""

    stockfish_policies: List[Dict[chess.Move, float]] = None  # Politiques Stockfish


@dataclass
class HybridTrainingConfig(TrainingConfig):
    """Configuration pour l'entraînement hybride."""

    # Modes d'entraînement
    use_stockfish_guidance: bool = True
    stockfish_depth: int = 16  # Plus profond pour plus de force

    # 🎲 VARIATION DE DEPTH pour éviter la convergence répétitive
    vary_stockfish_depth: bool = True  # Activer la variation
    depth_range: tuple = (8, 18)  # Min/Max depth (facile à difficile)
    depth_variation_mode: str = "adaptive"  # "random", "progressive", "adaptive"

    # Adaptation simplifiée
    adaptive_training: bool = True  # Adaptation de l'intensité d'entraînement
    neural_confidence_threshold: float = 0.8  # Pour l'auto-évaluation

    # Performance
    stockfish_timeout: float = 2.0  # Max 2s par position
    parallel_evaluation: bool = True

    # Évaluation précise pour l'entraînement
    use_stockfish_values: bool = True  # TOUJOURS actif pour l'entraînement
    stockfish_eval_depth: int = 10  # Profondeur pour évaluation


class HybridSelfPlayTrainer(SelfPlayTrainer):
    """
    🚀 Entraîneur hybride SIMPLIFIÉ - UN SEUL MODÈLE !

    Architecture claire :
    - UN SEUL réseau de neurones (self.network)
    - UN SEUL système de jeu (hybrid_mcts)
    - Stockfish utilisé SEULEMENT pour l'évaluation pendant MCTS et l'entraînement

    Plus de duplication de modèles, plus de confusion !
    Le Neural Network joue tous les coups et apprend de Stockfish.
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

        # 🚀 UN SEUL MODÈLE : MCTS Hybride avec Stockfish pour les évaluations !
        self.hybrid_mcts = create_stockfish_mcts(
            self.network,  # TOUJOURS le même réseau !
            self.reference_evaluator,
            c_puct=self.config.c_puct,
            device=str(self.device),
        )

        print(f"🤖 Évaluateur hybride initialisé")
        print(f"🎯 UN SEUL MODÈLE utilisé partout !")
        print(f"   • Politiques: Neural Network (rapidité)")
        print(f"   • Évaluations: Stockfish (précision)")

        # Statistiques
        self.stockfish_usage = 0
        self.neural_usage = 0
        self.hybrid_decisions = []

    def _get_adaptive_stockfish_depth(self, iteration: int, game_number: int) -> int:
        """
        🎲 Calcule la profondeur Stockfish adaptative pour varier les parties

        Args:
            iteration: Numéro d'itération d'entraînement
            game_number: Numéro de partie dans l'itération

        Returns:
            Profondeur Stockfish adaptée
        """
        if not self.hybrid_config.vary_stockfish_depth:
            return self.hybrid_config.stockfish_depth

        min_depth, max_depth = self.hybrid_config.depth_range
        mode = self.hybrid_config.depth_variation_mode

        if mode == "random":
            # 🎲 Variation aléatoire : créer de la diversité
            import random

            return random.randint(min_depth, max_depth)

        elif mode == "progressive":
            # 📈 Progression : commencer facile, devenir plus dur
            progress = iteration / max(20, iteration + 5)  # Progression 0->1
            depth = min_depth + int(progress * (max_depth - min_depth))
            return max(min_depth, min(max_depth, depth))

        elif mode == "adaptive":
            # 🧠 Adaptatif : varier selon la performance récente
            # Plus l'IA est forte, plus on augmente la difficulté
            if hasattr(self, "training_history") and self.training_history:
                recent_losses = [
                    h.get("training_loss", 5.0) for h in self.training_history[-5:]
                ]
                avg_loss = sum(recent_losses) / len(recent_losses)

                # Si la loss diminue (IA s'améliore), augmenter la difficulté
                if avg_loss < 2.0:  # IA forte
                    bias = 0.8  # Vers depths élevées
                elif avg_loss < 4.0:  # IA moyenne
                    bias = 0.5  # Depths moyennes
                else:  # IA faible
                    bias = 0.2  # Depths faibles

                # Ajouter variation aléatoire avec biais
                import random

                if random.random() < bias:
                    # Depth élevée
                    return random.randint((min_depth + max_depth) // 2, max_depth)
                else:
                    # Depth faible
                    return random.randint(min_depth, (min_depth + max_depth) // 2)
            else:
                # Pas d'historique : depth moyenne
                return (min_depth + max_depth) // 2

        return self.hybrid_config.stockfish_depth

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
        board_positions = []  # 🚀 NOUVEAU : Stocker les boards pour l'analyse Stockfish
        policies = []
        moves_played = []
        decision_log = []

        move_count = 0

        if verbose:
            print(f"\n🎮 Partie hybride (Stockfish + Neural Net)")

        while not board.is_game_over() and move_count < self.config.max_game_length:
            # Encoder la position pour le neural network
            position_tensor = encode_board(board)
            positions.append(position_tensor)

            # 🚀 NOUVEAU : Stocker aussi la position board pour l'analyse Stockfish
            board_positions.append(board.copy())

            # Obtenir la température
            temperature = self.get_temperature(move_count)

            # 🚀 UN SEUL SYSTÈME : MCTS Hybride (NN pour politique, Stockfish pour valeur)
            # Pas besoin de sélection de mode - toujours le même modèle hybride !
            move_distribution = self.hybrid_mcts.run(
                board, self.config.mcts_simulations
            )
            policies.append(move_distribution.copy())
            decision_log.append("HY")  # Hybride
            self.neural_usage += 1  # Compte comme neural usage

            # Sélectionner et jouer le coup
            selected_move = self._select_move_from_policy(
                move_distribution, temperature
            )

            if verbose and move_count < 10:
                sorted_moves = sorted(
                    move_distribution.items(), key=lambda x: x[1], reverse=True
                )
                top_moves = sorted_moves[:3]
                print(f"  {move_count+1:2d}. {selected_move} (HY) T={temperature:.1f}")
                for i, (move, prob) in enumerate(top_moves):
                    print(f"      {i+1}. {move}: {prob:.3f}")

            board.push(selected_move)
            moves_played.append(selected_move.uci())
            move_count += 1

        # Résultat et valeurs
        result = board.result()

        # 🚀 VRAIES VALEURS STOCKFISH pour un apprentissage précis !
        if self.hybrid_config.use_stockfish_values:
            values = self._calculate_stockfish_values(board_positions, result)
            if verbose:
                print(f"🎯 Valeurs Stockfish calculées pour {len(values)} positions")
        else:
            # Fallback: valeurs basiques (pour tests seulement)
            if result == "1-0":
                final_value = 1.0
            elif result == "0-1":
                final_value = -1.0
            else:
                final_value = 0.0

            values = []
            for i in range(len(positions)):
                if i % 2 == 0:  # Blancs
                    values.append(final_value)
                else:  # Noirs
                    values.append(-final_value)

        if verbose:
            # Statistiques du nouveau système hybride
            print(f"🏁 Partie terminée: {result} en {move_count} coups")
            print(f"🎯 MCTS Hybride: {move_count} coups")
            print(f"   • Politiques: Neural Network (guidage intelligent)")
            print(f"   • Évaluations: Stockfish (précision absolue)")
            print(f"🚀 Pas besoin de correction - évaluations déjà correctes !")

        self.hybrid_decisions.extend(decision_log)

        return HybridGameData(
            positions=positions,
            policies=policies,
            values=values,
            result=result,
            moves=moves_played,
            game_length=move_count,
            stockfish_policies=policies,  # 🚀 Les politiques sont déjà hybrides !
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
        """
        🚀 SUPPRIMÉ : Plus besoin de sélection !

        On utilise TOUJOURS le même système hybride :
        - Neural Network pour les politiques
        - Stockfish pour les évaluations
        - UN SEUL modèle partout !
        """
        # 🧠 TOUJOURS le même système hybride !
        return "hybrid"

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

    def _calculate_stockfish_values(
        self, positions: List[chess.Board], final_result: str
    ) -> List[float]:
        """
        🚀 NOUVEAU : Calcule les valeurs précises avec Stockfish !

        Au lieu d'utiliser seulement le résultat final, on évalue
        chaque position individuellement avec Stockfish.
        """
        values = []
        total_positions = len(positions)

        print(f"   📊 Évaluation de {total_positions} positions...")

        for i, board in enumerate(positions):
            try:
                # Évaluation Stockfish de la position
                stockfish_eval = self.reference_evaluator.evaluate_position(board)

                # Convertir en valeur [-1, 1]
                # Stockfish donne des centipawns → normaliser
                if abs(stockfish_eval) > 1000:  # Mat forcé
                    value = 1.0 if stockfish_eval > 0 else -1.0
                else:
                    # Fonction sigmoïde pour normaliser les centipawns
                    # 🚀 ÉCHELLE AJUSTÉE pour meilleure sensibilité
                    value = np.tanh(
                        stockfish_eval / 200.0
                    )  # 200cp = ~0.76 (plus sensible)

                # 🚀 CORRECTION : Stockfish évalue déjà du point de vue du joueur actuel !
                # Pas besoin d'inverser - Stockfish le fait automatiquement

                values.append(value)

                # Progress update
                if (i + 1) % 20 == 0:
                    print(f"      {i+1}/{total_positions} positions évaluées...")

            except Exception as e:
                print(f"⚠️ Erreur évaluation position {i}: {e}")
                # Fallback sur l'ancienne méthode
                if final_result == "1-0":
                    fallback_value = 1.0 if i % 2 == 0 else -1.0
                elif final_result == "0-1":
                    fallback_value = -1.0 if i % 2 == 0 else 1.0
                else:
                    fallback_value = 0.0
                values.append(fallback_value)

        print(f"   ✅ Évaluation Stockfish terminée !")
        print(f"   📈 Valeurs moyennes: {np.mean(values):.3f} (±{np.std(values):.3f})")

        return values

    def _calculate_stockfish_policies(
        self, positions: List[chess.Board]
    ) -> List[Dict[chess.Move, float]]:
        """
        🚀 NOUVEAU : Calcule les meilleures politiques avec Stockfish !

        Pour chaque position, demande à Stockfish son meilleur coup
        et crée une politique "professeur" pour l'entraînement.
        """
        stockfish_policies = []
        total_positions = len(positions)

        print(
            f"   🎯 Analyse des meilleurs coups Stockfish pour {total_positions} positions..."
        )

        for i, board in enumerate(positions):
            try:
                # Obtenir le meilleur coup de Stockfish
                best_move = self.reference_evaluator.get_best_move(board)

                if best_move and best_move in board.legal_moves:
                    # Créer une politique "concentrée" sur le meilleur coup Stockfish
                    stockfish_policy = {}
                    legal_moves = list(board.legal_moves)

                    for move in legal_moves:
                        if move == best_move:
                            # Le meilleur coup selon Stockfish = 80% de probabilité
                            stockfish_policy[move] = 0.8
                        else:
                            # Les autres coups se partagent le reste
                            stockfish_policy[move] = 0.2 / (len(legal_moves) - 1)

                    stockfish_policies.append(stockfish_policy)
                else:
                    # Fallback : distribution uniforme si erreur
                    legal_moves = list(board.legal_moves)
                    uniform_prob = 1.0 / len(legal_moves) if legal_moves else 1.0
                    stockfish_policies.append(
                        {move: uniform_prob for move in legal_moves}
                    )

                # Progress update
                if (i + 1) % 20 == 0:
                    print(f"      {i+1}/{total_positions} politiques analysées...")

            except Exception as e:
                print(f"⚠️ Erreur analyse coup position {i}: {e}")
                # Fallback : distribution uniforme
                legal_moves = list(board.legal_moves)
                uniform_prob = 1.0 / len(legal_moves) if legal_moves else 1.0
                stockfish_policies.append({move: uniform_prob for move in legal_moves})

        print(f"   ✅ Analyse des coups Stockfish terminée !")

        return stockfish_policies

    def train_hybrid_network(
        self, games_data: List[HybridGameData], verbose: bool = True
    ) -> Dict[str, float]:
        """
        🚀 NOUVEAU : Entraînement hybride avec politiques Stockfish !

        Le neural network apprend de deux sources :
        1. Évaluations Stockfish (comme avant)
        2. Meilleures politiques Stockfish (nouveau !)
        """
        if verbose:
            print(f"\n🧠 Entraînement hybride du réseau...")
            print(f"   📚 Apprentissage des évaluations ET des coups Stockfish")

        # Préparer les données hybrides
        positions, mcts_policies, stockfish_policies, target_values = (
            self._prepare_hybrid_training_data(games_data)
        )

        positions = positions.to(self.device)
        mcts_policies = mcts_policies.to(self.device)
        stockfish_policies = stockfish_policies.to(self.device)
        target_values = target_values.to(self.device).unsqueeze(1)

        if verbose:
            print(f"  📊 {len(positions)} exemples d'entraînement hybride")

        # DataLoader
        dataset = torch.utils.data.TensorDataset(
            positions, mcts_policies, stockfish_policies, target_values
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Entraînement
        self.network.train()
        total_loss = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        stockfish_policy_loss_total = 0.0
        num_batches = 0

        for epoch in range(self.config.epochs_per_iteration):
            epoch_loss = 0.0

            for (
                batch_positions,
                batch_mcts_policies,
                batch_stockfish_policies,
                batch_values,
            ) in dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                pred_policies, pred_values = self.network(batch_positions)

                # Loss des valeurs (comme avant)
                value_loss = torch.nn.MSELoss()(pred_values, batch_values)

                # 🚀 CORRECTION CRITIQUE : Utiliser KLDivLoss pour les probabilités
                # Loss des politiques MCTS (50% du poids)
                mcts_policy_loss = torch.nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(pred_policies, dim=1),
                    torch.softmax(batch_mcts_policies, dim=1),
                )

                # 🚀 NOUVEAU : Loss des politiques Stockfish (50% du poids)
                stockfish_policy_loss = torch.nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(pred_policies, dim=1),
                    torch.softmax(batch_stockfish_policies, dim=1),
                )

                # Loss total : apprentissage équilibré des deux sources
                policy_loss = 0.5 * mcts_policy_loss + 0.5 * stockfish_policy_loss
                total_batch_loss = policy_loss + value_loss

                # Backward pass
                total_batch_loss.backward()
                self.optimizer.step()

                epoch_loss += total_batch_loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                stockfish_policy_loss_total += stockfish_policy_loss.item()
                num_batches += 1

        # Métriques finales
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_policy_loss = policy_loss_total / num_batches if num_batches > 0 else 0
        avg_value_loss = value_loss_total / num_batches if num_batches > 0 else 0
        avg_stockfish_loss = (
            stockfish_policy_loss_total / num_batches if num_batches > 0 else 0
        )

        if verbose:
            print(f"  ✅ Entraînement terminé !")
            print(f"     💰 Loss valeurs: {avg_value_loss:.4f}")
            print(f"     🎯 Loss politiques MCTS: {avg_policy_loss:.4f}")
            print(f"     🤖 Loss politiques Stockfish: {avg_stockfish_loss:.4f}")
            print(f"     📈 Loss totale: {avg_total_loss:.4f}")

        return {
            "total_loss": avg_total_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "stockfish_policy_loss": avg_stockfish_loss,
        }

    def _prepare_hybrid_training_data(
        self, games_data: List[HybridGameData]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prépare les données pour l'entraînement hybride."""
        all_positions = []
        all_mcts_policies = []
        all_stockfish_policies = []
        all_values = []

        for game in games_data:
            for pos, mcts_policy, stockfish_policy, value in zip(
                game.positions, game.policies, game.stockfish_policies, game.values
            ):
                all_positions.append(pos)
                all_values.append(value)

                # Convertir les politiques en tenseurs
                from .network import move_to_index

                # Politique MCTS
                mcts_policy_vector = torch.zeros(4672)
                for move, prob in mcts_policy.items():
                    try:
                        idx = move_to_index(move)
                        if 0 <= idx < 4672:
                            mcts_policy_vector[idx] = prob
                    except:
                        continue
                all_mcts_policies.append(mcts_policy_vector)

                # 🚀 NOUVEAU : Politique Stockfish
                stockfish_policy_vector = torch.zeros(4672)
                for move, prob in stockfish_policy.items():
                    try:
                        idx = move_to_index(move)
                        if 0 <= idx < 4672:
                            stockfish_policy_vector[idx] = prob
                    except:
                        continue
                all_stockfish_policies.append(stockfish_policy_vector)

        # Convertir en tenseurs
        positions_tensor = torch.stack(all_positions)
        mcts_policies_tensor = torch.stack(all_mcts_policies)
        stockfish_policies_tensor = torch.stack(all_stockfish_policies)
        values_tensor = torch.tensor(all_values, dtype=torch.float32)

        return (
            positions_tensor,
            mcts_policies_tensor,
            stockfish_policies_tensor,
            values_tensor,
        )

    def _evaluate_model_performance(self, recent_games: List[GameData]) -> float:
        """
        🎯 Évalue la performance du modèle neural par rapport à Stockfish

        Retourne un score de 0 (mauvais) à 1 (excellent) basé sur :
        - Concordance des évaluations avec Stockfish
        - Qualité des politiques comparées aux meilleurs coups Stockfish
        """
        if not recent_games:
            return 0.5  # Performance neutre par défaut

        total_positions = 0
        value_agreement = 0
        policy_agreement = 0

        for game in recent_games[-5:]:  # Prendre les 5 dernières parties
            for i, position in enumerate(game.positions):
                if i >= len(game.policies) or i >= len(game.values):
                    continue

                try:
                    # Reconstruire la position du board
                    board = chess.Board()
                    for move_uci in game.moves[:i]:
                        board.push(chess.Move.from_uci(move_uci))

                    # Obtenir l'évaluation du réseau neural
                    position_tensor = position.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        policy_logits, value_pred = self.network(position_tensor)
                        neural_value = value_pred.item()

                    # Comparer avec l'évaluation Stockfish (stockée dans game.values)
                    stockfish_value = game.values[i]
                    value_diff = abs(neural_value - stockfish_value)

                    # Accord sur la valeur (plus proche = meilleur)
                    if value_diff < 0.2:  # Très bon accord
                        value_agreement += 1.0
                    elif value_diff < 0.5:  # Accord acceptable
                        value_agreement += 0.5

                    # Comparer les politiques : meilleur coup du réseau vs Stockfish
                    legal_moves = list(board.legal_moves)
                    if legal_moves and game.policies[i]:
                        # Meilleur coup selon le neural network (politique MCTS)
                        best_neural_move = max(
                            game.policies[i].items(), key=lambda x: x[1]
                        )[0]

                        # Meilleur coup selon Stockfish
                        try:
                            stockfish_best = self.reference_evaluator.get_best_move(
                                board
                            )
                            if best_neural_move == stockfish_best:
                                policy_agreement += 1.0
                        except:
                            continue  # Ignorer si erreur Stockfish

                    total_positions += 1

                except Exception as e:
                    continue  # Ignorer les erreurs de prédiction

        if total_positions == 0:
            return 0.5

        # Score combiné (50% valeur, 50% politique)
        value_score = value_agreement / total_positions if total_positions > 0 else 0
        policy_score = policy_agreement / total_positions if total_positions > 0 else 0
        overall_score = (value_score + policy_score) / 2

        print(
            f"📊 Performance du modèle: {overall_score:.3f} (valeur: {value_score:.3f}, politique: {policy_score:.3f})"
        )
        return overall_score

    def _adapt_training_parameters(self, performance_score: float):
        """
        🔄 Adaptation simplifiée : Intensité d'entraînement selon la performance

        Plus besoin d'adapter le ratio Stockfish/Neural puisque :
        - Parties : 100% Neural Network (fixe)
        - Entraînement : 80% Stockfish + 20% Neural (fixe)

        On adapte seulement l'intensité d'apprentissage.
        """
        if not self.hybrid_config.adaptive_training:
            return

        print(f" Performance: {performance_score:.3f}")

        # Adapter l'intensité d'entraînement selon la performance
        if performance_score > 0.8:
            # IA très forte → Entraînement minimal
            training_intensity = "minimal"
            print(f"   ✅ IA excellente → Entraînement de maintien")
        elif performance_score > 0.6:
            # IA correcte → Entraînement normal
            training_intensity = "normal"
            print(f"   📈 IA correcte → Entraînement standard")
        else:
            # IA faible → Entraînement intensif
            training_intensity = "intensif"
            print(f"   🔥 IA faible → Entraînement renforcé")

        # Plus besoin d'adapter les ratios, ils sont fixes maintenant
        print(f"   🎮 Jeu: 100% Neural Network (toujours)")
        print(f"   � Entraînement: 80% Stockfish + 20% Neural (toujours)")

    def train_single_game_hybrid(
        self, game_data: HybridGameData, verbose: bool = False
    ) -> Dict[str, float]:
        """
        🚀 NOUVEAU : Entraînement simplifié car MCTS hybride déjà optimal !

        Le MCTS hybride a utilisé Stockfish pour les évaluations,
        donc les politiques générées sont déjà de haute qualité.
        L'entraînement sert surtout à mémoriser ces bonnes décisions.

        Args:
            game_data: Données d'une partie hybride
            verbose: Afficher les détails

        Returns:
            Métriques d'entraînement
        """
        if verbose:
            print(
                f"🧠 Entraînement léger sur partie hybride ({game_data.game_length} coups)..."
            )
            print(f"   ✅ MCTS hybride a déjà utilisé Stockfish pendant la partie")
            print(f"   🎯 Objectif: mémoriser les bonnes décisions prises")

        # Préparer les données (politiques hybrides = déjà bonnes)
        positions, target_policies, target_values = self.prepare_training_data(
            [game_data]
        )

        positions = positions.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device).unsqueeze(1)

        if verbose:
            print(f"  📊 {len(positions)} exemples d'entraînement")

        # Entraînement très léger car pas besoin de corriger grand-chose
        epochs = 1  # Une seule époque car les données sont déjà bonnes
        batch_size = min(8, len(positions))

        # DataLoader
        dataset = torch.utils.data.TensorDataset(
            positions, target_policies, target_values
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Entraînement
        self.network.train()
        total_loss = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        num_batches = 0

        for batch_positions, batch_policies, batch_values in dataloader:
            self.optimizer.zero_grad()

            # Forward pass
            pred_policies, pred_values = self.network(batch_positions)

            # Loss des valeurs
            value_loss = torch.nn.MSELoss()(pred_values, batch_values)

            # Loss des politiques (hybrides = déjà optimales)
            targets = torch.argmax(batch_policies, dim=1)
            policy_loss = torch.nn.CrossEntropyLoss()(pred_policies, targets)

            # Loss total simple
            total_batch_loss = policy_loss + value_loss

            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            num_batches += 1

        # Métriques finales
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_policy_loss = policy_loss_total / num_batches if num_batches > 0 else 0
        avg_value_loss = value_loss_total / num_batches if num_batches > 0 else 0

        if verbose:
            print(f"  ✅ Entraînement léger terminé:")
            print(f"     💰 Loss valeurs: {avg_value_loss:.4f}")
            print(f"     🎯 Loss politiques: {avg_policy_loss:.4f}")
            print(f"     📈 Loss totale: {avg_total_loss:.4f}")
            print(f"     🧠 UN SEUL modèle utilisé partout - pas de duplication !")

        # 🚀 PLUS BESOIN de recréer les MCTS - ils utilisent déjà self.network !
        # Le réseau a été mis à jour "en place", donc tous les MCTS voient les changements

        return {
            "total_loss": avg_total_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
        }

    def play_game(self, verbose: bool = False) -> GameData:
        """
        🚀 Override : TOUJOURS utiliser Stockfish pour l'évaluation !
        """
        # Utiliser la méthode hybride qui intègre Stockfish
        return self.play_game_hybrid(verbose=verbose)

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
            # 🎲 Utiliser profondeur Stockfish adaptative pour chaque partie
            current_depth = self._get_adaptive_stockfish_depth(
                iteration=getattr(self, "iteration", 0), game_number=game_idx
            )

            # Temporairement changer la profondeur pour cette partie
            original_depth = self.hybrid_config.stockfish_depth
            self.hybrid_config.stockfish_depth = current_depth

            if verbose and game_idx < 3:
                print(f"  🎯 Partie {game_idx + 1}: Stockfish depth {current_depth}")

            game_data = self.play_game_hybrid(verbose=(game_idx < 3))
            games_data.append(game_data)

            # Restaurer la profondeur originale
            self.hybrid_config.stockfish_depth = original_depth

            if verbose and (game_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (game_idx + 1)
                eta = avg_time * (num_games - game_idx - 1)
                print(f"  📊 {game_idx + 1}/{num_games} parties (ETA: {eta:.1f}s)")

        # Statistiques finales
        if verbose:
            print(f"\n📈 Statistiques d'entraînement:")
            print(f"  🧠 Neural Network: 100% des coups pendant les parties")
            print(f"  🎓 Stockfish: Professeur pour l'entraînement après coup")
            if self.hybrid_config.vary_stockfish_depth:
                depths = [
                    self._get_adaptive_stockfish_depth(getattr(self, "iteration", 0), i)
                    for i in range(num_games)
                ]
                min_d, max_d = min(depths), max(depths)
                print(
                    f"  🎲 Variation Stockfish: depth {min_d}-{max_d} ({self.hybrid_config.depth_variation_mode})"
                )
            print(f"  📊 Apprentissage continu activé")

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

            # 🧠 Entraîner le réseau avec les politiques hybrides !
            training_metrics = self.train_hybrid_network(games_data, verbose=verbose)

            # Sauvegarder et loguer
            self.iteration += 1
            self.training_history.append(
                {
                    "iteration": self.iteration,
                    "games_played": len(games_data),
                    "training_loss": training_metrics.get("total_loss", 0),
                    "stockfish_usage": self.stockfish_usage,
                    "neural_usage": self.neural_usage,
                    "stockfish_policy_loss": training_metrics.get(
                        "stockfish_policy_loss", 0
                    ),
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
            games_per_iteration=5,  # Réduit pour tests plus rapides
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
