"""
Monte Carlo Tree Search (MCTS) pour AlphaZero appliqué aux échecs
================================================================

Implémente l'algorithme MCTS avec les 4 étapes principales :
1. Sélection : navigation dans l'arbre avec PUCT
2. Expansion : ajout de nouveaux nœuds
3. Évaluation : prédiction par le réseau de neurones
4. Backpropagation : mise à jour des statistiques

Classes principales :
- MCTSNode : Nœud de l'arbre MCTS
- MCTS : Algorithme principal
"""

import math
import numpy as np
import chess
import torch
from typing import Dict, List, Optional, Tuple, Any
from .network import encode_board, decode_policy, ChessNet


class MCTSNode:
    """
    Nœud de l'arbre MCTS.

    Stocke les statistiques nécessaires pour l'algorithme PUCT :
    - N(s,a) : nombre de visites
    - W(s,a) : somme des valeurs
    - Q(s,a) : valeur moyenne (W/N)
    - P(s,a) : probabilité initiale du réseau
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional["MCTSNode"] = None,
        move: Optional[chess.Move] = None,
        prior_prob: float = 0.0,
    ):
        """
        Initialise un nœud MCTS.

        Args:
            board: Position d'échecs de ce nœud
            parent: Nœud parent (None pour la racine)
            move: Coup qui a mené à ce nœud
            prior_prob: Probabilité a priori donnée par le réseau
        """
        self.board = board.copy()  # Position d'échecs
        self.parent = parent
        self.move = move
        self.prior_prob = prior_prob

        # Statistiques MCTS
        self.visit_count = 0  # N(s,a)
        self.value_sum = 0.0  # W(s,a)
        self.children: Dict[chess.Move, MCTSNode] = {}  # Enfants

        # État du nœud
        self.is_expanded = False
        self.is_terminal = board.is_game_over()

        # Cache pour les coups légaux
        self._legal_moves = None

    @property
    def q_value(self) -> float:
        """
        Calcule Q(s,a) = W(s,a) / N(s,a).
        Retourne 0 si jamais visité.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def legal_moves(self) -> List[chess.Move]:
        """Cache des coups légaux pour éviter les recalculs."""
        if self._legal_moves is None:
            self._legal_moves = list(self.board.legal_moves)
        return self._legal_moves

    def is_fully_expanded(self) -> bool:
        """Vérifie si tous les enfants légaux ont été créés."""
        return len(self.children) == len(self.legal_moves)

    def uct_score(self, child_move: chess.Move, c_puct: float = 1.0) -> float:
        """
        Calcule le score UCT (PUCT) pour un enfant.

        Formule PUCT : Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            child_move: Coup vers l'enfant
            c_puct: Constante d'exploration

        Returns:
            Score UCT pour ce coup
        """
        child = self.children.get(child_move)

        if child is None:
            # Enfant non visité : score basé uniquement sur la prior
            return float("inf")  # Priorité maximale pour les non-visités

        # Q-value de l'enfant (du point de vue du parent)
        q_value = -child.q_value  # Négation car alternance des joueurs

        # Terme d'exploration
        prior_prob = child.prior_prob
        exploration = (
            c_puct * prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
        )

        return q_value + exploration

    def select_child(self, c_puct: float = 1.0) -> Tuple[chess.Move, "MCTSNode"]:
        """
        Sélectionne le meilleur enfant selon PUCT.

        Args:
            c_puct: Constante d'exploration

        Returns:
            Tuple (move, child_node) du meilleur enfant
        """
        if not self.legal_moves:
            raise ValueError("Aucun coup légal disponible")

        best_move = None
        best_score = float("-inf")

        for move in self.legal_moves:
            score = self.uct_score(move, c_puct)
            if score > best_score:
                best_score = score
                best_move = move

        # Créer l'enfant s'il n'existe pas
        if best_move not in self.children:
            new_board = self.board.copy()
            new_board.push(best_move)
            # Prior sera mis à jour lors de l'expansion
            self.children[best_move] = MCTSNode(new_board, self, best_move, 0.0)

        return best_move, self.children[best_move]

    def expand(self, move_probs: Dict[chess.Move, float]):
        """
        Étend le nœud en créant tous les enfants avec leurs priors.

        Args:
            move_probs: Probabilités des coups données par le réseau
        """
        if self.is_expanded:
            return

        for move in self.legal_moves:
            if move not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                prior_prob = move_probs.get(move, 0.0)
                self.children[move] = MCTSNode(new_board, self, move, prior_prob)
            else:
                # Mettre à jour la prior si l'enfant existe déjà
                self.children[move].prior_prob = move_probs.get(move, 0.0)

        self.is_expanded = True

    def backup(self, value: float):
        """
        Remonte la valeur dans l'arbre (backpropagation).

        Args:
            value: Valeur à propager (du point de vue du joueur courant)
        """
        self.visit_count += 1
        self.value_sum += value

        # Remonter vers le parent avec valeur négée (alternance des joueurs)
        if self.parent is not None:
            self.parent.backup(-value)

    def get_terminal_value(self) -> float:
        """
        Évalue la valeur d'un nœud terminal.

        Returns:
            1.0 si victoire, -1.0 si défaite, 0.0 si nul
        """
        if not self.is_terminal:
            return 0.0

        result = self.board.result()
        if result == "1-0":  # Blancs gagnent
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        elif result == "0-1":  # Noirs gagnent
            return -1.0 if self.board.turn == chess.WHITE else 1.0
        else:  # Match nul
            return 0.0


class MCTS:
    """
    Algorithme Monte Carlo Tree Search pour AlphaZero.

    Implémente les 4 étapes :
    1. Sélection avec PUCT
    2. Expansion avec le réseau de neurones
    3. Évaluation par le réseau
    4. Backpropagation des valeurs
    """

    def __init__(self, neural_net: Any, c_puct: float = 1.0, device: str = "cpu"):
        """
        Initialise l'algorithme MCTS.

        Args:
            neural_net: Réseau de neurones avec méthode predict(state)
            c_puct: Constante d'exploration PUCT
            device: Device pour les calculs torch
        """
        self.neural_net = neural_net
        self.c_puct = c_puct
        self.device = device
        self.root = None

    def predict(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
        """
        Interface avec le réseau de neurones.

        Args:
            board: Position d'échecs

        Returns:
            Tuple (move_probs, value) où :
            - move_probs : {move: probabilité} pour les coups légaux
            - value : estimation de la valeur [-1, 1]
        """
        # Si le réseau a une méthode predict, l'utiliser
        if hasattr(self.neural_net, "predict"):
            return self.neural_net.predict(board)

        # Sinon, utiliser l'interface ChessNet directement
        self.neural_net.eval()
        with torch.no_grad():
            # Encoder la position
            encoded = encode_board(board).unsqueeze(0).to(self.device)

            # Forward pass
            policy_logits, value = self.neural_net(encoded)

            # Décoder la politique pour les coups légaux
            legal_moves = list(board.legal_moves)
            move_probs = decode_policy(policy_logits[0], legal_moves)

            return move_probs, value.item()

    def search(self, state: chess.Board) -> MCTSNode:
        """
        Lance une simulation MCTS unique.

        Étapes :
        1. Sélection : descendre dans l'arbre avec PUCT
        2. Expansion : ajouter les enfants si nouveau nœud
        3. Évaluation : prédiction par le réseau
        4. Backpropagation : remonter la valeur

        Args:
            state: Position d'échecs racine

        Returns:
            Nœud racine avec statistiques mises à jour
        """
        # Créer la racine si première simulation
        if self.root is None or not self.root.board.board_fen() == state.board_fen():
            self.root = MCTSNode(state)

        node = self.root
        path = []  # Chemin traversé pour la backpropagation

        # 1. SÉLECTION : descendre dans l'arbre
        while not node.is_terminal and node.is_expanded:
            move, node = node.select_child(self.c_puct)
            path.append(node)

        # 2. EXPANSION et 3. ÉVALUATION
        if node.is_terminal:
            # Nœud terminal : utiliser la valeur exacte
            value = node.get_terminal_value()
        else:
            # Prédiction par le réseau de neurones
            move_probs, value = self.predict(node.board)

            # Expansion si pas encore fait
            if not node.is_expanded:
                node.expand(move_probs)

        # 4. BACKPROPAGATION : remonter la valeur
        node.backup(value)

        return self.root

    def run(self, state: chess.Board, num_simulations: int) -> Dict[chess.Move, float]:
        """
        Exécute plusieurs simulations MCTS et retourne la distribution des coups.

        Args:
            state: Position d'échecs de départ
            num_simulations: Nombre de simulations à effectuer

        Returns:
            Distribution π des coups : {move: probabilité normalisée}
        """
        # Réinitialiser la racine si nécessaire
        if self.root is None or not self.root.board.board_fen() == state.board_fen():
            self.root = MCTSNode(state)

        # Effectuer les simulations
        for _ in range(num_simulations):
            self.search(state)

        # Calculer la distribution des coups basée sur les visites
        move_distribution = {}
        total_visits = sum(child.visit_count for child in self.root.children.values())

        if total_visits == 0:
            # Aucune visite : distribution uniforme sur les coups légaux
            legal_moves = list(state.legal_moves)
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
            return {move: uniform_prob for move in legal_moves}

        # Distribution proportionnelle aux visites
        for move, child in self.root.children.items():
            move_distribution[move] = child.visit_count / total_visits

        return move_distribution

    def select_move(
        self, move_distribution: Dict[chess.Move, float], temperature: float = 1.0
    ) -> chess.Move:
        """
        Sélectionne un coup selon la distribution avec température.

        Args:
            move_distribution: Distribution π des coups
            temperature: Contrôle l'aléa (0 = déterministe, >0 = stochastique)

        Returns:
            Coup sélectionné
        """
        if not move_distribution:
            raise ValueError("Distribution de coups vide")

        moves = list(move_distribution.keys())
        probs = list(move_distribution.values())

        if temperature == 0.0:
            # Mode déterministe : coup avec probabilité maximale
            best_idx = np.argmax(probs)
            return moves[best_idx]
        else:
            # Mode stochastique : échantillonnage avec température
            # Appliquer la température
            log_probs = np.log(np.array(probs) + 1e-10)  # Éviter log(0)
            scaled_probs = log_probs / temperature

            # Softmax pour renormaliser
            exp_probs = np.exp(scaled_probs - np.max(scaled_probs))
            final_probs = exp_probs / np.sum(exp_probs)

            # Échantillonnage
            selected_idx = np.random.choice(len(moves), p=final_probs)
            return moves[selected_idx]

    def get_action_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées de la racine pour debug.

        Returns:
            Dictionnaire avec statistiques des coups
        """
        if self.root is None:
            return {}

        stats = {
            "total_visits": self.root.visit_count,
            "q_value": self.root.q_value,
            "children_stats": {},
        }

        for move, child in self.root.children.items():
            stats["children_stats"][str(move)] = {
                "visits": child.visit_count,
                "q_value": child.q_value,
                "prior_prob": child.prior_prob,
                "uct_score": self.root.uct_score(move, self.c_puct),
            }

        return stats

    def reset(self):
        """Réinitialise l'arbre MCTS."""
        self.root = None


# Exemple d'utilisation et tests
if __name__ == "__main__":
    import torch
    from .network import ChessNet

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device : {device}")

    # Créer le réseau (modèle aléatoire pour test)
    net = ChessNet().to(device)
    net.eval()

    # Créer l'algorithme MCTS
    mcts = MCTS(net, c_puct=1.4, device=device)

    # Position de test
    board = chess.Board()
    print(f"Position initiale : {board.fen()}")
    print(f"Coups légaux : {len(list(board.legal_moves))}")

    # Effectuer des simulations
    num_simulations = 100
    print(f"\nExécution de {num_simulations} simulations MCTS...")

    move_distribution = mcts.run(board, num_simulations)

    # Afficher les résultats
    print("\nDistribution des coups (top 10) :")
    sorted_moves = sorted(move_distribution.items(), key=lambda x: x[1], reverse=True)
    for i, (move, prob) in enumerate(sorted_moves[:10]):
        print(f"{i+1:2d}. {move} : {prob:.4f}")

    # Sélectionner un coup
    selected_move = mcts.select_move(move_distribution, temperature=0.0)
    print(f"\nCoup sélectionné (température=0) : {selected_move}")

    selected_move_stoch = mcts.select_move(move_distribution, temperature=1.0)
    print(f"Coup sélectionné (température=1) : {selected_move_stoch}")

    # Afficher les statistiques
    stats = mcts.get_action_stats()
    print(f"\nStatistiques globales :")
    print(f"Total des visites : {stats['total_visits']}")
    print(f"Q-value racine : {stats['q_value']:.4f}")
