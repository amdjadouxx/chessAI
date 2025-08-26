"""
Intégration AlphaZero avec l'interface de jeu
===========================================

Ce module permet d'utiliser l'IA AlphaZero dans l'interface graphique.
Supporte les modes réseau simple et MCTS avancé.
"""

import chess
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from ..ai.network import ChessNet, encode_board, decode_policy
from ..ai.mcts import MCTS


class AlphaZeroPlayer:
    """
    Joueur IA utilisant le réseau AlphaZero.
    Supporte les modes réseau simple et MCTS avancé.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_mcts: bool = False,
        mcts_simulations: int = 400,
        c_puct: float = 1.4,
    ):
        """
        Initialise le joueur IA.

        Args:
            model_path: Chemin vers un modèle pré-entraîné (optionnel)
            device: Device pour l'inférence ("cpu" ou "cuda")
            use_mcts: Utiliser MCTS (True) ou évaluation directe (False)
            mcts_simulations: Nombre de simulations MCTS si activé
            c_puct: Constante d'exploration PUCT pour MCTS
        """
        self.device = torch.device(device)
        self.net = ChessNet()
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations

        if model_path:
            self.load_model(model_path)

        self.net.to(self.device)
        self.net.eval()

        # Initialiser MCTS si demandé
        if use_mcts:
            self.mcts = MCTS(self.net, c_puct=c_puct, device=str(self.device))
            print(f"🤖 IA MCTS initialisée ({mcts_simulations} simulations)")
        else:
            self.mcts = None
            print("🤖 IA en mode évaluation directe")

    def load_model(self, model_path: str):
        """Charge un modèle pré-entraîné."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Modèle chargé depuis {model_path}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")

    def save_model(self, model_path: str):
        """Sauvegarde le modèle."""
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
        }
        torch.save(checkpoint, model_path)
        print(f"✅ Modèle sauvegardé vers {model_path}")

    def evaluate_position(self, board: chess.Board) -> Tuple[float, dict]:
        """
        Évalue une position et retourne la valeur + politique.

        Args:
            board: Position à évaluer

        Returns:
            Tuple (valeur, {move: probability})
        """
        # Encoder la position
        encoded = encode_board(board)
        batch = encoded.unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            policy_logits, value = self.net(batch)

        # Décoder la politique pour les coups légaux
        legal_moves = list(board.legal_moves)
        move_probs = decode_policy(policy_logits[0].cpu(), legal_moves)

        return value.item(), move_probs

    def select_move(self, board: chess.Board, temperature: float = 0.1) -> chess.Move:
        """
        Sélectionne le meilleur coup selon l'IA.

        Args:
            board: Position actuelle
            temperature: Température pour la sélection (0 = déterministe)

        Returns:
            Meilleur coup selon l'IA
        """
        if self.use_mcts and self.mcts:
            # Mode MCTS avancé
            move_distribution = self.mcts.run(board, self.mcts_simulations)
            if not move_distribution:
                return None
            return self.mcts.select_move(move_distribution, temperature)
        else:
            # Mode évaluation directe
            value, move_probs = self.evaluate_position(board)

            if not move_probs:
                # Aucun coup légal (situation anormale)
                return None

            if temperature == 0:
                # Sélection déterministe
                best_move = max(move_probs.items(), key=lambda x: x[1])[0]
            else:
                # Sélection probabiliste
                moves = list(move_probs.keys())
                probs = list(move_probs.values())

                # Appliquer la température
                probs = torch.tensor(probs)
                if temperature != 1.0:
                    probs = probs / temperature

                probs = F.softmax(probs, dim=0)

                # Échantillonner
                idx = torch.multinomial(probs, 1).item()
                best_move = moves[idx]

            return best_move

    def get_move_analysis(self, board: chess.Board, top_k: int = 5) -> dict:
        """
        Analyse détaillée d'une position.

        Args:
            board: Position à analyser
            top_k: Nombre de meilleurs coups à retourner

        Returns:
            Dictionnaire avec l'analyse
        """
        if self.use_mcts and self.mcts:
            # Analyse MCTS
            move_distribution = self.mcts.run(board, self.mcts_simulations)
            value, _ = self.evaluate_position(board)  # Valeur du réseau

            # Statistiques MCTS
            mcts_stats = self.mcts.get_action_stats()

            # Trier par distribution MCTS
            sorted_moves = sorted(
                move_distribution.items(), key=lambda x: x[1], reverse=True
            )

            analysis = {
                "evaluation": value,
                "turn": "Blancs" if board.turn else "Noirs",
                "top_moves": sorted_moves[:top_k],
                "total_legal_moves": len(move_distribution),
                "mcts_enabled": True,
                "mcts_simulations": self.mcts_simulations,
                "mcts_visits": mcts_stats.get("total_visits", 0),
            }
        else:
            # Analyse réseau direct
            value, move_probs = self.evaluate_position(board)

            # Trier les coups par probabilité
            sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)

            analysis = {
                "evaluation": value,
                "turn": "Blancs" if board.turn else "Noirs",
                "top_moves": sorted_moves[:top_k],
                "total_legal_moves": len(move_probs),
                "mcts_enabled": False,
            }

        return analysis

    def analyze_position(self, board: chess.Board):
        """Alias pour get_move_analysis pour compatibilité."""
        return self.get_move_analysis(board)

    def get_move(self, board: chess.Board):
        """Alias pour select_move pour compatibilité."""
        return self.select_move(board)

    def enable_mcts(self, simulations: int = 400, c_puct: float = 1.4):
        """Active le mode MCTS."""
        self.use_mcts = True
        self.mcts_simulations = simulations
        if not self.mcts:
            self.mcts = MCTS(self.net, c_puct=c_puct, device=str(self.device))
        print(f"🤖 MCTS activé ({simulations} simulations)")

    def disable_mcts(self):
        """Désactive le mode MCTS."""
        self.use_mcts = False
        print("🤖 MCTS désactivé - mode évaluation directe")

    def reset_mcts(self):
        """Réinitialise l'arbre MCTS."""
        if self.mcts:
            self.mcts.reset()


class MCTSAlphaZeroPlayer(AlphaZeroPlayer):
    """
    Version spécialisée avec MCTS toujours activé.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        mcts_simulations: int = 800,
        c_puct: float = 1.4,
        temperature: float = 1.0,
    ):
        """
        Initialise un joueur MCTS AlphaZero.

        Args:
            model_path: Chemin vers le modèle
            device: Device de calcul
            mcts_simulations: Nombre de simulations par coup
            c_puct: Constante d'exploration
            temperature: Température par défaut
        """
        super().__init__(
            model_path,
            device,
            use_mcts=True,
            mcts_simulations=mcts_simulations,
            c_puct=c_puct,
        )
        self.default_temperature = temperature

    def get_move(self, board: chess.Board, temperature: Optional[float] = None):
        """Sélectionne un coup avec la température par défaut."""
        temp = temperature if temperature is not None else self.default_temperature
        return self.select_move(board, temp)


def add_ai_to_gui(gui_class):
    """
    Décorateur pour ajouter l'IA à une classe GUI.

    Usage:
        @add_ai_to_gui
        class MyChessGUI:
            ...
    """
    # Ajouter l'IA comme attribut
    original_init = gui_class.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Initialiser l'IA
        try:
            self.ai_player = AlphaZeroPlayer()
            self.ai_enabled = True
            print("🤖 IA AlphaZero activée")
        except Exception as e:
            print(f"⚠️  IA non disponible : {e}")
            self.ai_player = None
            self.ai_enabled = False

    gui_class.__init__ = new_init

    # Ajouter méthodes d'IA
    def get_ai_move(self):
        """Obtient le coup suggéré par l'IA."""
        if not self.ai_enabled or not hasattr(self, "environment"):
            return None

        try:
            board = self.environment.board
            move = self.ai_player.select_move(board)
            return move
        except Exception as e:
            print(f"❌ Erreur IA : {e}")
            return None

    def get_ai_analysis(self):
        """Obtient l'analyse de la position par l'IA."""
        if not self.ai_enabled or not hasattr(self, "environment"):
            return None

        try:
            board = self.environment.board
            analysis = self.ai_player.get_move_analysis(board)
            return analysis
        except Exception as e:
            print(f"❌ Erreur analyse IA : {e}")
            return None

    def toggle_ai_hint(self):
        """Active/désactive les suggestions de l'IA."""
        if hasattr(self, "show_ai_hints"):
            self.show_ai_hints = not self.show_ai_hints
        else:
            self.show_ai_hints = True

        status = "activées" if self.show_ai_hints else "désactivées"
        print(f"🤖 Suggestions IA {status}")

    # Ajouter les méthodes à la classe
    gui_class.get_ai_move = get_ai_move
    gui_class.get_ai_analysis = get_ai_analysis
    gui_class.toggle_ai_hint = toggle_ai_hint

    return gui_class


# Exemple d'utilisation simple
def demo_ai_vs_random():
    """Démonstration : AlphaZero vs Random avec et sans MCTS."""
    import random

    print("🤖 Démonstration : AlphaZero vs Random")
    print("=" * 40)

    # Test mode direct
    print("\n1. Mode évaluation directe:")
    board = chess.Board()
    ai_direct = AlphaZeroPlayer(use_mcts=False)

    analysis = ai_direct.get_move_analysis(board, top_k=3)
    print(f"   Évaluation : {analysis['evaluation']:+.3f}")
    print("   Top 3 coups :")
    for i, (move, prob) in enumerate(analysis["top_moves"], 1):
        print(f"      {i}. {move} ({prob:.3f})")

    # Test mode MCTS
    print("\n2. Mode MCTS (100 simulations):")
    ai_mcts = AlphaZeroPlayer(use_mcts=True, mcts_simulations=100)

    analysis_mcts = ai_mcts.get_move_analysis(board, top_k=3)
    print(f"   Évaluation : {analysis_mcts['evaluation']:+.3f}")
    print(f"   Visites MCTS : {analysis_mcts.get('mcts_visits', 0)}")
    print("   Top 3 coups :")
    for i, (move, prob) in enumerate(analysis_mcts["top_moves"], 1):
        print(f"      {i}. {move} ({prob:.3f})")

    # Partie courte
    print("\n3. Partie courte (MCTS vs Random):")
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < 10:
        print(f"\nCoup {move_count + 1} ({'Blancs' if board.turn else 'Noirs'}):")

        if board.turn == chess.WHITE:
            # IA MCTS joue les blancs
            move = ai_mcts.select_move(board, temperature=0.5)
            print(f"   IA MCTS joue : {move}")
        else:
            # Coups aléatoires pour les noirs
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            print(f"   Random joue : {move}")

        board.push(move)
        move_count += 1

    print(f"\n📋 Partie terminée après {move_count} coups")
    print(f"Résultat : {board.result()}")


if __name__ == "__main__":
    demo_ai_vs_random()
