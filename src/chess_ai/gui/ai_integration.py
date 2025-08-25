"""
Intégration AlphaZero avec l'interface de jeu
===========================================

Ce module permet d'utiliser l'IA AlphaZero dans l'interface graphique.
"""

import chess
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from ..ai.network import ChessNet, encode_board, decode_policy


class AlphaZeroPlayer:
    """
    Joueur IA utilisant le réseau AlphaZero.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialise le joueur IA.

        Args:
            model_path: Chemin vers un modèle pré-entraîné (optionnel)
            device: Device pour l'inférence ("cpu" ou "cuda")
        """
        self.device = torch.device(device)
        self.net = ChessNet()

        if model_path:
            self.load_model(model_path)

        self.net.to(self.device)
        self.net.eval()

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
        value, move_probs = self.evaluate_position(board)

        # Trier les coups par probabilité
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)

        analysis = {
            "evaluation": value,
            "turn": "Blancs" if board.turn else "Noirs",
            "top_moves": sorted_moves[:top_k],
            "total_legal_moves": len(move_probs),
        }

        return analysis

    def analyze_position(self, board: chess.Board):
        """Alias pour get_move_analysis pour compatibilité."""
        return self.get_move_analysis(board)

    def get_move(self, board: chess.Board):
        """Alias pour select_move pour compatibilité."""
        return self.select_move(board)


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
    """Démonstration : IA vs coups aléatoires."""
    import random

    print("🤖 Démonstration : AlphaZero vs Random")
    print("=" * 40)

    board = chess.Board()
    ai = AlphaZeroPlayer()

    move_count = 0

    while not board.is_game_over() and move_count < 20:
        print(f"\nCoup {move_count + 1} ({'Blancs' if board.turn else 'Noirs'}):")

        if board.turn == chess.WHITE:
            # IA joue les blancs
            analysis = ai.get_move_analysis(board, top_k=3)
            print(f"   Évaluation IA : {analysis['evaluation']:+.3f}")
            print("   Top 3 coups :")
            for i, (move, prob) in enumerate(analysis["top_moves"], 1):
                print(f"      {i}. {move} ({prob:.3f})")

            move = ai.select_move(board)
            print(f"   IA joue : {move}")
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
