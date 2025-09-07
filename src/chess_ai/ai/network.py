"""
Réseau de neurones AlphaZero pour les échecs
==========================================

Implémente :
- Encodage des positions d'échecs en tenseurs
- Architecture CNN résiduelle avec deux têtes (politique et valeur)
- Décodage des politiques avec filtrage des coups légaux
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from typing import Dict, List, Tuple, Optional


def encode_board(
    board: chess.Board, history: Optional[List[chess.Board]] = None
) -> torch.FloatTensor:
    """
    Encode un état d'échecs en tenseur pour le réseau AlphaZero.

    Args:
        board: Position d'échecs actuelle
        history: Historique des positions (optionnel)

    Returns:
        Tenseur (N, 8, 8) où N est le nombre de canaux :
        - 6 canaux pour pièces blanches (P, N, B, R, Q, K)
        - 6 canaux pour pièces noires (p, n, b, r, q, k)
        - 1 canal pour le tour (1 si blancs, 0 si noirs)
        - 2 canaux pour droits de roque (blancs, noirs)
        - 1 canal pour en passant
        - 2 canaux pour historique récent (optionnel)
    """
    # Configuration des canaux
    piece_channels = 12  # 6 blanches + 6 noires
    turn_channel = 1
    castling_channels = 2
    en_passant_channel = 1
    history_channels = 2 if history else 0

    total_channels = (
        piece_channels
        + turn_channel
        + castling_channels
        + en_passant_channel
        + history_channels
    )

    # Initialiser le tenseur
    tensor = torch.zeros((total_channels, 8, 8), dtype=torch.float32)

    # Canal pour chaque type de pièce
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Encoder les pièces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file, rank = chess.square_file(square), chess.square_rank(square)
            base_channel = piece_to_channel[piece.piece_type]

            if piece.color == chess.WHITE:
                channel = base_channel
            else:
                channel = base_channel + 6

            tensor[channel, rank, file] = 1.0

    # Canal du tour (12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Canaux des droits de roque (13-14)
    if board.has_kingside_castling_rights(
        chess.WHITE
    ) or board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(
        chess.BLACK
    ) or board.has_queenside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0

    # Canal en passant (15)
    if board.ep_square is not None:
        ep_file, ep_rank = chess.square_file(board.ep_square), chess.square_rank(
            board.ep_square
        )
        tensor[15, ep_rank, ep_file] = 1.0

    # Canaux d'historique (16-17) - optionnel
    if history and len(history) >= 2:
        # Position précédente
        prev_board = history[-1]
        for square in chess.SQUARES:
            piece = prev_board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                file, rank = chess.square_file(square), chess.square_rank(square)
                tensor[16, rank, file] = 1.0

        # Position avant-précédente
        if len(history) >= 2:
            prev2_board = history[-2]
            for square in chess.SQUARES:
                piece = prev2_board.piece_at(square)
                if piece and piece.color == chess.WHITE:
                    file, rank = chess.square_file(square), chess.square_rank(square)
                    tensor[17, rank, file] = 1.0

    return tensor


def move_to_index(move: chess.Move) -> int:
    """
    🚀 CORRECTION : Convertit un coup d'échecs en index pour le réseau neuronal.

    Mapping unifié basé sur les squares de départ et d'arrivée.

    Args:
        move: Coup d'échecs

    Returns:
        Index entre 0 et 4671
    """
    from_square = move.from_square
    to_square = move.to_square

    # Mapping basique : 64*64 = 4096 combinaisons possibles
    base_index = from_square * 64 + to_square

    # Gérer les promotions
    if move.promotion:
        # Ajouter offset pour les promotions (4 types × 64 positions = 256)
        promotion_offset = 4096
        promotion_type = move.promotion - 1  # 0-3 pour Queen, Rook, Bishop, Knight
        base_index = promotion_offset + to_square * 4 + promotion_type

    # S'assurer que l'index est dans les limites
    return min(base_index, 4671)  # 4672 - 1


def index_to_move(index: int) -> chess.Move:
    """
    Convertit un index de politique en mouvement.
    """
    # Gérer les promotions
    if index >= 12288:  # Knight promotion
        index -= 12288
        promotion = chess.KNIGHT
    elif index >= 8192:  # Bishop promotion
        index -= 8192
        promotion = chess.BISHOP
    elif index >= 4096:  # Rook promotion
        index -= 4096
        promotion = chess.ROOK
    elif index >= 4096:  # Queen promotion (implicit)
        promotion = chess.QUEEN
    else:
        promotion = None

    from_square = index // 64
    to_square = index % 64

    return chess.Move(from_square, to_square, promotion)


class ResidualBlock(nn.Module):
    """Bloc résiduel pour le backbone CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)

        return x


class ChessNet(nn.Module):
    """
    Réseau de neurones AlphaZero pour les échecs.

    Architecture :
    - Couche d'entrée CNN
    - Blocks résiduels
    - Deux têtes : politique et valeur
    """

    def __init__(
        self,
        input_channels: int = 16,
        hidden_channels: int = 256,
        num_residual_blocks: int = 5,
        policy_size: int = 4672,
    ):
        super().__init__()

        # Couche d'entrée
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_channels)

        # Blocks résiduels
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)]
        )

        # Tête de politique
        self.policy_conv = nn.Conv2d(hidden_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)

        # Tête de valeur
        self.value_conv = nn.Conv2d(hidden_channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass du réseau.

        Args:
            x: Batch de positions (B, C, 8, 8)

        Returns:
            policy_logits: (B, policy_size)
            value: (B, 1)
        """
        # Backbone
        x = F.relu(self.input_bn(self.input_conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        # Tête de politique
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Tête de valeur
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value


def decode_policy(
    policy_logits: torch.Tensor, legal_moves: List[chess.Move], temperature: float = 1.0
) -> Dict[chess.Move, float]:
    """
    Décode les logits de politique en probabilités pour les coups légaux.

    Args:
        policy_logits: Logits du réseau (taille policy_size)
        legal_moves: Liste des coups légaux
        temperature: Température pour le softmax (1.0 = normal, <1.0 = plus déterministe)

    Returns:
        Dictionnaire {move: probability} pour les coups légaux uniquement
    """
    if not legal_moves:
        return {}

    try:
        # Extraire les logits pour les coups légaux
        legal_indices = [move_to_index(move) for move in legal_moves]

        # Vérifier que tous les indices sont valides
        max_index = policy_logits.size(0) - 1
        legal_indices = [min(idx, max_index) for idx in legal_indices]

        legal_logits = policy_logits[legal_indices]

        # Appliquer la température
        if temperature != 1.0:
            legal_logits = legal_logits / temperature

        # Softmax pour obtenir les probabilités
        legal_probs = F.softmax(legal_logits, dim=0)

        # Créer le dictionnaire move -> probabilité
        move_probs = {}
        for move, prob in zip(legal_moves, legal_probs):
            move_probs[move] = prob.item()

        return move_probs

    except Exception as e:
        # Fallback : probabilités uniformes
        uniform_prob = 1.0 / len(legal_moves)
        return {move: uniform_prob for move in legal_moves}


def batch_encode_boards(boards: List[chess.Board]) -> torch.FloatTensor:
    """
    Encode un batch de positions d'échecs.

    Args:
        boards: Liste de positions

    Returns:
        Tenseur (B, C, 8, 8)
    """
    encoded_boards = [encode_board(board) for board in boards]
    return torch.stack(encoded_boards, dim=0)


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer une position d'échecs
    board = chess.Board()

    # Encoder la position
    encoded = encode_board(board)
    print(f"Forme du tenseur encodé : {encoded.shape}")

    # Créer le réseau
    net = ChessNet()

    # Forward pass avec un batch de taille 1
    batch = encoded.unsqueeze(0)  # (1, C, 8, 8)
    policy_logits, value = net(batch)

    print(f"Politique shape : {policy_logits.shape}")
    print(f"Valeur shape : {value.shape}")
    print(f"Valeur estimée : {value.item():.3f}")

    # Décoder la politique pour les coups légaux
    legal_moves = list(board.legal_moves)
    move_probs = decode_policy(policy_logits[0], legal_moves)

    print(f"Coups légaux : {len(legal_moves)}")
    print("Top 5 coups :")
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    for move, prob in sorted_moves[:5]:
        print(f"  {move} : {prob:.4f}")
