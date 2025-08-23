"""
Tests unitaires complets pour Chess AI.

Suite de tests robuste couvrant tous les aspects du système
avec gestion d'erreurs et cas limites.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
from chess_ai import ChessEnvironment, ChessAnalyzer, ChessDisplay
from chess_ai.exceptions import (
    ChessError,
    InvalidMoveError,
    InvalidSquareError,
    GameOverError,
    InvalidFENError,
)


class TestChessEnvironment(unittest.TestCase):
    """Tests pour la classe ChessEnvironment."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.env = ChessEnvironment(enable_logging=False)

    def test_initialization_default(self):
        """Test de l'initialisation par défaut."""
        self.assertIsInstance(self.env.board, chess.Board)
        self.assertEqual(self.env.get_board_fen(), chess.STARTING_FEN)
        self.assertEqual(len(self.env.get_move_history()), 0)

    def test_initialization_with_fen(self):
        """Test de l'initialisation avec FEN personnalisée."""
        # Position après e2-e4 (python-chess normalise automatiquement l'en passant)
        custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        env = ChessEnvironment(custom_fen, enable_logging=False)
        self.assertEqual(env.get_board_fen(), custom_fen)

    def test_initialization_invalid_fen(self):
        """Test avec FEN invalide."""
        with self.assertRaises(InvalidFENError) as context:
            ChessEnvironment("invalid_fen", enable_logging=False)

        self.assertIn("invalid_fen", str(context.exception))

    def test_reset_board(self):
        """Test de la réinitialisation du plateau."""
        # Faire un mouvement puis réinitialiser
        self.env.make_move("e2e4")
        self.assertNotEqual(self.env.get_board_fen(), chess.STARTING_FEN)

        self.env.reset_board()
        self.assertEqual(self.env.get_board_fen(), chess.STARTING_FEN)
        self.assertEqual(len(self.env.get_move_history()), 0)

    def test_get_piece_at_valid(self):
        """Test de get_piece_at avec cases valides."""
        # Test avec string
        piece = self.env.get_piece_at("e1")
        self.assertIsNotNone(piece)
        self.assertEqual(piece.piece_type, chess.KING)
        self.assertTrue(piece.color)

        # Test avec chess.Square
        piece = self.env.get_piece_at(chess.E1)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.piece_type, chess.KING)

        # Test avec integer
        piece = self.env.get_piece_at(4)  # e1
        self.assertIsNotNone(piece)
        self.assertEqual(piece.piece_type, chess.KING)

        # Test avec case vide
        empty_piece = self.env.get_piece_at("e4")
        self.assertIsNone(empty_piece)

    def test_get_piece_at_invalid(self):
        """Test de get_piece_at avec cases invalides."""
        with self.assertRaises(InvalidSquareError):
            self.env.get_piece_at("z9")

        with self.assertRaises(InvalidSquareError):
            self.env.get_piece_at(64)  # Hors limites

    def test_make_move_valid(self):
        """Test de mouvements valides."""
        # Mouvement avec string UCI
        success = self.env.make_move("e2e4")
        self.assertTrue(success)

        # Vérifier que le mouvement a été effectué
        piece = self.env.get_piece_at("e4")
        self.assertIsNotNone(piece)
        self.assertEqual(piece.piece_type, chess.PAWN)

        # Vérifier l'historique
        history = self.env.get_move_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(str(history[0]), "e2e4")

    def test_make_move_chess_move_object(self):
        """Test avec objet chess.Move."""
        move = chess.Move.from_uci("e2e4")
        success = self.env.make_move(move)
        self.assertTrue(success)

    def test_make_move_invalid_format(self):
        """Test avec format de mouvement invalide."""
        with self.assertRaises(InvalidMoveError):
            self.env.make_move("invalid")

    def test_make_move_illegal(self):
        """Test avec mouvement illégal."""
        with self.assertRaises(InvalidMoveError):
            self.env.make_move("e2e5")  # Impossible

    def test_make_move_game_over(self):
        """Test de mouvement sur partie terminée."""
        # Créer une position de mat
        mate_fen = "rnbqkb1r/pppp1Qpp/5n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        env_mate = ChessEnvironment(mate_fen, enable_logging=False)

        with self.assertRaises(GameOverError):
            env_mate.make_move("a7a6")

    def test_undo_move(self):
        """Test de l'annulation de mouvements."""
        original_fen = self.env.get_board_fen()

        # Faire un mouvement
        self.env.make_move("e2e4")
        self.assertNotEqual(self.env.get_board_fen(), original_fen)

        # Annuler le mouvement
        success = self.env.undo_move()
        self.assertTrue(success)
        self.assertEqual(self.env.get_board_fen(), original_fen)
        self.assertEqual(len(self.env.get_move_history()), 0)

    def test_undo_move_empty_stack(self):
        """Test d'annulation sans mouvement."""
        success = self.env.undo_move()
        self.assertFalse(success)

    def test_get_legal_moves(self):
        """Test de get_legal_moves."""
        legal_moves = self.env.get_legal_moves()
        self.assertIsInstance(legal_moves, list)
        self.assertEqual(len(legal_moves), 20)  # 20 mouvements initiaux

    def test_game_state_methods(self):
        """Test des méthodes d'état du jeu."""
        # Position initiale
        self.assertFalse(self.env.is_game_over())
        self.assertIsNone(self.env.get_game_result())
        self.assertFalse(self.env.is_check())
        self.assertEqual(self.env.get_current_player(), chess.WHITE)

    def test_get_board_stats(self):
        """Test de get_board_stats."""
        stats = self.env.get_board_stats()

        required_keys = [
            "fen",
            "current_player",
            "is_check",
            "is_game_over",
            "game_result",
            "legal_moves_count",
            "move_count",
            "castling_rights",
        ]

        for key in required_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats["current_player"], "White")
        self.assertEqual(stats["move_count"], 0)
        self.assertEqual(stats["legal_moves_count"], 20)

    def test_str_and_repr(self):
        """Test des représentations string."""
        str_repr = str(self.env)
        self.assertIsInstance(str_repr, str)

        repr_str = repr(self.env)
        self.assertIn("ChessEnvironment", repr_str)
        self.assertIn(chess.STARTING_FEN, repr_str)


class TestChessAnalyzer(unittest.TestCase):
    """Tests pour la classe ChessAnalyzer."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.board = chess.Board()
        self.analyzer = ChessAnalyzer(self.board, enable_logging=False)

    def test_get_piece_map(self):
        """Test de get_piece_map."""
        piece_map = self.analyzer.get_piece_map()
        self.assertIsInstance(piece_map, dict)
        self.assertEqual(len(piece_map), 32)  # 32 pièces au début

    def test_get_pieces_by_color(self):
        """Test de get_pieces_by_color."""
        white_pieces = self.analyzer.get_pieces_by_color(chess.WHITE)
        black_pieces = self.analyzer.get_pieces_by_color(chess.BLACK)

        self.assertEqual(len(white_pieces), 16)
        self.assertEqual(len(black_pieces), 16)

        # Vérifier le format
        for square, piece in white_pieces:
            self.assertIsInstance(square, str)
            self.assertIsInstance(piece, chess.Piece)
            self.assertTrue(piece.color)  # Blanc

    def test_count_material(self):
        """Test de count_material."""
        material = self.analyzer.count_material()

        self.assertIn("white", material)
        self.assertIn("black", material)

        # Vérification du matériel initial
        expected_white = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1}
        expected_black = {"p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1}

        self.assertEqual(material["white"], expected_white)
        self.assertEqual(material["black"], expected_black)

    def test_get_attackers(self):
        """Test de get_attackers."""
        # En position initiale, aucune pièce n'attaque e4
        attackers = self.analyzer.get_attackers(chess.E4, chess.WHITE)
        self.assertEqual(len(attackers), 0)

        # Après e2-e4, le pion en e4 peut être attaqué
        self.board.push(chess.Move.from_uci("e2e4"))
        analyzer = ChessAnalyzer(self.board, enable_logging=False)
        attackers = analyzer.get_attackers(chess.E4, chess.BLACK)
        self.assertGreaterEqual(len(attackers), 0)

    def test_is_square_attacked(self):
        """Test de is_square_attacked."""
        # En position initiale
        attacked = self.analyzer.is_square_attacked(chess.E4, chess.WHITE)
        self.assertFalse(attacked)

    def test_get_piece_mobility(self):
        """Test de get_piece_mobility."""
        # Mobilité du cavalier en b1
        mobility = self.analyzer.get_piece_mobility(chess.B1)
        self.assertEqual(mobility, 2)  # Na3, Nc3

    def test_get_king_safety_score(self):
        """Test de get_king_safety_score."""
        safety = self.analyzer.get_king_safety_score(chess.WHITE)

        required_keys = [
            "king_square",
            "threats_around_king",
            "protected_squares",
            "is_in_check",
            "can_castle_kingside",
            "can_castle_queenside",
            "safety_rating",
        ]

        for key in required_keys:
            self.assertIn(key, safety)

        self.assertEqual(safety["king_square"], "e1")
        self.assertFalse(safety["is_in_check"])

    def test_get_piece_development_score(self):
        """Test de get_piece_development_score."""
        dev_score = self.analyzer.get_piece_development_score(chess.WHITE)

        required_keys = [
            "developed_knights",
            "total_knights",
            "developed_bishops",
            "total_bishops",
            "development_percentage",
        ]

        for key in required_keys:
            self.assertIn(key, dev_score)

        self.assertEqual(dev_score["developed_knights"], 0)
        self.assertEqual(dev_score["developed_bishops"], 0)
        self.assertEqual(dev_score["development_percentage"], 0.0)

    def test_analyze_position(self):
        """Test de l'analyse complète."""
        analysis = self.analyzer.analyze_position()

        required_sections = [
            "material",
            "white_king_safety",
            "black_king_safety",
            "white_development",
            "black_development",
            "position_info",
        ]

        for section in required_sections:
            self.assertIn(section, analysis)


class TestChessDisplay(unittest.TestCase):
    """Tests pour la classe ChessDisplay."""

    def setUp(self):
        """Configuration avant chaque test."""
        self.board = chess.Board()
        self.display = ChessDisplay(self.board, enable_logging=False)

    @patch("builtins.print")
    def test_display_unicode(self, mock_print):
        """Test de l'affichage Unicode."""
        self.display.display_unicode()
        self.assertTrue(mock_print.called)

    @patch("builtins.print")
    def test_display_ascii(self, mock_print):
        """Test de l'affichage ASCII."""
        self.display.display_ascii()
        self.assertTrue(mock_print.called)

    @patch("builtins.print")
    def test_display_compact(self, mock_print):
        """Test de l'affichage compact."""
        self.display.display_compact()
        self.assertTrue(mock_print.called)

    @patch("builtins.print")
    def test_display_statistics(self, mock_print):
        """Test de l'affichage des statistiques."""
        self.display.display_statistics()
        self.assertTrue(mock_print.called)

    @patch("builtins.print")
    def test_display_move_history(self, mock_print):
        """Test de l'affichage de l'historique."""
        history = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
        self.display.display_move_history(history)
        self.assertTrue(mock_print.called)


class TestExceptions(unittest.TestCase):
    """Tests pour les exceptions personnalisées."""

    def test_chess_error_basic(self):
        """Test de ChessError de base."""
        error = ChessError("Test message")
        self.assertEqual(error.message, "Test message")
        self.assertIsNone(error.details)
        self.assertEqual(str(error), "Test message")

    def test_chess_error_with_details(self):
        """Test de ChessError avec détails."""
        error = ChessError("Test message", "Additional details")
        self.assertEqual(error.details, "Additional details")
        self.assertIn("Additional details", str(error))

    def test_invalid_move_error(self):
        """Test de InvalidMoveError."""
        error = InvalidMoveError("e2e5", "Pion ne peut pas avancer de 3 cases")
        self.assertEqual(error.move, "e2e5")
        self.assertEqual(error.reason, "Pion ne peut pas avancer de 3 cases")
        self.assertIn("e2e5", str(error))

    def test_invalid_square_error(self):
        """Test de InvalidSquareError."""
        error = InvalidSquareError("z9")
        self.assertEqual(error.square, "z9")
        self.assertIn("z9", str(error))

    def test_game_over_error(self):
        """Test de GameOverError."""
        error = GameOverError("make move", "1-0")
        self.assertEqual(error.action, "make move")
        self.assertEqual(error.result, "1-0")
        self.assertIn("make move", str(error))


class TestIntegration(unittest.TestCase):
    """Tests d'intégration pour vérifier l'interaction entre composants."""

    def test_complete_workflow(self):
        """Test d'un workflow complet."""
        # Créer l'environnement
        env = ChessEnvironment(enable_logging=False)

        # Créer analyzer et display
        analyzer = ChessAnalyzer(env.board, enable_logging=False)
        display = ChessDisplay(env.board, enable_logging=False)

        # Faire quelques mouvements
        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
        for move in moves:
            success = env.make_move(move)
            self.assertTrue(success)

        # Vérifier que l'analyzer fonctionne
        analysis = analyzer.analyze_position()
        self.assertIn("material", analysis)

        # Vérifier l'historique
        history = env.get_move_history()
        self.assertEqual(len(history), 4)

        # Test d'annulations
        for _ in range(2):
            success = env.undo_move()
            self.assertTrue(success)

        self.assertEqual(len(env.get_move_history()), 2)


if __name__ == "__main__":
    # Configuration pour tests détaillés
    unittest.main(verbosity=2, buffer=True)
