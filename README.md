# Chess AI - Architecture Modulaire avec Interface Graphique 🎮

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GUI: Pygame](https://img.shields.io/badge/GUI-Pygame-green.svg)](https://www.pygame.org/)

## 🎯 Vue d'ensemble

Chess AI est une implémentation **modulaire** et **robuste** d'un système de gestion de plateau d'échecs, construite sur la librairie `python-chess`. Cette architecture offre une **gestion d'erreurs exhaustive**, un **logging complet**, une **séparation claire des responsabilités**, et maintenant une **interface graphique moderne** !

### ✨ Caractéristiques principales

- 🏗️ **Architecture modulaire** avec séparation des responsabilités
- 🎮 **Interface graphique moderne** avec Pygame (animations, interactions)
- 🛡️ **Gestion d'erreurs robuste** avec exceptions personnalisées
- 📝 **Logging robuste** pour traçabilité complète
- 🧪 **Tests unitaires complets** avec couverture étendue
- 📚 **Documentation exhaustive** avec exemples détaillés
- 🎨 **Multiples modes d'affichage** (Unicode, ASCII, compact, graphique)
- 🔍 **Analyse avancée** de positions avec métriques détaillées
- ⚡ **Performance optimisée** utilisant pleinement python-chess

## 🎮 Interface Graphique - NOUVEAU !

### Lancement Rapide
```python
from chess_ai import launch_gui

# Interface graphique en une ligne !
launch_gui()
```

### Fonctionnalités Graphiques
- 🖱️ **Interactions souris** : Clic pour sélectionner/déplacer
- 🎞️ **Animations fluides** : Mouvements animés des pièces
- 🎨 **Highlighting intelligent** : Cases sélectionnées, mouvements légaux
- 🔄 **Perspectives multiples** : Vue blancs/noirs
- ⌨️ **Raccourcis clavier** : N (nouveau), U (annuler), F (retourner)
- 📊 **Informations temps réel** : État partie, tour, mouvements légaux

![Chess AI GUI Preview](docs/images/chess_ai_gui_preview.png)

## 🏗️ Architecture

```
chess_ai/
├── src/chess_ai/               # Code source principal
│   ├── core/                   # Composants principaux
│   │   ├── environment.py      # Environnement d'échecs
│   │   ├── analyzer.py         # Analyseur de positions
│   │   └── display.py          # Gestionnaire d'affichage
│   ├── gui/                    # Interface graphique (NOUVEAU!)
│   │   ├── chess_gui.py        # Interface principale
│   │   ├── board_renderer.py   # Rendu du plateau
│   │   └── piece_renderer.py   # Rendu des pièces
│   ├── exceptions/             # Exceptions personnalisées
│   │   └── __init__.py         # Gestion d'erreurs robuste
│   └── utils/                  # Utilitaires
│       ├── validation.py       # Validation des entrées
│       └── logging_config.py   # Configuration logging
├── tests/                      # Tests unitaires
├── examples/                   # Exemples d'utilisation
│   └── demo.py                 # Démonstration complète
└── docs/                       # Documentation
```

## 🚀 Installation

### Installation Complète (Interface Graphique + Core)

```bash
git clone https://github.com/your-repo/chess-ai.git
cd chess-ai

# Installation complète avec interface graphique
pip install -r requirements.txt
pip install -e .
```

### Installation Core Seulement

```bash
# Sans interface graphique (plus léger)
pip install python-chess
pip install -e .
```

### Vérification de l'Installation

```python
from chess_ai import ChessEnvironment, GUI_AVAILABLE

print(f"✅ Chess AI installé!")
print(f"🎮 Interface graphique disponible: {GUI_AVAILABLE}")

# Test rapide
env = ChessEnvironment()
env.make_move("e2e4")
print(f"🎯 Position: {env.get_board_fen()}")
```

## ⚡ Démarrage Rapide

### 🎮 Interface Graphique (Recommandé)

```python
from chess_ai import launch_gui

# Lancement direct de l'interface graphique
launch_gui()
```

### 📝 Mode Console

```python
from chess_ai import ChessEnvironment

# Créer l'environnement
env = ChessEnvironment()

# Jouer des mouvements
env.make_move("e2e4")
env.make_move("e7e5")

# Afficher le plateau
print(env.display_board())
```

### 🔍 Analyse Avancée
cd chess-ai
pip install -e .
pip install -r requirements.txt
```

## 💡 Utilisation

### Utilisation de base

```python
from chess_ai import ChessEnvironment, ChessAnalyzer, ChessDisplay
import chess

# Créer l'environnement
env = ChessEnvironment()

# Créer les composants d'analyse et d'affichage
analyzer = ChessAnalyzer(env.board)
display = ChessDisplay(env.board)

# Afficher le plateau initial
display.display_unicode()

# Effectuer des mouvements avec gestion d'erreurs
try:
    env.make_move('e2e4')
    env.make_move('e7e5')
    env.make_move('g1f3')
    print("✅ Mouvements effectués avec succès")
except InvalidMoveError as e:
    print(f"❌ Mouvement invalide: {e}")
except GameOverError as e:
    print(f"🏁 Partie terminée: {e}")

# Analyser la position
analysis = analyzer.analyze_position()
print(f"Matériel: {analysis['material']}")
print(f"Sécurité roi blanc: {analysis['white_king_safety']['safety_rating']}/10")
```

### Utilisation avancée avec logging

```python
from chess_ai.utils import setup_logging, ChessLogger

# Configuration du logging
setup_logging(level="INFO", log_file="chess_game.log")
chess_logger = ChessLogger("my_game")

# Créer l'environnement avec logging
env = ChessEnvironment(enable_logging=True)

# Logger automatique des mouvements
for move in ['e2e4', 'e7e5', 'g1f3']:
    success = env.make_move(move)
    chess_logger.log_move(move, success)
```

## 🔍 Fonctionnalités détaillées

### ChessEnvironment - Gestion du plateau

```python
env = ChessEnvironment()

# Mouvements avec validation
env.make_move('e2e4')           # Notation UCI
env.make_move(chess.Move.from_uci('e7e5'))  # Objet Move

# Informations sur le plateau
piece = env.get_piece_at('e4')  # Obtenir une pièce
moves = env.get_legal_moves()   # Mouvements légaux
stats = env.get_board_stats()   # Statistiques complètes

# Gestion de l'historique
env.undo_move()                 # Annuler le dernier mouvement
history = env.get_move_history() # Historique complet
```

### ChessAnalyzer - Analyse avancée

```python
analyzer = ChessAnalyzer(env.board)

# Analyse de matériel
material = analyzer.count_material()
print(f"Pions blancs: {material['white']['P']}")

# Sécurité du roi
safety = analyzer.get_king_safety_score(chess.WHITE)
print(f"Score de sécurité: {safety['safety_rating']}/10")

# Développement des pièces
development = analyzer.get_piece_development_score(chess.WHITE)
print(f"Développement: {development['development_percentage']:.1f}%")

# Analyse complète
analysis = analyzer.analyze_position()
```

### ChessDisplay - Affichage flexible

```python
display = ChessDisplay(env.board)

# Différents modes d'affichage
display.display_unicode()      # Affichage Unicode élégant
display.display_ascii()        # Affichage ASCII compatible
display.display_compact()      # Vue compacte avec FEN
display.display_statistics()   # Statistiques détaillées

# Personnalisation
display.display_unicode(perspective=chess.BLACK)  # Vue des noirs
display.display_move_history(env.get_move_history())  # Historique
```

## 🛡️ Gestion d'erreurs

Chess AI utilise un système d'exceptions hiérarchique pour une gestion d'erreurs robuste :

```python
from chess_ai.exceptions import *

try:
    env.make_move("invalid_move")
except InvalidMoveError as e:
    print(f"Mouvement invalide: {e.move} - {e.reason}")
except GameOverError as e:
    print(f"Partie terminée: {e.result}")
except InvalidSquareError as e:
    print(f"Case invalide: {e.square}")
except ChessError as e:
    print(f"Erreur générale: {e.message}")
```

### Types d'exceptions

- `ChessError` - Exception de base
- `InvalidMoveError` - Mouvement invalide ou illégal
- `InvalidSquareError` - Case invalide
- `GameOverError` - Action sur partie terminée
- `InvalidFENError` - Notation FEN invalide
- `ChessBoardStateError` - Erreur d'état du plateau

## 🧪 Tests

Exécuter la suite de tests complète :

```bash
# Tests unitaires
python -m pytest tests/ -v

# Tests avec couverture
python -m pytest tests/ --cov=chess_ai --cov-report=html

# Tests d'un module spécifique
python -m pytest tests/test_chess_ai.py::TestChessEnvironment -v
```

## 📖 Exemples

### Exemple complet - Partie d'échecs

```python
from chess_ai import ChessEnvironment, ChessAnalyzer, ChessDisplay
from chess_ai.utils import setup_logging

# Configuration
setup_logging(level="INFO")
env = ChessEnvironment()
display = ChessDisplay(env.board)
analyzer = ChessAnalyzer(env.board)

# Ouverture italienne
moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5']

print("🎮 Simulation d'ouverture italienne")
for i, move in enumerate(moves, 1):
    try:
        env.make_move(move)
        print(f"✅ {i}. {move}")
    except Exception as e:
        print(f"❌ {i}. {move} - Erreur: {e}")
        break

# Affichage final
display.display_unicode()

# Analyse de la position
analysis = analyzer.analyze_position()
print(f"\n📊 Analyse:")
print(f"Développement blanc: {analysis['white_development']['development_percentage']:.1f}%")
print(f"Sécurité roi blanc: {analysis['white_king_safety']['safety_rating']}/10")
```

### Exemple - Analyse de position FEN

```python
# Position complexe du milieu de partie
fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
env = ChessEnvironment(fen)
analyzer = ChessAnalyzer(env.board)

# Analyse détaillée
analysis = analyzer.analyze_position()

print("🔍 Analyse de position:")
print(f"Tour: {analysis['position_info']['turn']}")
print(f"Mouvements légaux: {analysis['position_info']['legal_moves_count']}")
print(f"En échec: {analysis['position_info']['is_check']}")

# Sécurité des rois
white_safety = analysis['white_king_safety']
black_safety = analysis['black_king_safety']

print(f"\n👑 Sécurité des rois:")
print(f"Blanc: {white_safety['safety_rating']}/10")
print(f"Noir: {black_safety['safety_rating']}/10")
```

## 🎯 Cas d'usage

Cette architecture est parfaite pour :

- 🤖 **Développement d'IA d'échecs** avec analyse robuste
- 📊 **Analyse de parties** et génération de rapports
- 🎓 **Applications éducatives** avec visualisation claire
- 🔬 **Recherche en informatique** sur les jeux d'échecs
- 🎮 **Interfaces de jeu** avec gestion d'erreurs complète
- 📈 **Outils d'analyse** pour joueurs avancés

## 🔧 Configuration avancée

### Logging personnalisé

```python
from chess_ai.utils import setup_logging, ChessLogger

# Configuration détaillée
setup_logging(
    level="DEBUG",
    log_file="chess_debug.log",
    format_string="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Logger spécialisé
logger = ChessLogger("advanced_analysis", "DEBUG")
logger.log_game_state(env.get_board_fen(), "White", env.is_check())
```

### Validation personnalisée

```python
from chess_ai.utils.validation import validate_move, validate_square

# Validation manuelle
try:
    move = validate_move("e2e4", env.board)
    square = validate_square("e4")
    print("✅ Entrées valides")
except ValueError as e:
    print(f"❌ Validation échouée: {e}")
```

## 📈 Performance

- ⚡ Utilisation optimisée de `python-chess`
- 🚀 Validation efficace des entrées
- 💾 Gestion mémoire optimisée
- 🔄 Cache intelligent pour analyses répétées

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

## 🏆 Crédits

- Construit sur la remarquable librairie [python-chess](https://github.com/niklasf/python-chess)
- Architecture inspirée des meilleures pratiques Python
- Tests unitaires exhaustifs pour fiabilité maximale

---

**Chess AI** - *Une architecture modulaire pour l'avenir des échecs digitaux* 🚀
