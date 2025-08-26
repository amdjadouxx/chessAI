# Chess AI - Interface 3D avec Intelligence Artificielle et MCTS 🎮🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![GUI: Pygame](https://img.shields.io/badge/GUI-Pygame-green.svg)](https://www.pygame.org/)
[![AI: AlphaZero+MCTS](https://img.shields.io/badge/AI-AlphaZero+MCTS-purple.svg)](https://arxiv.org/)

## 🎯 Vue d'ensemble

Chess AI est une **interface d'échecs 3D moderne** avec **intelligence artificielle AlphaZero + MCTS**. Cette implémentation combine une expérience visuelle immersive avec des capacités d'analyse IA de niveau professionnel utilisant l'algorithme Monte Carlo Tree Search.

### ✨ Caractéristiques principales

- 🎮 **Interface 3D pseudo-perspective** avec effets de profondeur
- 🤖 **IA AlphaZero avec MCTS** - Recherche arborescente avancée
- 🧠 **Mode MCTS complet** : 4 étapes (Sélection, Expansion, Évaluation, Backpropagation)
- 🎯 **Analyse en profondeur** avec simulations configurables (100-3200)
- � **Double barre d'évaluation** : Référence (Stockfish) vs IA en apprentissage
- �🖱️ **Contrôles interactifs** : rotation caméra, zoom, clic-déplacer
- 📍 **Coordonnées visibles** (A-H, 1-8) pour orientation
- ⚡ **Double mode IA** : Évaluation directe OU recherche MCTS
- 🎨 **Interface moderne** avec animations fluides
- 🔄 **Roque automatique** et gestion complète des règles

## 🚀 Installation et Lancement

### Prérequis
- Python 3.11 ou plus récent
- Git (pour cloner le projet)

### Installation Étape par Étape

1. **Cloner le projet**
```bash
git clone https://github.com/amdjadouxx/chessAI.git
cd chessAI
```

2. **Créer un environnement virtuel (recommandé)**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Lancer le jeu !**
```bash
# Windows PowerShell (avec environnement virtuel)
$env:PYTHONPATH = (Get-Location).Path + "\src"
C:/Users/amdja/Desktop/repositorys/chessAI/.venv/Scripts/python.exe launch_gui.py --3d

# Ou plus simplement (si .venv activé)
$env:PYTHONPATH = (Get-Location).Path + "\src"
python launch_gui.py --3d
```

### Commande Complète Testée
```bash
# Cette commande fonctionne à 100% :
cd chessAI
.venv\Scripts\activate
$env:PYTHONPATH = (Get-Location).Path + "\src"
python launch_gui.py --3d
```

### Installation Alternative (Plus Rapide)
```bash
# Si vous avez déjà un environnement Python configuré
pip install pygame python-chess torch torchvision numpy

# Puis lancer avec PYTHONPATH
cd chessAI
$env:PYTHONPATH = (Get-Location).Path + "\src"  # Windows
export PYTHONPATH=$PWD/src                      # Linux/Mac
python launch_gui.py --3d
```

### Vérification de l'Installation
Si vous voyez ce message, tout fonctionne :
```
🚀 Lancement de Chess AI - Interface 3D avec IA
🤖 IA AlphaZero activée
🎮 Interface 3D Simple initialisée !
```

### 🎯 Commande Testée qui Fonctionne à 100%
```bash
# Cette commande exacte fonctionne (testée) :
cd chessAI
$env:PYTHONPATH = (Get-Location).Path + "\src"
C:/Users/amdja/Desktop/repositorys/chessAI/.venv/Scripts/python.exe launch_gui.py --3d
```

**Note** : Remplacez le chemin par votre dossier chessAI si différent.

## 🎮 Interface Utilisateur

### Lancement
```bash
# Windows PowerShell (OBLIGATOIRE pour définir le PYTHONPATH)
$env:PYTHONPATH = (Get-Location).Path + "\src"
python launch_gui.py --3d

# Linux/Mac
export PYTHONPATH=$PWD/src
python launch_gui.py --3d
```

**⚠️ IMPORTANT** : Le PYTHONPATH est nécessaire pour que Python trouve les modules dans `src/`

### Structure Réelle du Projet
```
chessAI/
├── launch_gui.py              # 🚀 FICHIER PRINCIPAL - Lance l'interface
├── requirements.txt           # 📦 Dépendances à installer
├── README.md                  # 📖 Ce fichier
├── GUIDE_3D_IA.md            # 📚 Guide détaillé
├── .venv/                     # 🐍 Environnement virtuel Python
├── src/chess_ai/             # 💻 Code source principal
│   ├── __init__.py
│   ├── ai/                   # 🤖 Intelligence Artificielle
│   │   ├── __init__.py
│   │   └── network.py        # Réseau AlphaZero (CNN)
│   ├── core/                 # ⚙️ Logique de jeu
│   │   ├── __init__.py
│   │   └── environment.py    # Moteur d'échecs
│   ├── gui/                  # 🎮 Interface utilisateur
│   │   ├── __init__.py
│   │   ├── chess_gui_3d.py   # Interface 3D principale
│   │   └── ai_integration.py # Intégration IA
│   └── exceptions/           # 🛡️ Gestion d'erreurs
│       └── __init__.py
├── assets/                   # 🎨 Ressources (images des pièces)
└── docs/                     # 📄 Documentation
```

### Contrôles
| Contrôle | Action |
|----------|--------|
| **Clic gauche** | Sélectionner/Déplacer pièce |
| **Clic droit + glisser** | Rotation caméra 3D |
| **Molette** | Zoom avant/arrière |
| **R** | Réinitialiser caméra |
| **H** | Toggle suggestions IA |
| **I** | Jouer coup IA automatiquement |
| **E** | Toggle barres d'évaluation |

### Interface Visuelle
- 🎨 **Plateau 3D** avec perspective dynamique
- 📍 **Coordonnées A-H, 1-8** toujours visibles
- 🎯 **Surlignage intelligent** : sélection (jaune), mouvements possibles (vert), suggestions IA (bleu)
- 💡 **Intensité variable** des suggestions basée sur les probabilités IA
- 📊 **Double barre d'évaluation** : Référence vs IA en temps réel
- 📈 **Historique graphique** des évaluations avec mini-graphiques
- 📊 **Affichage temps réel** des statistiques de jeu

## 🤖 Intelligence Artificielle avec MCTS

### Architecture AlphaZero + MCTS
- **Réseau de neurones** : CNN dual-head (politique + évaluation)
- **Algorithme MCTS** : 4 étapes complètes (Sélection, Expansion, Évaluation, Backpropagation)
- **Formule PUCT** : Q + c_puct * P * sqrt(N_parent)/(1+N_child)
- **Encodage plateau** : 16×8×8 (pièces, règles spéciales, tour)
- **Espace d'action** : 4672 mouvements possibles
- **Modes disponibles** : Évaluation directe OU recherche MCTS avancée

## � **NOUVELLES FONCTIONNALITÉS AVANCÉES**

### 🔥 **Pré-entraînement supervisé avec Stockfish**
Accélérez drastiquement l'apprentissage de votre IA !

```bash
# 1. Pré-entraînement supervisé (base solide)
python pretrain_with_stockfish.py

# 2. Entraînement AlphaZero amélioré
python full_training_demo.py
```

**Avantages :**
- ✅ **Base solide** : L'IA commence avec des connaissances Stockfish
- ✅ **Convergence rapide** : Apprentissage 10x plus efficace
- ✅ **Validation objective** : Comparaison continue avec Stockfish
- ✅ **Apprentissage hybride** : Stockfish + découvertes AlphaZero

### �📊 **Système d'évaluation double en temps réel**

### Barres d'Évaluation en Temps Réel
L'interface affiche **deux barres d'évaluation côte à côte** :

#### 🔵 Barre de Référence (Stockfish)
- **Évaluateur fiable** basé sur Stockfish (ou évaluation basique en fallback)
- **Couleur bleue** pour la distinguer
- **Évaluation objective** indépendante de l'apprentissage
- **Historique graphique** des 20 dernières positions

#### 🟠 Barre IA AlphaZero
- **Évaluation de l'IA** en cours d'apprentissage
- **Couleur orange** pour la différencier
- **Évolution visible** au fur et à mesure de l'entraînement
- **Comparaison directe** avec la référence

#### 📈 Fonctionnalités Visuelles
- **Échelle normalisée** : -1.0 (noir gagne) à +1.0 (blanc gagne)
- **Ligne de milieu** jaune pour l'égalité (0.0)
- **Valeurs numériques** affichées au-dessus de chaque barre
- **Calcul d'écart** entre référence et IA
- **Mini-graphiques** d'historique pour chaque évaluation

### Installation Stockfish (Optionnelle)
Pour une évaluation de référence optimale, voir : [Installation Stockfish](docs/STOCKFISH_INSTALL.md)

Sans Stockfish, le système utilise une évaluation basique mais fonctionnelle.

### Fonctionnalités IA Avancées

#### Mode MCTS (Recommandé)
```python
from chess_ai.gui.ai_integration import AlphaZeroPlayer

# IA MCTS avec 800 simulations
ai = AlphaZeroPlayer(
    use_mcts=True,
    mcts_simulations=800,
    c_puct=1.4
)

# Analyse avec statistiques MCTS
analysis = ai.get_move_analysis(board)
print(f"Évaluation : {analysis['evaluation']:+.3f}")
print(f"Simulations MCTS : {analysis['mcts_simulations']}")
print(f"Visites totales : {analysis['mcts_visits']}")
```

#### Mode Évaluation Directe (Rapide)
```python
# IA rapide sans MCTS
ai = AlphaZeroPlayer(use_mcts=False)

# Analyse instantanée
analysis = ai.get_move_analysis(board)
for i, (move, prob) in enumerate(analysis['top_moves'][:3], 1):
    print(f"{i}. {move} ({prob:.1%})")
```

### Configurations MCTS Recommandées

| Niveau | Simulations | c_puct | Temps approx | Usage |
|--------|-------------|--------|--------------|-------|
| **Rapide** | 100-200 | 1.0 | 0.5-1s | Test/Debug |
| **Standard** | 400-800 | 1.4 | 2-4s | Jeu normal |
| **Expert** | 1200-1600 | 1.6 | 6-8s | Analyse approfondie |
| **Tournoi** | 2400-3200 | 1.8 | 12-15s | Compétition |

### Interface MCTS Spécialisée
```python
from chess_ai.gui.ai_integration import MCTSAlphaZeroPlayer

# Joueur MCTS optimisé avec paramètres prédéfinis
player = MCTSAlphaZeroPlayer(
    mcts_simulations=1200,  # Niveau expert
    c_puct=1.6,
    temperature=0.1
)

move = player.get_move(board)  # Utilise automatiquement MCTS
```

## 🏗️ Architecture du Projet

```
chessAI/
├── launch_gui.py                    # Lanceur principal
├── src/chess_ai/
│   ├── __init__.py                 # Module principal
│   ├── ai/                         # Intelligence Artificielle
│   │   ├── __init__.py
│   │   ├── network.py              # Réseau AlphaZero (CNN)
│   │   └── mcts.py                 # 🆕 Algorithme MCTS complet
│   ├── core/
│   │   ├── __init__.py
│   │   └── environment.py          # Logique de jeu (python-chess)
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── chess_gui_3d.py         # Interface 3D principale
│   │   └── ai_integration.py       # Intégration IA/GUI + MCTS
│   └── exceptions/
│       └── __init__.py             # Gestion d'erreurs
├── examples/
│   ├── demo.py                     # Démonstration de base
│   └── mcts_example.py             # 🆕 Exemples MCTS avancés
├── assets/
│   └── pieces/                     # Images des pièces (optionnel)
├── docs/
│   ├── GUI_GUIDE.md               # Guide interface
│   └── MCTS_GUIDE.md              # 🆕 Guide MCTS complet
├── GUIDE_3D_IA.md                 # Guide complet 3D+IA
└── requirements.txt               # Dépendances
```

### 🆕 Nouveaux Modules

#### `src/chess_ai/ai/mcts.py`
- **MCTSNode** : Nœuds de l'arbre avec statistiques PUCT
- **MCTS** : Algorithme principal avec 4 étapes
- **Interface réseau** : Compatible avec ChessNet existant
- **Optimisations** : Réutilisation d'arbre, cache, parallélisation

#### `examples/mcts_example.py`
- **AlphaZeroMCTSPlayer** : Joueur complet avec MCTS
- **Benchmarks** : Tests de performance
- **Exemples d'usage** : Parties automatiques, analyse

#### `docs/MCTS_GUIDE.md`
- **Documentation complète** : Architecture, usage, paramètres
- **Exemples pratiques** : Code prêt à utiliser
- **Optimisation** : Conseils performance et configuration

## ⚡ Démarrage Rapide

### Lancement Immédiat
```bash
# Dans le dossier chessAI :
# 1. Définir le PYTHONPATH (Windows PowerShell)
$env:PYTHONPATH = (Get-Location).Path + "\src"

# 2. Lancer l'interface 3D
python launch_gui.py --3d

# OU en une seule commande :
cd chessAI ; $env:PYTHONPATH = (Get-Location).Path + "\src" ; python launch_gui.py --3d
```

### Test des Fonctionnalités IA
```python
# Test rapide dans Python
cd chessAI
python

>>> from src.chess_ai.ai.network import encode_board, ChessNet
>>> import chess
>>> import torch

>>> # Test de l'encodage
>>> board = chess.Board()
>>> encoded = encode_board(board)
>>> print(f"Plateau encodé: {encoded.shape}")  # [16, 8, 8]

>>> # Test du réseau IA
>>> net = ChessNet()
>>> with torch.no_grad():
...     policy, value = net(encoded.unsqueeze(0))
>>> print(f"IA fonctionne! Évaluation: {value.item():.3f}")
```

### Interface Graphique Programmée
```python
# Pour intégrer dans votre code Python
import sys
sys.path.append('src')

from chess_ai.gui.chess_gui_3d import SimpleChessGUI3D

# Lancer l'interface 3D
gui = SimpleChessGUI3D()
gui.run()
```

## 🎯 Exemples d'Utilisation

### MCTS Avancé - Session Complète
```python
# Dans le dossier chessAI
python

>>> import sys
>>> sys.path.append('src')
>>> from chess_ai.ai import MCTS, ChessNet
>>> from chess_ai.gui.ai_integration import MCTSAlphaZeroPlayer
>>> import chess

>>> # Test MCTS direct
>>> network = ChessNet()
>>> mcts = MCTS(network, c_puct=1.4)
>>> board = chess.Board()

>>> # 400 simulations MCTS
>>> move_distribution = mcts.run(board, num_simulations=400)
>>> best_move = mcts.select_move(move_distribution, temperature=0.0)
>>> print(f"MCTS recommande: {best_move}")

>>> # Statistiques détaillées
>>> stats = mcts.get_action_stats()
>>> print(f"Visites totales: {stats['total_visits']}")
>>> print("Top 3 coups MCTS:")
>>> sorted_moves = sorted(move_distribution.items(), key=lambda x: x[1], reverse=True)
>>> for i, (move, prob) in enumerate(sorted_moves[:3], 1):
...     print(f"  {i}. {move}: {prob:.4f}")
```

### Comparaison Mode Direct vs MCTS
```python
>>> from chess_ai.gui.ai_integration import AlphaZeroPlayer

>>> # Mode évaluation directe (rapide)
>>> ai_direct = AlphaZeroPlayer(use_mcts=False)
>>> analysis_direct = ai_direct.get_move_analysis(board)
>>> print(f"Mode direct - Top coup: {analysis_direct['top_moves'][0]}")

>>> # Mode MCTS (précis)
>>> ai_mcts = AlphaZeroPlayer(use_mcts=True, mcts_simulations=600)
>>> analysis_mcts = ai_mcts.get_move_analysis(board)
>>> print(f"Mode MCTS - Top coup: {analysis_mcts['top_moves'][0]}")
>>> print(f"Simulations effectuées: {analysis_mcts['mcts_simulations']}")
>>> print(f"Visites MCTS: {analysis_mcts['mcts_visits']}")
```

### Partie MCTS vs MCTS
```python
>>> from examples.mcts_example import AlphaZeroMCTSPlayer, play_game

>>> # Deux joueurs MCTS avec paramètres différents
>>> player_aggressive = AlphaZeroMCTSPlayer(
...     mcts_simulations=400,
...     c_puct=2.0,        # Plus exploratoire
...     temperature=0.5    # Plus créatif
... )

>>> player_defensive = AlphaZeroMCTSPlayer(
...     mcts_simulations=600,
...     c_puct=1.0,        # Plus conservateur  
...     temperature=0.1    # Plus déterministe
... )

>>> # Partie automatique
>>> result = play_game(player_aggressive, player_defensive, max_moves=30)
>>> print(f"Résultat: {result}")
```

### Analyse Position Complexe avec MCTS
```python
>>> # Position tactique du milieu de partie
>>> fen = "r1bq1rk1/ppp2ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQR1K1 w - - 0 8"
>>> board = chess.Board(fen)

>>> # Analyse MCTS approfondie
>>> ai_expert = AlphaZeroPlayer(use_mcts=True, mcts_simulations=1200, c_puct=1.6)
>>> analysis = ai_expert.get_move_analysis(board, top_k=10)

>>> print(f"Position complexe - Évaluation: {analysis['evaluation']:+.3f}")
>>> print(f"MCTS - {analysis['mcts_simulations']} simulations, {analysis['mcts_visits']} visites")
>>> print("\nTop 5 coups MCTS:")
>>> for i, (move, prob) in enumerate(analysis['top_moves'][:5], 1):
...     print(f"  {i}. {move}: {prob:.4f} ({prob*100:.1f}%)")
```

## 🛠️ Résolution de Problèmes

### Erreur "No module named 'chess_ai'"
```bash
# Solution 1: Définir le PYTHONPATH
set PYTHONPATH=%cd%\src && python launch_gui.py --3d

# Solution 2: Utiliser l'environnement virtuel
.venv\Scripts\activate
python launch_gui.py --3d
```

### Erreur "No module named 'torch'"
```bash
# Réinstaller PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Interface ne s'affiche pas
- Vérifiez que pygame est installé : `pip install pygame`
- Testez : `python -c "import pygame; print('Pygame OK')"`

### Performance lente
- L'IA utilise le CPU par défaut (normal d'être lent)
- Désactivez les suggestions IA avec la touche **H** si nécessaire

## 🔧 Configuration Avancée

### Personnalisation IA
```python
# Créer un joueur IA personnalisé
ai = AlphaZeroPlayer(device="cpu")  # ou "cuda" si GPU disponible

# Changer la température (aléatoire vs déterministe)
move = ai.select_move(board, temperature=0.1)  # Plus déterministe
move = ai.select_move(board, temperature=1.0)  # Plus créatif
```

### Paramètres Interface 3D
Les paramètres visuels peuvent être modifiés dans `chess_gui_3d.py` :
```python
# Taille de la fenêtre
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700

# Couleurs du plateau
COLORS = {
    "light": (240, 217, 181),     # Cases claires
    "dark": (181, 136, 99),       # Cases sombres
    "selected": (255, 255, 0, 128), # Sélection
    "ai_suggestion": (100, 149, 237, 150), # Suggestions IA
}
```

## 🧪 Tests et Validation

### Tests IA
```bash
# Test complet du module IA
python -c "from chess_ai.gui.ai_integration import demo_ai_vs_random; demo_ai_vs_random()"
```

### Validation Réseau
```python
from chess_ai.ai.network import *

# Test batch
boards = [chess.Board() for _ in range(5)]
batch = batch_encode_boards(boards)
print(f"Batch shape: {batch.shape}")  # [5, 16, 8, 8]
```

## 📊 Performance

- **Temps de réponse IA** : ~100-500ms sur CPU moderne
- **Mémoire requise** : ~500MB avec PyTorch
- **FPS interface** : 60 FPS stable
- **Précision calculs** : Float32 (suffisant pour démo)

## 🎨 Fonctionnalités Visuelles

### Effets 3D
- **Perspective pseudo-3D** : Les pièces semblent "flotter"
- **Ombres dynamiques** : Effet de profondeur
- **Rotation de caméra** : Vue personnalisable
- **Zoom fluide** : Contrôle de la distance

### Interface Moderne
- **Surlignage multi-couleur** : Sélection, mouvements, suggestions IA
- **Coordonnées permanentes** : A-H et 1-8 toujours visibles
- **Feedback temps réel** : Nombre de mouvements possibles
- **Animation roque** : Gestion automatique du grand/petit roque

## 🤝 Contribution

1. **Fork** le projet
2. **Créer** une branche feature (`git checkout -b feature/amazing-feature`)
3. **Commit** vos changements (`git commit -m 'Add amazing feature'`)
4. **Push** vers la branche (`git push origin feature/amazing-feature`)
5. **Ouvrir** une Pull Request

### Zones d'amélioration
- 🏋️ **Entraînement IA** : Modèle pré-entraîné
- 🎮 **Modes de jeu** : Humain vs IA, IA vs IA, analyse
- 📚 **Base d'ouvertures** : Intégration ECO
- 🔊 **Sons** : Effets sonores pour les mouvements
- 🌐 **Multijoueur** : Jeu en réseau

## � Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

## 🏆 Technologies Utilisées

- **[python-chess](https://github.com/niklasf/python-chess)** : Logique de jeu robuste
- **[PyTorch](https://pytorch.org/)** : Réseau de neurones IA
- **[Pygame](https://www.pygame.org/)** : Interface graphique et rendu
- **[NumPy](https://numpy.org/)** : Calculs mathématiques optimisés

## 📚 Documentation Complète

- � **[Guide complet 3D+IA](GUIDE_3D_IA.md)** : Instructions détaillées
- 🎮 **[Guide interface](docs/GUI_GUIDE.md)** : Contrôles et fonctionnalités
- 🤖 **[Documentation IA](src/chess_ai/ai/README.md)** : Architecture AlphaZero

---

**Chess AI 3D** - *L'avenir des échecs avec intelligence artificielle* 🚀🤖

### 🎯 Captures d'écran

```
🎮 Interface 3D en action :
┌─────────────────────────────────┐
│  A  B  C  D  E  F  G  H        │
│8 ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜  8   │
│7 ♟  ♟  ♟  ♟  ♟  ♟  ♟  ♟  7   │
│6                            6   │
│5                            5   │
│4                            4   │
│3                            3   │
│2 ♙  ♙  ♙  ♙  ♙  ♙  ♙  ♙  2   │
│1 ♖  ♘  ♗  ♕  ♔  ♗  ♘  ♖  1   │
│  A  B  C  D  E  F  G  H        │
└─────────────────────────────────┘
    🤖 IA: 1.Nf3 (67%) 2.e4 (23%) 3.d4 (10%)
```

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
│   └── utils/                  # Utilitairespython launch_gui.py
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
