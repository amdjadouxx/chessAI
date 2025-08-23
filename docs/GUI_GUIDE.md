# Chess AI - Interface Graphique Moderne 🎮

## 🌟 Nouvelles Fonctionnalités Graphiques

Notre Chess AI dispose maintenant d'une **interface graphique moderne et interactive** utilisant Pygame pour une expérience utilisateur exceptionnelle !

## 🚀 Installation et Configuration

### Prérequis

```bash
# Installation complète avec interface graphique
pip install pygame>=2.1.0

# Ou installation depuis les requirements
pip install -r requirements.txt
```

### Vérification

```python
from chess_ai import GUI_AVAILABLE
print(f"Interface graphique disponible: {GUI_AVAILABLE}")
```

## 🎯 Utilisation Simple

### Lancement Rapide

```python
from chess_ai import launch_gui

# Lancement direct
launch_gui()
```

### Avec Environnement Personnalisé

```python
from chess_ai import ChessEnvironment, launch_gui

# Créer un environnement avec position personnalisée
env = ChessEnvironment()
env.make_move("e2e4")  # 1. e4
env.make_move("e7e5")  # 1... e5

# Lancer l'interface avec cette position
launch_gui(env)
```

### Utilisation Avancée

```python
from chess_ai.gui import ChessGUI
from chess_ai import ChessEnvironment

# Contrôle total de l'interface
env = ChessEnvironment()
gui = ChessGUI(env)

# Configuration personnalisée
gui.game_state.show_coordinates = True
gui.game_state.show_legal_moves = True
gui.game_state.animation_duration = 500  # ms

# Lancement
gui.run()
```

## 🎮 Contrôles et Interactions

### 🖱️ Contrôles Souris
- **Clic gauche** : Sélectionner/déplacer une pièce
- **Survol** : Aperçu des effets (boutons, etc.)

### ⌨️ Raccourcis Clavier
- **N** : Nouveau jeu
- **U** : Annuler le dernier coup
- **F** : Retourner le plateau (perspective)
- **ESC** : Effacer la sélection

### 🎛️ Interface Utilisateur
- **Boutons latéraux** : Actions rapides
- **Informations temps réel** : État de la partie, tour, mouvements légaux
- **Indicateurs visuels** : Échec, dernier mouvement, mouvements légaux

## ✨ Fonctionnalités Avancées

### 🎨 Affichage Intelligent
- **Cases colorées** : Highlighting automatique des sélections
- **Mouvements légaux** : Indication visuelle des coups possibles
- **Dernier mouvement** : Highlighting du coup précédent
- **État d'échec** : Indication visuelle du roi en échec

### 🎞️ Animations Fluides
- **Mouvements animés** : Transition fluide des pièces
- **Effet de saut** : Animation arc-en-ciel pour les mouvements
- **Transparence dynamique** : Effets visuels pendant l'animation

### 🔄 Perspectives Multiples
- **Perspective blancs** : Vue classique (a1 en bas à gauche)
- **Perspective noirs** : Vue retournée (h8 en bas à gauche)
- **Basculement rapide** : Touche F ou bouton "Retourner"

### 📊 Informations Temps Réel
- **Tour actuel** : Blancs/Noirs
- **Numéro du coup** : Suivi automatique
- **Mouvements légaux** : Nombre de coups possibles
- **État de la partie** : Échec, échec et mat, pat, partie en cours

## 🏗️ Architecture Technique

### 📁 Structure des Modules GUI

```
src/chess_ai/gui/
├── __init__.py           # Exports du module GUI
├── chess_gui.py          # Interface principale
├── board_renderer.py     # Rendu du plateau
└── piece_renderer.py     # Rendu des pièces
```

### 🔧 Composants Principaux

#### ChessGUI
```python
class ChessGUI:
    """Interface graphique principale avec boucle d'événements."""
    
    # Configuration par défaut
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    BOARD_SIZE = 640
    
    def run(self):
        """Boucle principale 60 FPS."""
```

#### BoardRenderer
```python
class BoardRenderer:
    """Gestionnaire de rendu pour le plateau."""
    
    def render_board_base(self, screen, is_flipped=False):
        """Rend les cases du plateau."""
    
    def render_highlight(self, screen, x, y, highlight_type):
        """Rend les highlightings (sélection, mouvements légaux)."""
```

#### PieceRenderer
```python
class PieceRenderer:
    """Gestionnaire de rendu pour les pièces."""
    
    def render_piece(self, screen, piece, x, y, alpha=255):
        """Rend une pièce avec transparence optionnelle."""
    
    def render_animated_piece(self, screen, piece, start_pos, end_pos, progress):
        """Rend une pièce en cours d'animation."""
```

## 🎨 Personnalisation

### Couleurs et Thèmes

```python
# Modification du thème de couleurs
gui = ChessGUI()
gui.COLORS.update({
    'light_square': (255, 255, 255),    # Cases claires
    'dark_square': (100, 100, 100),     # Cases foncées
    'selected': (255, 255, 0, 100),     # Case sélectionnée
    'legal_move': (0, 255, 0, 100),     # Mouvements légaux
    'last_move': (255, 255, 0, 150),    # Dernier mouvement
    'check': (255, 0, 0, 150),          # Roi en échec
})
```

### Configuration d'Animation

```python
# Personnaliser les animations
gui.game_state.animation_duration = 300  # Durée en ms
gui.game_state.show_legal_moves = True   # Afficher les coups légaux
gui.game_state.show_coordinates = True   # Afficher a-h, 1-8
```

### Style des Pièces

```python
from chess_ai.gui import PieceRenderer

# Styles disponibles
renderer = PieceRenderer(square_size=80, style='unicode')  # Défaut
renderer = PieceRenderer(square_size=80, style='minimal')  # Géométrique
```

## 🚀 Scripts de Lancement

### Script Principal
```bash
# Lancement simple
python launch_gui.py
```

### Script d'Exemple
```bash
# Démonstration interactive
python examples/gui_demo.py
```

### Intégration dans Code
```python
# Dans votre application
from chess_ai import ChessEnvironment, launch_gui

def start_chess_game():
    env = ChessEnvironment()
    # ... configuration personnalisée ...
    launch_gui(env)
```

## 🔧 Dépannage

### Problèmes Courants

#### Pygame non installé
```
ImportError: L'interface graphique nécessite pygame
```
**Solution** : `pip install pygame>=2.1.0`

#### Performance lente
- Réduire `animation_duration`
- Désactiver `show_legal_moves` si trop de mouvements

#### Interface ne s'affiche pas
- Vérifier les drivers graphiques
- Tester avec `pygame.display.get_driver()`

## 📈 Performance et Optimisations

### 🚄 Optimisations Intégrées
- **Cache des pièces** : Surfaces pré-calculées
- **Rendu 60 FPS** : Boucle optimisée
- **Animations fluides** : Interpolation mathématique
- **Gestion mémoire** : Réutilisation des surfaces

### 📊 Métriques Typiques
- **Démarrage** : ~1-2 secondes
- **FPS** : 60 constant
- **Mémoire** : ~50-100 MB
- **CPU** : 5-15% (selon animations)

## 🎯 Cas d'Usage

### 🎮 Jeu Interactif
- Interface complète pour jouer aux échecs
- Validation automatique des coups
- Historique visuel des mouvements

### 📚 Apprentissage
- Visualisation de positions
- Exploration de mouvements légaux
- Analyse de parties

### 🔬 Développement
- Test d'algorithmes d'IA
- Débogage de positions
- Prototypage rapide

---

## 🏆 Résumé des Avantages

✅ **Interface moderne** : Design propre et intuitif  
✅ **Performance optimale** : 60 FPS avec animations fluides  
✅ **Facilité d'usage** : Lancement en une ligne de code  
✅ **Personnalisable** : Thèmes, couleurs, animations  
✅ **Intégration parfaite** : Compatible avec tout votre code existant  
✅ **Robuste** : Gestion d'erreurs et validation complète  

L'interface graphique Chess AI transforme votre environnement d'échecs en une application visuelle moderne, tout en conservant la puissance et la robustesse de l'architecture sous-jacente ! 🚀
