# Module MCTS pour AlphaZero Chess AI

## Vue d'ensemble

Le module `mcts.py` implémente l'algorithme Monte Carlo Tree Search (MCTS) optimisé pour AlphaZero appliqué aux échecs. Il fournit une recherche avancée dans l'arbre de jeu pour améliorer significativement la force de jeu de l'IA.

## Architecture

### Classes principales

#### `MCTSNode`
Représente un nœud dans l'arbre de recherche MCTS.

**Attributs clés :**
- `visit_count` (N) : Nombre de visites du nœud
- `value_sum` (W) : Somme des valeurs accumulées
- `q_value` (Q) : Valeur moyenne (W/N)
- `prior_prob` (P) : Probabilité a priori du réseau
- `children` : Dictionnaire des nœuds enfants

**Méthodes importantes :**
- `uct_score()` : Calcule le score PUCT pour la sélection
- `select_child()` : Sélectionne le meilleur enfant selon PUCT
- `expand()` : Crée les nœuds enfants avec leurs priors
- `backup()` : Remonte les valeurs dans l'arbre

#### `MCTS`
Algorithme principal Monte Carlo Tree Search.

**Méthodes principales :**
- `search(state)` : Une simulation MCTS complète
- `run(state, num_simulations)` : Multiples simulations + distribution
- `select_move(distribution, temperature)` : Sélection finale du coup

## Utilisation

### Utilisation de base

```python
from src.chess_ai.ai import ChessNet, MCTS
import chess

# Créer le réseau et MCTS
network = ChessNet()
mcts = MCTS(network, c_puct=1.4)

# Position d'échecs
board = chess.Board()

# Effectuer 400 simulations
move_distribution = mcts.run(board, num_simulations=400)

# Sélectionner le meilleur coup
best_move = mcts.select_move(move_distribution, temperature=0.0)
```

### Intégration avec l'interface GUI

```python
from src.chess_ai.gui.ai_integration import AlphaZeroPlayer

# Créer un joueur avec MCTS
ai_player = AlphaZeroPlayer(
    use_mcts=True,
    mcts_simulations=800,
    c_puct=1.4
)

# Obtenir un coup
move = ai_player.select_move(board, temperature=0.1)

# Analyse détaillée
analysis = ai_player.get_move_analysis(board)
print(f"Évaluation: {analysis['evaluation']}")
print(f"Simulations: {analysis['mcts_simulations']}")
```

### Joueur MCTS spécialisé

```python
from src.chess_ai.gui.ai_integration import MCTSAlphaZeroPlayer

# Joueur MCTS optimisé
player = MCTSAlphaZeroPlayer(
    mcts_simulations=1200,
    c_puct=1.6,
    temperature=0.8
)

move = player.get_move(board)
```

## Algorithme MCTS

### Les 4 étapes

1. **Sélection** : Navigation dans l'arbre avec la formule PUCT
   ```
   PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
   ```

2. **Expansion** : Ajout de nouveaux nœuds avec prédiction réseau

3. **Évaluation** : Appel du réseau de neurones pour valeur + politique

4. **Backpropagation** : Remontée des valeurs avec alternance des joueurs

### Formule PUCT

La sélection utilise la formule PUCT (Predictor + Upper Confidence bounds applied to Trees) :

- **Q(s,a)** : Valeur moyenne du coup
- **P(s,a)** : Probabilité a priori du réseau
- **N(s)** : Nombre de visites du parent
- **N(s,a)** : Nombre de visites de l'enfant
- **c_puct** : Constante d'exploration (typiquement 1.0-2.0)

## Paramètres de configuration

### Paramètres MCTS

| Paramètre | Description | Valeur recommandée |
|-----------|-------------|-------------------|
| `num_simulations` | Nombre de simulations par coup | 400-1600 |
| `c_puct` | Constante d'exploration PUCT | 1.0-2.0 |
| `temperature` | Contrôle stochastique (0=déterministe) | 0.0-1.0 |

### Recommandations par niveau

**Débutant/Rapide** : 
- `num_simulations=100-200`
- `c_puct=1.0`
- `temperature=0.3`

**Intermédiaire** :
- `num_simulations=400-800` 
- `c_puct=1.4`
- `temperature=0.1`

**Expert/Tournoi** :
- `num_simulations=1200-3200`
- `c_puct=1.6`
- `temperature=0.0`

## Interface avec le réseau de neurones

Le MCTS s'interface avec le réseau via deux méthodes :

### Méthode 1 : Interface directe ChessNet
```python
# Le MCTS appelle directement le réseau
mcts = MCTS(neural_network, c_puct=1.4)
```

### Méthode 2 : Interface avec predict()
```python
class CustomPredictor:
    def predict(self, board):
        # Votre logique personnalisée
        return move_probs, value

mcts = MCTS(custom_predictor, c_puct=1.4)
```

## Gestion de l'arbre

### Réutilisation de l'arbre
```python
# L'arbre persiste entre les coups pour réutiliser les calculs
mcts = MCTS(network)

# Premier coup
move_dist1 = mcts.run(board1, 400)

# L'arbre est conservé automatiquement
# Si board2 correspond à un enfant de board1, 
# les calculs précédents sont réutilisés
move_dist2 = mcts.run(board2, 400)
```

### Réinitialisation manuelle
```python
# Forcer la réinitialisation de l'arbre
mcts.reset()
```

## Statistiques et debugging

### Obtenir les statistiques détaillées
```python
stats = mcts.get_action_stats()
print(f"Visites totales: {stats['total_visits']}")
print(f"Q-value racine: {stats['q_value']}")

for move, child_stats in stats['children_stats'].items():
    print(f"{move}: {child_stats['visits']} visites, Q={child_stats['q_value']:.3f}")
```

### Analyse de la distribution des coups
```python
move_distribution = mcts.run(board, 800)

# Trier par popularité MCTS
sorted_moves = sorted(move_distribution.items(), key=lambda x: x[1], reverse=True)

for move, probability in sorted_moves[:10]:
    print(f"{move}: {probability:.4f}")
```

## Exemples complets

### Exemple 1: Analyse d'une position
```python
from src.chess_ai.ai import ChessNet, MCTS
import chess

# Configuration
network = ChessNet()
mcts = MCTS(network, c_puct=1.4)

# Position spécifique
board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")

# Analyse MCTS
print("Analyse MCTS en cours...")
move_distribution = mcts.run(board, num_simulations=600)

# Affichage des résultats
print("\nTop 5 des coups MCTS:")
sorted_moves = sorted(move_distribution.items(), key=lambda x: x[1], reverse=True)
for i, (move, prob) in enumerate(sorted_moves[:5], 1):
    print(f"{i}. {move}: {prob:.4f}")

# Sélection du coup final
best_move = mcts.select_move(move_distribution, temperature=0.0)
print(f"\nCoup sélectionné: {best_move}")
```

### Exemple 2: Partie automatique
```python
from examples.mcts_example import AlphaZeroMCTSPlayer, play_game

# Créer deux joueurs avec paramètres différents
player1 = AlphaZeroMCTSPlayer(
    num_simulations=400,
    c_puct=1.4,
    temperature=0.3
)

player2 = AlphaZeroMCTSPlayer(
    num_simulations=600,
    c_puct=1.6,
    temperature=0.1
)

# Partie automatique
result = play_game(player1, player2, max_moves=50, verbose=True)
print(f"Résultat: {result}")
```

## Performance et optimisation

### Temps de calcul typiques
- 100 simulations: ~0.5-1s
- 400 simulations: ~2-4s  
- 800 simulations: ~4-8s
- 1600 simulations: ~8-15s

*Temps mesurés sur CPU standard, GPU peut être 5-10x plus rapide*

### Conseils d'optimisation

1. **GPU** : Utiliser CUDA si disponible
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   mcts = MCTS(network, device=device)
   ```

2. **Batch processing** : Traiter plusieurs positions en parallèle
3. **Cache réseau** : Réutiliser les évaluations identiques
4. **Pruning** : Éliminer les branches peu prometteuses

### Monitoring des performances
```python
import time

start_time = time.time()
move_distribution = mcts.run(board, 800)
elapsed = time.time() - start_time

print(f"Temps MCTS: {elapsed:.2f}s")
print(f"Simulations/seconde: {800/elapsed:.1f}")
```

## Intégration dans le projet

Le module MCTS est intégré dans le projet ChessAI via :

1. **`src/chess_ai/ai/mcts.py`** : Module principal
2. **`src/chess_ai/gui/ai_integration.py`** : Interface GUI
3. **`examples/mcts_example.py`** : Exemples d'utilisation
4. **`launch_gui.py`** : Utilisation dans l'interface 3D

Pour utiliser MCTS dans l'interface graphique, modifiez la création du joueur IA :

```python
# Dans chess_gui_3d.py ou équivalent
from src.chess_ai.gui.ai_integration import AlphaZeroPlayer

# Remplacer l'IA standard par MCTS
self.ai_player = AlphaZeroPlayer(
    use_mcts=True,
    mcts_simulations=600,
    c_puct=1.4
)
```

## Troubleshooting

### Erreurs communes

**"ModuleNotFoundError: torch"**
- Solution: Installer PyTorch via `pip install torch`

**"MCTS trop lent"**
- Réduire `num_simulations`
- Utiliser GPU si disponible
- Vérifier que le réseau est en mode `.eval()`

**"Coups illégaux"**
- Vérifier la synchronisation board/MCTS
- Appeler `mcts.reset()` pour nouvelles parties

**"Memory error"**
- Réduire `num_simulations`
- Nettoyer l'arbre avec `mcts.reset()`

### Debug MCTS
```python
# Activer les logs détaillés
stats = mcts.get_action_stats()
print(f"État de l'arbre: {stats}")

# Vérifier la cohérence
assert sum(move_distribution.values()) ≈ 1.0
```
