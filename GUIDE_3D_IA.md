# Guide d'utilisation - Interface 3D avec IA AlphaZero

## 🚀 Lancement

```bash
# Lancer l'interface 3D avec IA
python launch_gui.py --3d

# Ou lancer l'interface 2D classique
python launch_gui.py
```

## 🎮 Contrôles du jeu

### Contrôles de base
- **Clic gauche** : Sélectionner/Déplacer une pièce
- **Clic droit + glisser** : Rotation de la caméra 3D
- **Molette** : Zoom avant/arrière
- **R** : Réinitialiser la caméra

### Contrôles IA (nouveaux!)
- **H** : Toggle des suggestions IA (affiche/masque les hints)
- **I** : Jouer le meilleur coup suggéré par l'IA

## 🤖 Fonctionnalités IA

### Suggestions IA
- Appuyez sur **H** pour voir les 3 meilleurs coups suggérés par l'IA
- Les cases de destination sont surlignées en **bleu clair**
- L'intensité de la couleur indique la probabilité du coup
- Les suggestions s'affichent aussi sous forme de texte à l'écran

### Coup automatique IA
- Appuyez sur **I** pour que l'IA joue automatiquement le meilleur coup
- Parfait pour s'entraîner contre l'IA ou voir ses suggestions

### Analyse en temps réel
- L'IA analyse continuellement la position
- Affichage des pourcentages de probabilité pour chaque coup
- Évaluation de la position (avantage blanc/noir)

## 🎯 Affichage 3D

### Perspective dynamique
- Effet de profondeur pseudo-3D
- Les pièces "flottent" au-dessus du plateau
- Rotation de caméra fluide
- Zoom adaptatif

### Couleurs et thèmes
- **Cases blanches** : Beige clair
- **Cases noires** : Marron foncé
- **Sélection** : Jaune translucide
- **Mouvements possibles** : Vert translucide
- **Suggestions IA** : Bleu translucide avec intensité variable

## 🏁 États du jeu

L'interface affiche automatiquement :
- **Échec** : Alerte rouge
- **Échec et mat** : Annonce du gagnant
- **Pat** : Match nul

## 🔧 Exigences techniques

### Packages requis
```bash
pip install pygame python-chess torch torchvision numpy
```

### Configuration
- Python 3.11+ recommandé
- PyTorch CPU (pas besoin de GPU)
- 4 Go de RAM minimum pour l'IA

## 🧠 Architecture IA

### Réseau de neurones
- **Type** : CNN dual-head (AlphaZero style)
- **Entrée** : Encodage 16x8x8 du plateau
- **Sortie** : Politique (4672 coups possibles) + Évaluation
- **Modèle** : Non pré-entraîné (poids aléatoires pour démo)

### Encodage du plateau
- 12 couches pour les pièces (6 types × 2 couleurs)
- 2 couches pour le roque
- 1 couche pour la prise en passant
- 1 couche pour le tour de jeu

## 📁 Structure du projet

```
chessAI/
├── launch_gui.py           # Lanceur principal
├── src/chess_ai/
│   ├── gui/
│   │   ├── chess_gui.py       # Interface 2D
│   │   ├── chess_gui_3d.py    # Interface 3D + IA
│   │   └── ai_integration.py  # Intégration IA
│   ├── ai/
│   │   └── network.py         # Réseau AlphaZero
│   └── core/
│       └── environment.py     # Logique du jeu
└── test_gui_ia.py          # Tests des fonctionnalités IA
```

## 🎪 Démonstrations

### Test rapide des fonctionnalités
```bash
python test_gui_ia.py
```

### Démonstration IA vs Random
```bash
python -c "from src.chess_ai.gui.ai_integration import demo_ai_vs_random; demo_ai_vs_random()"
```

## 🐛 Dépannage

### IA non disponible
Si vous voyez "⚠️ Module IA non disponible", vérifiez :
```bash
python -c "import torch; print('PyTorch OK')"
```

### Problèmes d'import
Définissez le PYTHONPATH :
```bash
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path + "\src"

# Linux/Mac
export PYTHONPATH=$PWD/src
```

### Performance lente
- L'IA peut être lente sur CPU (normal)
- Réduisez la complexité en désactivant les hints (H)

## 🎊 Fonctionnalités avancées

### Personnalisation IA
- Modifiez `temperature` dans `select_move()` pour des coups plus/moins aléatoires
- Ajustez `top_k` dans `analyze_position()` pour plus/moins de suggestions

### Extension possible
- Ajout d'un modèle pré-entraîné
- Interface pour sauvegarder/charger des parties
- Mode analyse approfondie
- Intégration d'ouvertures

---

**Amusez-vous bien avec votre échiquier 3D intelligent ! 🎯🤖**
