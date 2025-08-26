# Installation de Stockfish

Pour avoir une évaluation de référence fiable, vous pouvez installer Stockfish.

## ⚡ Installation Rapide Windows

### Méthode 2 : Manuelle (Recommandée)

1. **Télécharger Stockfish :**
   - Aller sur https://stockfishchess.org/download/
   - Télécharger `stockfish-windows-x86-64-avx2.zip`

2. **Installation :**
   ```bash
   # Extraire dans C:\stockfish\
   # Le fichier doit être : C:\stockfish\stockfish.exe
   ```

3. **Test :**
   ```bash
   C:\stockfish\stockfish.exe
   # Doit lancer Stockfish en mode UCI
   ```

### Méthode 3 : Portable (dans le projet)
```bash
# Créer un dossier stockfish dans le projet
mkdir stockfish
# Placer stockfish.exe dans ce dossier
# Le programme le détectera automatiquement
```

## Linux/macOS

```bash
# Ubuntu/Debian
sudo apt install stockfish

# macOS avec Homebrew
brew install stockfish

# Arch Linux
sudo pacman -S stockfish
```

## Configuration

Le programme détecte automatiquement Stockfish dans :
- `stockfish` (dans le PATH)
- `/usr/bin/stockfish` (Linux)
- `/opt/homebrew/bin/stockfish` (macOS)
- `C:\stockfish\stockfish.exe` (Windows)

## Fallback

Si Stockfish n'est pas trouvé, l'évaluateur utilise une évaluation basique basée sur :
- Comptage du matériel
- Mobilité des pièces
- Quelques facteurs positionnels simples

Cette évaluation est moins précise mais permet de comparer avec l'IA en apprentissage.
