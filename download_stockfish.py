#!/usr/bin/env python3
"""
Script d'installation automatique de Stockfish
==============================================
"""

import os
import zipfile
import urllib.request
import sys
import shutil


def download_stockfish():
    """Télécharge et installe Stockfish automatiquement."""

    print("🔽 Téléchargement de Stockfish...")

    # URL de Stockfish Windows
    url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip"
    zip_file = "stockfish.zip"
    temp_dir = "stockfish_temp"
    install_dir = "stockfish"

    try:
        # Téléchargement
        print("📥 Téléchargement depuis GitHub...")
        urllib.request.urlretrieve(url, zip_file)
        print("✅ Téléchargement terminé")

        # Extraction
        print("📦 Extraction...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        print("✅ Extraction terminée")

        # Installation
        print("💾 Installation...")
        if not os.path.exists(install_dir):
            os.makedirs(install_dir)

        # Trouver l'exécutable
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".exe") and "stockfish" in file.lower():
                    src = os.path.join(root, file)
                    dst = os.path.join(install_dir, "stockfish.exe")
                    shutil.copy2(src, dst)
                    print(f"✅ Stockfish installé : {dst}")
                    break

        # Test
        stockfish_path = os.path.join(install_dir, "stockfish.exe")
        if os.path.exists(stockfish_path):
            print("🧪 Test de l'installation...")
            print(f"📍 Stockfish disponible dans : {os.path.abspath(stockfish_path)}")
            print("✅ Installation réussie !")
        else:
            print("❌ Erreur : fichier non créé")
            return False

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False
    finally:
        # Nettoyage
        print("🧹 Nettoyage...")
        if os.path.exists(zip_file):
            os.remove(zip_file)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    print("")
    print("🎮 Redémarrez Chess AI pour utiliser Stockfish !")
    return True


if __name__ == "__main__":
    download_stockfish()
