"""
Configuration d'installation pour Chess AI.

Package pour la modélisation d'un plateau d'échecs
avec architecture modulaire et gestion d'erreurs robuste.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lire le README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Lire les requirements
requirements = [
    "python-chess>=1.999",
    "pygame>=2.1.0",
]

setup(
    name="chess-ai",
    version="1.0.0",
    author="Amdjadouxx",
    description="Modélisation d'un plateau d'échecs avec python-chess",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "sphinx>=5.0",
        ],
        "test": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chess-ai-demo=chess_ai.examples.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "chess",
        "python-chess",
        "board-game",
        "ai",
        "analysis",
        "chess-engine",
        "chess-board",
        "chess-position",
        "game-development",
    ],
)
