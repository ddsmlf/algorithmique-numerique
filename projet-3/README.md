# Compression d'image à travers la factorisation SVD

## Description

Ce projet implémente un algorithme de compression d'image utilisant la décomposition en valeurs singulières (SVD). L'objectif est de réduire la taille des images tout en minimisant la perte de qualité visuelle. Le projet inclut des outils pour la manipulation des matrices, la transformation de Householder, la mise sous forme bidiagonale et la décomposition SVD.

## Structure du projet

```
.
├── Bidiagonalization.py
├── Householder.py
├── ImageCompressor.py
├── SVD.py
├── tools.py
├── plot_scripts/
│   ├── graphs.py
│   └── Plotter.py
├── tests/
│   ├── SVD_test.py
│   ├── BidiagonalDecomposition_test.py
│   ├── Householder_test.py
│   ├── ImageCompressor_test.py
│   ├── img_distance_test.py
│   └── __init__.py
├── requirements.txt
├── run_test.sh
├── README.md
└── main.tex
```

## Installation

1. Installez les dépendances :
    ```sh
    pip install -r requirements.txt
    ```

## Utilisation

### Compression d'image

Pour compresser une image, utilisez le script ImageCompressor.py. Par exemple :

```sh
python ImageCompressor.py
```

### Scripts de visualisation

Les scripts peuvent être utilisés pour générer des graphiques et analyser les performances des algorithmes de la façon suivante :

```sh
python -m plot_scripts.graphs --gpu -a visual_comparison
```
- `--gpu` : Utilise le GPU pour les calculs.
- `-a` : Spécifie l'analyse à effectuer. Les analyses disponibles sont :
    - `householder` : Compare la précision et le temps d'exécution de la transformation de Householder par rapport à la multiplication matricielle standard.
    - `qr_comparison` : Compare les performances en termes de temps et d'erreur des méthodes QR standard et bidiagonale.
    - `bidiagonal_stability` : Analyse la stabilité numérique de l'algorithme de bidiagonalisation en fonction du nombre de conditionnement.
    - `svd_convergence_speed` : Évalue la vitesse de convergence de l'algorithme SVD personnalisé par rapport à NumPy.
    - `qr_convergence` : Visualise la convergence de l'algorithme QR en traçant la norme des éléments hors diagonale de la matrice S.
    - `compression_error` : Montre l'erreur de compression d'une image en fonction du rang k pour différentes méthodes de compression.
    - `compression_time` : Compare le temps de calcul de la compression d'image en fonction de la taille de l'image pour différentes méthodes.
    - `visual_comparison` : Affiche une comparaison visuelle des images compressées pour différentes valeurs de k.

### Tests

Pour exécuter les tests unitaires, utilisez le script run_test.sh (éxécuter par défaut lors d'un push) :

```sh
./run_test.sh
```

## Contenu des fichiers

- Bidiagonalization.py : Implémente l'algorithme de mise sous forme bidiagonale.
- Householder.py : Implémente la transformation de Householder.
- ImageCompressor.py : Implémente la compression d'image utilisant la SVD.
- SVD.py : Implémente l'algorithme de décomposition SVD.
- tools.py : Contient des fonctions utilitaires pour la manipulation des matrices.
- plot_scripts : Contient des scripts pour générer des graphiques et analyser les performances des algorithmes.
- tests : Contient les tests unitaires pour les différentes parties du projet.
- requirements.txt : Liste des dépendances nécessaires pour exécuter le projet.
- run_test.sh : Script pour exécuter les tests unitaires.
- main.tex : Document LaTeX décrivant la méthodologie et les résultats du projet.

## Auteurs

- Antonin Cochon 
- Mano Domingo 
- Matheline Chevalier
- Melissa Colin
- Numa Guiot
