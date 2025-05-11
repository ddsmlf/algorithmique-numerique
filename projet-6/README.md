# Résolution d'Équations Différentielles et Applications

Ce projet explore la résolution d'équations différentielles ordinaires (EDO) et leurs applications, notamment la modélisation de populations et le filtrage d'images. Il inclut des implémentations de solveurs numériques, des simulations, et des visualisations graphiques.

## Installation

Pour installer les dépendances nécessaires, exécutez la commande suivante dans le terminal :

```bash
pip install -r requirements.txt
```

Le fichier `requirements.txt` contient les bibliothèques Python suivantes :
- `matplotlib`: pour la visualisation des graphiques.
- `numpy`: pour les calculs numériques.
- `scipy`: pour les outils scientifiques, notamment les convolutions.
- `skimage`: pour le traitement d'images.

## Structure du Projet

- **Code principal** : Le code source implémente des solveurs numériques pour les EDO, des modèles de population, et des algorithmes de filtrage d'images.
- **Tests** : Les tests unitaires sont situés dans le dossier `tests/`. Vous pouvez exécuter tous les tests avec le script suivant :
  ```bash
  ./run_test.sh
  ```
- **Graphiques** : Les graphiques générés pour visualiser les performances des algorithmes et les résultats des simulations sont situés dans le dossier `graphs/`. Pour exécuter les scripts de visualisation :
  - Pour un algorithme spécifique, utilisez l'option `-a` suivie du nom de l'algorithme :
    ```bash
    python -m graphs.graphs -a alg_a_run
    ```
  - Pour exécuter tous les algorithmes, omettez l'option `-a` :
    ```bash
    python -m graphs.graphs
    ```

## Fonctionnalités

### 1. Résolution d'Équations Différentielles
Le projet implémente plusieurs méthodes numériques pour résoudre des EDO, notamment :
- Méthode d'Euler
- Méthode du point-milieu
- Méthode de Heun
- Méthode de Runge-Kutta d'ordre 4

Ces méthodes sont utilisées pour résoudre des problèmes de Cauchy et comparer leurs performances.

### 2. Modélisation de Populations
Deux modèles de croissance de population sont implémentés :
- **Modèle de Malthus** : croissance exponentielle.
- **Modèle de Verhulst** : croissance logistique avec une capacité limite.

Le projet inclut également une simulation du système proie-prédateur de Lotka-Volterra.

### 3. Filtrage d'Images
Deux méthodes de filtrage sont implémentées :
- **Équation de la chaleur** : pour un lissage uniforme.
- **Filtre de Perona-Malik** : pour préserver les contours tout en lissant les zones homogènes.

Les résultats des filtrages sont appliqués à l'image `camera` de `skimage`.

## Résultats
Les résultats des simulations et des visualisations sont sauvegardés dans le dossier graphs. Les graphiques incluent :
- Comparaison des méthodes numériques.
- Évolution des populations dans les modèles de Malthus, Verhulst, et Lotka-Volterra.
- Résultats des filtrages d'images.
