# PyTorch 1D Playground

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

Une application web interactive pour visualiser et comprendre comment les réseaux de neurones apprennent à approximer des fonctions mathématiques complexes et des distributions statistiques en 1D.


## Objectif du projet

Ce projet a été conçu pour servir de bac à sable (playground) visuel. Il permet de :
1. **Générer des données** basées sur des fonctions (sin, cos, exp...) ou des lois statistiques (Normale, Uniforme, Cauchy...).
2. **Configurer dynamiquement** l'architecture d'un Perceptron Multicouche (MLP) via PyTorch (couches, neurones, fonction d'activation, learning rate).
3. **Visualiser l'apprentissage** du modèle pour comprendre concrètement les concepts d'underfitting/overfitting et l'impact des hyperparamètres.

## Fonctionnalités
- **Mode Mathématique** : Régression non linéaire sur des fonctions trigonométriques et logarithmiques.
- **Mode Statistique** : Estimation de densité (KDE) pour recréer des lois de probabilités.
- **Paramétrage en temps réel** : Modification de l'architecture du réseau de neurones à la volée.
- **Dataviz avancée** : Graphiques générés avec Plotly (Histogrammes, courbes de loss, QQ-Plots).

## 🛠️ Installation et Exécution en local

1. Clonez ce dépôt :
```bash
git clone [https://github.com/votre-nom/pytorch-1d-playground.git](https://github.com/votre-nom/pytorch-1d-playground.git)
cd pytorch-1d-playground
