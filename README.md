# PORTFOLIO SILVER INTELLIGENCE : MAINTENANCE PRÉDICTIVE HYBRIDE

Ce projet présente une solution complète de surveillance industrielle basée sur le dataset **AI4I 2020**. L'objectif est de transformer des flux de données capteurs en indicateurs décisionnels pour réduire les coûts opérationnels de maintenance.

## 1. STRUCTURE DU PROJET

Le répertoire est organisé de manière modulaire pour séparer la phase d'apprentissage de la phase de production :

* **Dossier /data** : Contient le dataset source `ai4i2020.csv`.
* **Dossier /models** : Regroupe l'ensemble des actifs exportés (Pipelines, modèles XGBoost, dictionnaires de fiabilité).
* **Fichier 01_Entrainement.py** : Script de Data Science gérant le Feature Engineering, l'équilibrage des classes et l'entraînement.
* **Fichier 02_Dashboard.py** : Interface de pilotage Streamlit pour la simulation et le monitoring.
* **Fichier requirements.txt** : Liste exhaustive des dépendances Python nécessaires.

## 2. MÉTHODOLOGIE TECHNIQUE

### Architecture du Modèle
Le système repose sur une approche hybride pour garantir la robustesse du diagnostic :

* **Détection Binaire** : Utilisation d'un **Random Forest** pour identifier la probabilité d'une panne imminente.
* **Diagnostic Multi-classe** : Un modèle **XGBoost**, optimisé par la méthode **SMOTE** (Synthetic Minority Over-sampling Technique), classifie le type de défaillance (Usure, Thermique, Puissance, Surcharge).
* **Système Expert** : Une couche de règles métier vient renforcer le Machine Learning pour les cas critiques de surcharge mécanique.

### Feature Engineering
Quatre indicateurs physiques ont été calculés pour améliorer la performance prédictive :

* **Power** : Produit du couple et de la vitesse de rotation.
* **Strain** : Intensité de l'effort cumulé (Couple x Usure).
* **Heat Stress** : Stress thermique basé sur le différentiel de température.
* **Temp Diff** : Écart entre la température du process et la température ambiante.

## 3. OPTIMISATION ÉCONOMIQUE ET ROI

Le projet intègre une dimension métier via une fonction de coût personnalisée. L'algorithme calcule le seuil de sensibilité optimal en fonction des paramètres suivants :

* **Coût d'une panne imprévue** : 500 €
* **Coût d'une intervention de maintenance prédictive** : 100 €
* **Coût d'une fausse alerte** : 50 €

Cette approche permet de maximiser le gain net par rapport à une stratégie de maintenance réactive classique.

## 4. GUIDE D'INSTALLATION ET DE LANCEMENT

### Installation des dépendances
Exécutez la commande suivante pour préparer l'environnement :  
`pip install -r requirements.txt`

### Lancement de l'application
Pour accéder au dashboard interactif, utilisez la commande :  
`python -m streamlit run 02_Dashboard.py`

## 5. ANALYSE DE FIABILITÉ

Le système inclut un module d'audit technique qui évalue le taux de détection (Recall) pour chaque paramètre capteur. Cette transparence permet aux opérateurs de connaître précisément la confiance accordée à chaque diagnostic produit par l'IA.

---------------------------------------------
**Auteur : Silver AI Project - Janvier 2026**