# Étape 2, membre 4 : Abdelghani

## LSTM/GRU sur la suite des étapes

- Idée : le réseau lit les étapes de la trace dans l'ordre et donne le type de panne à la fin, comme DeepLog
  lisait les événements à l'étape 1.
- Données : `etapes.parquet` (`charger_etapes()`) : listes `ops`, `roles`, `durees_ms`, `erreurs` par trace.
  Le numéro 0 est libre dans `ops` pour le remplissage.
- Préparation : couper ou compléter les suites à une longueur fixe L, choisie d'après les traces de train.
- Modèle : embedding du type d'étape, couche GRU (ou LSTM), couche de sortie avec une probabilité par type.
- Entraînement : poids de classe, arrêt anticipé quand le F1 macro sur val ne progresse plus.
- Réglage sur val : L, taille cachée, taux d'apprentissage.

## MLP

- Idée : un réseau simple sur les caractéristiques résumées, sans l'ordre des étapes.
- Données : `traces.parquet` (`colonnes_caract()`), mises à l'échelle.
- Entraînement : MLPClassifier (scikit-learn) ou un petit réseau PyTorch.
- Intérêt : si le GRU fait nettement mieux, l'ordre des étapes aide au diagnostic ; sinon, les
  caractéristiques résumées suffisent. Les deux résultats servent au rapport.

## Avant de commencer

Lire `hdfs_v3/README.md` : comment démarrer, contenu des fichiers communs, exemple de code, règles
(découpage fourni, graine 42, réglages sur val, un seul passage sur test, score avec `evaluate.py`, F1 macro).

## Ce que je rends

- un notebook par modèle dans ce dossier ;
- les prédictions sur test de chaque modèle, notées avec `evaluate.py` ;
- ce readme complété : réglages retenus, scores, temps, machine, difficultés.
