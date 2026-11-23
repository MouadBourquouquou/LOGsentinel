# Étape 2, membre 3 : Oumnia

## Invariant Mining par type de panne

- Idée : trouver les règles toujours vraies pour un type de panne (par exemple « autant de réceptions que
  d'envois »). Le code de l'étape 1 (`hdfs_v1/algorithmes/invariant_mining.py`) peut être repris.
- Données : les comptages des types d'étapes par trace.
- Entraînement : pour chaque type, les invariants respectés par les traces de train de ce type.
- Prédiction : la trace va au type dont elle respecte le mieux les règles.
- Intérêt : le modèle le plus explicable ; ses règles pourront servir au texte des alertes.

## Régression logistique

- Idée : un modèle linéaire simple qui donne une probabilité par type de panne.
- Entraînement : LogisticRegression multiclasse, `class_weight="balanced"`, caractéristiques mises à l'échelle.
- Réglage sur val : C.

## Arbre de décision

- Idée : une suite de questions oui ou non, lisible par un humain.
- Entraînement : DecisionTreeClassifier, `class_weight="balanced"`.
- Réglage sur val : profondeur maximale, nombre minimal de traces par feuille.
- À rendre en plus : l'arbre exporté en texte (`export_text`), pour le rapport.

## Avant de commencer

Attendre les fichiers communs de l'étape 2 (voir `hdfs_v3/data/readme.md`). Les règles communes sont dans
`hdfs_v3/README.md` : découpage fourni, graine 42, réglages sur val, un seul passage sur test, score avec
`evaluate.py`, F1 macro comme mesure principale.

## Ce que je rends

- un notebook par modèle dans ce dossier ;
- les prédictions sur test de chaque modèle, notées avec `evaluate.py` ;
- ce readme complété : réglages retenus, scores, temps, machine, difficultés.
