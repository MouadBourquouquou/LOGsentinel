# Étape 2, membre 1 : Assia

## PCA par type de panne

- Idée : un PCA appris sur les traces de chaque type de panne. Chaque PCA reconstruit bien les traces de
  son type et mal les autres.
- Données : les caractéristiques résumées des traces.
- Entraînement : StandardScaler appris sur train, puis un PCA par type, sur les traces de train de ce type.
- Prédiction : erreur de reconstruction de la trace avec chaque PCA ; la trace va au type dont l'erreur est
  la plus petite.
- Réglage sur val : nombre de composantes (par exemple 90 %, 95 % ou 99 % de la variance gardée).

## XGBoost

- Idée : des arbres construits l'un après l'autre, chacun corrige les erreurs du précédent.
- Entraînement : XGBClassifier multiclasse, avec des poids pour les types rares.
- Réglage sur val : profondeur des arbres, nombre d'arbres, taux d'apprentissage.
- À rendre en plus : l'importance des caractéristiques (à comparer avec le Random Forest de Reda).

## Avant de commencer

Attendre les fichiers communs de l'étape 2 (voir `hdfs_v3/data/readme.md`). Les règles communes sont dans
`hdfs_v3/README.md` : découpage fourni, graine 42, réglages sur val, un seul passage sur test, score avec
`evaluate.py`, F1 macro comme mesure principale.

## Ce que je rends

- un notebook par modèle dans ce dossier ;
- les prédictions sur test de chaque modèle, notées avec `evaluate.py` ;
- ce readme complété : réglages retenus, scores, temps, machine, difficultés.
