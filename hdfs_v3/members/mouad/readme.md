# Étape 2, membre 2 : Mouad

## One-Class SVM par type de panne

- Idée : pour chaque type de panne, une frontière autour des traces de ce type.
- Données : les caractéristiques résumées, mises à l'échelle (StandardScaler appris sur train).
- Entraînement : un One-Class SVM par type, sur les traces de train de ce type.
- Prédiction : score de chaque modèle (`decision_function`) ; la trace va au type pour lequel elle est le
  plus à l'intérieur de la frontière.
- Réglage sur val : nu et gamma. Si c'est trop lent, entraîner sur un échantillon et le noter ici.

## SVM multiclasse

- Idée : un seul SVM apprend à séparer tous les types de panne entre eux.
- Entraînement : SVC, noyau RBF, `class_weight="balanced"`.
- Réglage sur val : C et gamma.
- Intérêt : comparer la version par type (One-Class) et la version directe.

## k plus proches voisins

- Idée : une trace reçoit le type le plus fréquent parmi ses k voisines les plus proches dans train.
- Entraînement : KNeighborsClassifier sur les caractéristiques mises à l'échelle.
- Réglage sur val : k et le poids des voisins (uniforme ou selon la distance).

## Avant de commencer

Attendre les fichiers communs de l'étape 2 (voir `hdfs_v3/data/readme.md`). Les règles communes sont dans
`hdfs_v3/README.md` : découpage fourni, graine 42, réglages sur val, un seul passage sur test, score avec
`evaluate.py`, F1 macro comme mesure principale.

## Ce que je rends

- un notebook par modèle dans ce dossier ;
- les prédictions sur test de chaque modèle, notées avec `evaluate.py` ;
- ce readme complété : réglages retenus, scores, temps, machine, difficultés.
