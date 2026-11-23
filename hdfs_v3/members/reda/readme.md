# Étape 2, membre 6 : Reda

## Fichiers communs (avant le premier jour)

- Préparer et pousser les fichiers décrits dans `hdfs_v3/data/readme.md`, les fonctions de chargement et
  `evaluate.py` adapté au F1 macro.
- Compléter `hdfs_v3/README.md` avec le contenu réel des fichiers.

## Random Forest

- Idée : beaucoup d'arbres entraînés chacun sur une partie des données, qui votent ensemble.
- Entraînement : RandomForestClassifier, `class_weight="balanced"`.
- Réglage sur val : nombre d'arbres, profondeur maximale.
- À rendre en plus : l'importance de chaque caractéristique.

## Chaîne finale

- Un notebook qui fait tourner toute la chaîne : modèle de détection de l'étape 1 et son seuil, lien
  trace_id simulé, diagnostic de l'étape 2, résultat affiché.
- Au début, le diagnostic utilise le Random Forest comme modèle provisoire ; à la fin, le meilleur modèle
  du classement de l'étape 2 le remplace.
- Un petit test fait à la main : quelques blocs anormaux reliés à des traces connues, avec la réponse attendue.
- Ce notebook est un test de bout en bout, pas encore le moteur temps réel de l'application.

## Avant de commencer

Attendre les fichiers communs de l'étape 2 (voir `hdfs_v3/data/readme.md`). Les règles communes sont dans
`hdfs_v3/README.md` : découpage fourni, graine 42, réglages sur val, un seul passage sur test, score avec
`evaluate.py`, F1 macro comme mesure principale.

## Ce que je rends

- un notebook par modèle dans ce dossier ;
- les prédictions sur test de chaque modèle, notées avec `evaluate.py` ;
- ce readme complété : réglages retenus, scores, temps, machine, difficultés.
