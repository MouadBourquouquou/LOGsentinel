# Étape 2, membre 5 : Rida

## Réseau de neurones sur graphe (GNN)

- Idée : une trace est un arbre d'étapes. Le GNN lit directement cette structure : quelles étapes sont liées,
  où se trouve l'erreur dans l'arbre, sur quelle machine.
- Données : les traces sous forme de graphes, fournies toutes prêtes dans les fichiers communs (nœuds avec
  type d'étape, durée et statut ; arêtes entre chaque étape et son parent).
- Modèle : PyTorch Geometric, deux ou trois couches GCN ou GraphSAGE, regroupement global (moyenne ou
  maximum), couche de sortie avec une probabilité par type de panne.
- Entraînement : poids de classe, arrêt anticipé sur le F1 macro de val.
- Réglage sur val : nombre de couches, taille cachée, type de regroupement, taux d'apprentissage.
- En cas de blocage technique, prévenir l'équipe tôt.

## Avant de commencer

Attendre les fichiers communs de l'étape 2 (voir `hdfs_v3/data/readme.md`). Les règles communes sont dans
`hdfs_v3/README.md` : découpage fourni, graine 42, réglages sur val, un seul passage sur test, score avec
`evaluate.py`, F1 macro comme mesure principale.

## Ce que je rends

- un notebook par modèle dans ce dossier ;
- les prédictions sur test de chaque modèle, notées avec `evaluate.py` ;
- ce readme complété : réglages retenus, scores, temps, machine, difficultés.
