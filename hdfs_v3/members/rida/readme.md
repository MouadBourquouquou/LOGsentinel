# Étape 2, membre 5 : Rida

## Réseau de neurones sur graphe (GNN)

- Idée : une trace est un arbre d'étapes. Le GNN lit directement cette structure : quelles étapes sont liées,
  où se trouve l'erreur dans l'arbre, sur quelle machine.
- Données : `etapes.parquet` pour les nœuds (une étape = un nœud : `ops`, `roles`, `durees_ms`, `erreurs`) et
  `aretes.parquet` pour les liens (`aretes_par_trace()` donne directement le tableau src, dst de chaque trace ;
  type 0 = appel d'une étape vers une autre, type 1 = étapes qui se suivent dans le même fil).
- Modèle : PyTorch Geometric, deux ou trois couches GCN ou GraphSAGE, regroupement global (moyenne ou
  maximum), couche de sortie avec une probabilité par type de panne.
- Entraînement : poids de classe, arrêt anticipé sur le F1 macro de val.
- Réglage sur val : nombre de couches, taille cachée, type de regroupement, taux d'apprentissage.
- En cas de blocage technique, prévenir l'équipe tôt.

## Avant de commencer

Lire `hdfs_v3/README.md` : comment démarrer, contenu des fichiers communs, exemple de code, règles
(découpage fourni, graine 42, réglages sur val, un seul passage sur test, score avec `evaluate.py`, F1 macro).

## Ce que je rends

- un notebook par modèle dans ce dossier ;
- les prédictions sur test de chaque modèle, notées avec `evaluate.py` ;
- ce readme complété : réglages retenus, scores, temps, machine, difficultés.
