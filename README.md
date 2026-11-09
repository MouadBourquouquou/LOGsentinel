# LOGsentinel

Projet d'intelligence artificielle, Génie Informatique 5e année, ENSA Marrakech.
Équipe : Mouad, Assia, Oumnia, Abdelghany, Reda, Rida.

LOGsentinel lit les logs d'un cluster Hadoop, apprend le comportement normal et lève des alertes expliquées.
Règle des encadrants : nous entraînons nos propres modèles (pas d'appel à une API, pas d'IA générative).

## Les étapes du projet

| Étape | Contenu | Dossier | État |
|---|---|---|---|
| 1. Choix des modèles sur HDFS_v1 | chaque membre entraîne et teste un ou deux modèles sur les mêmes données ; on compare et on garde les 2 ou 3 meilleurs | [`hdfs_v1/`](hdfs_v1/) | en cours |
| 2. Détection et diagnostic sur HDFS_v3 | même méthode sur les traces HDFS_v3 (plusieurs types de pannes) | à créer | à faire |
| 3. Application LOGsentinel | les modèles retenus intégrés dans l'outil final, avec les alertes expliquées | à créer | à faire |

## Où on en est (9 novembre 2026)

- Fait : choix des données, cahier des charges, plan de travail HDFS_v1, dépôt Git.
- Fait : outils communs de l'étape 1 (préparation des données, `evaluate.py`, `donnees.py`).
- En cours : partage des fichiers communs sur le Drive, entraînement des modèles HDFS_v1.

Le détail de l'étape 1 (règles, utilisation, répartition, classement des modèles) est dans [`hdfs_v1/README.md`](hdfs_v1/README.md).

## Organisation du dépôt

```
LOGsentinel/
  data/          données brutes (HDFS_v1, HDFS_v3_TraceBench), non suivies par Git ; voir data/readme.md
  hdfs_v1/       étape 1 : entraînement et choix des modèles sur HDFS_v1
```

## Règles de travail

- Un dossier par membre dans chaque étape (`hdfs_v1/members/<prenom>/`).
- Les gros fichiers de données passent par le Drive, pas par Git.
- Un commit = une modification avec un rôle clair, et un message qui le dit.
- Personne ne modifie les fichiers communs (`evaluate.py`, les fichiers de données) sans prévenir le groupe.
- Un point d'équipe chaque semaine ; un membre bloqué plus d'un jour le dit tout de suite.
