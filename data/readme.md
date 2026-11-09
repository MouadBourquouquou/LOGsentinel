# Données brutes

Ce dossier contient les données brutes de Loghub. Elles sont trop grosses pour Git : chacun les télécharge
s'il en a besoin. Pour l'étape 1, la plupart des membres n'en ont pas besoin : les fichiers communs prêts à
l'emploi sont décrits dans `hdfs_v1/README.md`.

Loghub (https://github.com/logpai/loghub) renvoie vers Zenodo : https://zenodo.org/records/8196385

| Fichier sur Zenodo | Taille | À extraire dans | Sert à |
|---|---|---|---|
| `HDFS_v1.zip` | 187 Mo | `data/HDFS_v1/` (contient HDFS.log et preprocessed/) | étape 1, fabriquer les fichiers communs |
| `HDFS_v3_TraceBench.zip` | 567 Mo | `data/HDFS_v3_TraceBench/` | étape 2 |

## Fichiers communs de l'étape 1

`sequences.parquet` et `counts.parquet` sont sur le Drive de l'équipe : (lien à ajouter)
Ils se posent dans `hdfs_v1/data/`.
