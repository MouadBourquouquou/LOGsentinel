# Données

## Les fichiers communs (à utiliser par tout le monde)

Télécharger depuis le Drive de l'équipe et poser dans ce dossier `data/` :

- `sequences.parquet`
- `counts.parquet`
- `templates.csv`
- `MANIFEST.txt`

Lien du Drive : (à ajouter)

Ne pas modifier ces fichiers et ne pas refaire son propre découpage.
Pour vérifier qu'on a les bons fichiers, comparer les empreintes MD5 de `MANIFEST.txt`.

## Les données brutes (seulement pour refaire les fichiers communs)

Loghub (https://github.com/logpai/loghub) renvoie vers Zenodo : https://zenodo.org/records/8196385

- `HDFS_v1.zip` (187 Mo) : à extraire dans `data/HDFS_v1/` (contient HDFS.log et preprocessed/)
- `HDFS_v3_TraceBench.zip` (567 Mo) : pour la deuxième partie du projet, dans `data/HDFS_v3_TraceBench/`

Ces dossiers sont trop gros pour Git : ils sont ignorés par `.gitignore`.

Pour refaire les fichiers communs à partir de `data/HDFS_v1/` :

```
python scripts/preparer_donnees.py
```
