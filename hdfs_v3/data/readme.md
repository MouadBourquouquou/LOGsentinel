# Fichiers communs de l'étape 2

| Fichier | Où le trouver |
|---|---|
| `traces.parquet` | Drive, dossier `LOGsentinel donnees HDFS_v3` |
| `etapes.parquet` | Drive, dossier `LOGsentinel donnees HDFS_v3` |
| `aretes.parquet` | Drive, dossier `LOGsentinel donnees HDFS_v3` |
| `vocabulaire.json` | ici, via Git |
| `MANIFEST.txt` | ici, via Git |

Les trois fichiers .parquet se posent dans ce dossier. Leur contenu est décrit dans `hdfs_v3/README.md`,
partie « Les fichiers communs ». Leurs empreintes MD5 sont à la fin de `MANIFEST.txt` : elles doivent être
identiques chez tout le monde.

Ils sont fabriqués par `scripts/preparer_donnees.py` à partir de `data/HDFS_v3_TraceBench.zip`.
