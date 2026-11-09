# Étape 1 : choix des modèles sur HDFS_v1

Chaque membre entraîne et teste un ou deux modèles avec les mêmes données, le même découpage et le même
script de score. On compare tout dans un seul tableau, puis on garde les 2 ou 3 meilleurs pour LOGsentinel.

La question posée à chaque modèle : « ce bloc HDFS se comporte-t-il normalement ? ». Toutes les lignes de log
qui citent le même identifiant `blk_...` forment un bloc, l'unité que chaque modèle juge.

## Classement actuel

F1 sur la partie test, calculé uniquement par `evaluate.py`. Mis à jour avec `python scripts/classement.py`.

| Rang | Membre | Modèle | Précision | Rappel | F1 | Temps d'entraînement (min) |
|---|---|---|---|---|---|---|
| | | aucun modèle noté pour l'instant | | | | |

### Répartition et repères publiés

| # | Membre | Modèle(s) | Fichier lu | Repère publié (F1) | État |
|---|---|---|---|---|---|
| 1 | Mouad | Comptage des ERROR (base) | counts | pas de repère | à faire |
| 1 | Mouad | PCA | counts | 0,769 | à faire |
| 2 | Assia | Isolation Forest | counts | 0,802 | à faire |
| 2 | Assia | One-Class SVM | counts | 0,363 | à faire |
| 3 | Oumnia | Invariant Mining | counts | 0,915 | à faire |
| 4 | Abdelghany | DeepLog (LSTM) | sequences | 0,96 | à faire |
| 5 | Reda | Autoencodeur LSTM | sequences | pas de repère | en cours |
| 6 | Rida | Arbre de décision (supervisé, référence) | counts | 0,998 | à faire |
| 6 | Rida | Forêt aléatoire (supervisé, référence) | counts | pas de repère | à faire |
| 6 | Rida | Combinaison finale (vote des meilleurs) | | | à faire |

Repères : article Loghub (Zhu et al., ISSRE 2023) pour PCA, Isolation Forest, One-Class SVM, Invariant Mining
et l'arbre ; Du et al. (CCS 2017) pour DeepLog. Ils sont donnés à titre indicatif : leur découpage n'est pas
le nôtre, seuls nos chiffres comptent pour la comparaison. Chacun met à jour sa ligne quand il avance.

## Démarrer

1. Depuis le dossier `hdfs_v1`, créer l'environnement et installer les bibliothèques :
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   (dans Git Bash : `source .venv/Scripts/activate`). Abdelghany et Reda installent aussi PyTorch :
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. Télécharger depuis le Drive `sequences.parquet` et `counts.parquet` et les poser dans `hdfs_v1/data/`
   (`templates.csv` et `MANIFEST.txt` y sont déjà, via Git).
3. Travailler dans son dossier `hdfs_v1/members/<prenom>/`.

## Contenu du dossier

```
hdfs_v1/
  data/                    fichiers communs (les .parquet viennent du Drive)
  members/<prenom>/        code et notebooks de chaque membre
  results/                 prédictions sur test, summary.csv, classement.md
  scripts/
    preparer_donnees.py    fabrique les fichiers communs à partir des données brutes (une seule fois)
    classement.py          classement des modèles à partir de summary.csv
  donnees.py               fonctions pour charger les données de la même façon partout
  evaluate.py              le script de score commun
  requirements.txt
```

## Les fichiers communs

| Fichier | Une ligne = | Colonnes | Pour qui |
|---|---|---|---|
| `counts.parquet` | un bloc | block_id, label (0 normal, 1 anomalie), split, E1 à E30, n_warn, n_error, length, n_unique, duration_s | Mouad, Assia, Oumnia, Rida |
| `sequences.parquet` | un bloc | block_id, label, split, events (liste ordonnée des événements, de 1 à 30) | Abdelghany, Reda |
| `templates.csv` | un événement | num, texte (par exemple 5 = « Receiving block ... ») | tout le monde, pour expliquer les alertes |
| `MANIFEST.txt` | | blocs et anomalies par partie, empreintes MD5 | vérifier qu'on a tous les mêmes fichiers |

Les blocs sont rangés par ordre d'apparition dans HDFS.log, puis coupés : 70 % train, 15 % val, 15 % test.
La colonne `split` est déjà remplie : personne ne refait son propre découpage.

## Règles communes

1. Modèles non supervisés : entraînement sur `train` avec `label == 0` uniquement.
2. Seuil d'alerte : quantile 99,5 % des scores des blocs normaux de `val`.
3. Les réglages se choisissent selon le F1 sur `val`.
4. Graine aléatoire 42 partout.
5. Toute transformation (TF-IDF, StandardScaler...) s'apprend sur `train` seulement.
6. Le test se passe une seule fois, réglages figés, et se note uniquement avec `evaluate.py`.
7. Personne ne modifie `evaluate.py` ni les fichiers de `data/` sans prévenir le groupe.

## Exemple de code (modèle à compteurs)

```python
import sys; sys.path.append("../..")      # depuis hdfs_v1/members/<prenom>/
from donnees import charger_compteurs, parties, seuil, ecrire_predictions, COLONNES_E

df = charger_compteurs()
p = parties(df)          # train, train_normal, val, val_normal, test

model.fit(p["train_normal"][COLONNES_E].values)

s = seuil(-model.score_samples(p["val_normal"][COLONNES_E].values))
score_test = -model.score_samples(p["test"][COLONNES_E].values)
ecrire_predictions(p["test"], score_test, s, "../../results/isolation_forest_test.csv")
```

Pour les séquences : `charger_sequences()` à la place de `charger_compteurs()`.

## Noter son modèle

Le fichier de prédictions contient une ligne par bloc de test, avec les colonnes `block_id,score,pred`
(score plus haut = moins normal ; pred = 1 si alerte). Depuis le dossier `hdfs_v1` :

```
python evaluate.py --pred results/isolation_forest_test.csv --model "Isolation Forest" --member Assia --train-time 2.5 --pred-time 1.1 --machine "PC i5 16 Go"
python scripts/classement.py
```

`evaluate.py` vérifie le fichier, affiche précision, rappel, F1 et matrice de confusion, puis ajoute une ligne
à `results/summary.csv`. Il refuse de noter deux fois le même modèle du même membre.
Mesure principale : le F1 (pas l'exactitude : avec 2,9 % d'anomalies, répondre toujours « normal » donnerait 97 %).
Après avoir noté son modèle : commiter `results/<modele>_test.csv`, `results/summary.csv`, et mettre à jour
le classement ci-dessus.

## Ce que chaque membre rend

- un notebook propre dans `members/<prenom>/` : chargement, entraînement, réglages, test ;
- `results/<modele>_test.csv` et sa ligne dans `results/summary.csv` ;
- 3 anomalies bien trouvées et 3 ratées, avec une explication ;
- une phrase : garder ce modèle dans LOGsentinel ou non, et pourquoi ;
- une page pour le rapport : principe, schéma, résultats.

## Refaire les fichiers communs (Reda)

Avec les données brutes extraites dans `data/HDFS_v1/` à la racine du dépôt, depuis le dossier `hdfs_v1` :

```
python scripts/preparer_donnees.py
```
