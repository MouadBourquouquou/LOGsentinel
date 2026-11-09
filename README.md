# LOGsentinel

Projet d'intelligence artificielle, Génie Informatique 5e année, ENSA Marrakech.
Équipe : Mouad, Assia, Oumnia, Abdelghany, Reda, Rida.

LOGsentinel lit les logs d'un cluster Hadoop, apprend le comportement normal et lève des alertes expliquées.
Règle des encadrants : nous entraînons nos propres modèles (pas d'appel à une API, pas d'IA générative).

## Où on en est (mise à jour du 9 novembre 2026)

| Élément | État |
|---|---|
| Choix des données (HDFS_v1 puis HDFS_v3) | fait |
| Cahier des charges et plan de travail HDFS_v1 | fait |
| Dépôt Git commun | fait |
| Fichiers communs HDFS_v1 : script de préparation, `evaluate.py`, `donnees.py` | fait (Reda) |
| `sequences.parquet` et `counts.parquet` partagés sur le Drive | en cours (Reda) |
| Entraînement et test des modèles sur HDFS_v1 | en cours, voir le tableau ci-dessous |
| Tableau de comparaison rempli | à faire |
| HDFS_v3 (diagnostic) | après HDFS_v1 |

### Répartition HDFS_v1

| # | Membre | Modèle(s) | Fichier lu | État |
|---|---|---|---|---|
| 1 | Mouad | Comptage des ERROR, PCA | counts.parquet | à faire |
| 2 | Assia | Isolation Forest, One-Class SVM | counts.parquet | à faire |
| 3 | Oumnia | Invariant Mining | counts.parquet | à faire |
| 4 | Abdelghany | DeepLog (LSTM) | sequences.parquet | à faire |
| 5 | Reda | Autoencodeur LSTM | sequences.parquet | en cours |
| 6 | Rida | Arbre de décision, Forêt aléatoire, puis combinaison finale | counts.parquet | à faire |

Chacun met à jour sa ligne quand il avance.

## Organisation du dépôt

```
LOGsentinel/
  data/                    données (les gros fichiers ne passent pas par Git, voir data/readme.md)
  members/<prenom>/        le code et les notebooks de chaque membre
  results/                 les prédictions sur test et summary.csv (le tableau commun)
  scripts/
    preparer_donnees.py    fabrique les fichiers communs (lancé une seule fois)
  donnees.py               fonctions pour charger les données de la même façon partout
  evaluate.py              le script de score commun
  requirements.txt
```

## Démarrer

1. Cloner le dépôt et installer les bibliothèques :
   ```
   git clone https://github.com/MouadBourquouquou/LOGsentinel.git
   cd LOGsentinel
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Abdelghany et Reda installent aussi PyTorch :
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. Télécharger depuis le Drive `sequences.parquet`, `counts.parquet`, `templates.csv` et `MANIFEST.txt`,
   et les poser dans `data/` (détails dans `data/readme.md`).
3. Travailler dans son dossier `members/<prenom>/`.

## Les fichiers communs

| Fichier | Une ligne = | Colonnes | Pour qui |
|---|---|---|---|
| `counts.parquet` | un bloc HDFS | block_id, label (0 normal, 1 anomalie), split, E1 à E30, n_warn, n_error, length, n_unique, duration_s | Mouad, Assia, Oumnia, Rida |
| `sequences.parquet` | un bloc HDFS | block_id, label, split, events (liste ordonnée des événements, de 1 à 30) | Abdelghany, Reda |
| `templates.csv` | un événement | num, texte (par exemple 5 = « Receiving block ... ») | tout le monde, pour expliquer les alertes |
| `MANIFEST.txt` | | nombre de blocs et d'anomalies par partie, empreintes MD5 | pour vérifier qu'on a tous les mêmes fichiers |

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
import sys; sys.path.append("../..")      # depuis members/<prenom>/
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
(score plus haut = moins normal ; pred = 1 si alerte).

```
python evaluate.py --pred results/isolation_forest_test.csv --model "Isolation Forest" --member Assia --train-time 2.5 --pred-time 1.1 --machine "PC i5 16 Go"
```

Le script vérifie le fichier, affiche précision, rappel, F1 et matrice de confusion, puis ajoute une ligne
à `results/summary.csv`. Il refuse de noter deux fois le même modèle du même membre.
Mesure principale : le F1 (pas l'exactitude : avec 2,9 % d'anomalies, répondre toujours « normal » donnerait 97 %).

## Ce que chaque membre rend

- un notebook propre dans `members/<prenom>/` : chargement, entraînement, réglages, test ;
- `results/<modele>_test.csv` et sa ligne dans `results/summary.csv` ;
- 3 anomalies bien trouvées et 3 ratées, avec une explication ;
- une phrase : garder ce modèle dans LOGsentinel ou non, et pourquoi ;
- une page pour le rapport : principe, schéma, résultats.

## Sources

Loghub (github.com/logpai/loghub), données sur Zenodo (zenodo.org/records/8196385) ;
Xu et al., SOSP 2009 (PCA) ; Lou et al., USENIX ATC 2010 (Invariant Mining) ; Du et al., CCS 2017 (DeepLog).
