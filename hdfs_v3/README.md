# Étape 2 : diagnostic sur HDFS_v3

À l'étape 1, les modèles lisent les logs HDFS_v1 et disent si un bloc est normal ou anormal (la détection).
À l'étape 2, on répond à la question suivante : quand il y a un problème, de quel type de panne s'agit-il ?
C'est le diagnostic. Les données sont les traces de HDFS_v3 TraceBench : chaque trace décrit une opération
du cluster sous forme d'étapes (spans), avec leur durée, leur machine, leur statut et le lien parent enfant.

Points importants :

- Les modèles de l'étape 2 font uniquement le diagnostic. La détection reste le travail des modèles de l'étape 1.
- Ils sont entraînés et testés uniquement sur des traces anormales : une trace n'arrive au diagnostic que
  si elle a déjà été détectée comme anormale.
- Tous les modèles sont supervisés : on connaît le type de panne de chaque trace.
- Même règle qu'à l'étape 1 : nos propres modèles, sans API et sans IA générative.

## Comment les étapes 1 et 2 s'enchaînent

1. Les logs arrivent et sont regroupés par bloc.
2. Les modèles de l'étape 1 décident si le bloc est normal ou anormal. S'il est normal, on s'arrête là.
3. S'il est anormal, on lit le trace_id présent dans ses lignes de log et on récupère la trace.
4. Les modèles de l'étape 2 lisent la trace et donnent le type de panne avec un niveau de confiance.
5. L'alerte contient le bloc, ce qui était anormal et la cause probable.

HDFS_v1 et HDFS_v3 viennent de deux clusters différents : ils n'ont aucun trace_id en commun. Pour la
démonstration, le lien est simulé (un trace_id ajouté à quelques blocs anormaux, relié à une trace du type
de panne correspondant). Il sert à montrer que la chaîne fonctionne, pas à calculer un score.

## Classement actuel

F1 macro sur la partie test, calculé uniquement par `evaluate.py`. Mis à jour avec `python scripts/classement.py`.

Aucun modèle noté pour l'instant.

## Répartition

| # | Membre | Modèle(s) | Fichier lu | Type | État |
|---|---|---|---|---|---|
| 1 | Assia | PCA par type de panne | traces | repris de l'étape 1, adapté | à faire |
| 1 | Assia | XGBoost | traces | supervisé | à faire |
| 2 | Mouad | One-Class SVM par type de panne | traces | repris de l'étape 1, adapté | à faire |
| 2 | Mouad | SVM multiclasse | traces | supervisé | à faire |
| 2 | Mouad | k plus proches voisins | traces | supervisé | à faire |
| 3 | Oumnia | Invariant Mining par type de panne | traces | repris de l'étape 1, adapté | à faire |
| 3 | Oumnia | Régression logistique | traces | supervisé | à faire |
| 3 | Oumnia | Arbre de décision | traces | supervisé | à faire |
| 4 | Abdelghani | LSTM/GRU sur la suite des étapes | etapes | supervisé, séquence | à faire |
| 4 | Abdelghani | MLP | traces | supervisé | à faire |
| 5 | Rida | Réseau de neurones sur graphe (GNN) | etapes + aretes | supervisé, graphe | à faire |
| 6 | Reda | Random Forest | traces | supervisé | à faire |
| 6 | Reda | Chaîne finale (détection, lien trace_id, diagnostic) | tous | intégration | à faire |
| 6 | Reda | Fichiers communs et evaluate.py de l'étape 2 | | commun | fait |

Les trois meilleurs modèles de l'étape 1 (One-Class SVM, PCA, Invariant Mining) sont repris pour leur
méthode, pas pour leurs fichiers entraînés : ceux-ci ont appris des événements de logs (E1 à E30) et
répondent à normal ou anormal. Ici, on entraîne un modèle par type de panne sur les traces de ce type, et une
nouvelle trace est attribuée au type dont le modèle lui correspond le mieux.

Le détail du travail de chaque membre est dans son dossier `members/<prenom>/readme.md`.

## Démarrer

1. Faire `git pull`.
2. Depuis le dossier `hdfs_v3`, créer l'environnement et installer les bibliothèques :
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   (dans Git Bash : `source .venv/Scripts/activate`). Abdelghani et Rida installent aussi PyTorch :
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`, et Rida PyTorch Geometric :
   `pip install torch_geometric`.
3. Télécharger depuis le Drive (dossier `LOGsentinel donnees HDFS_v3`) `traces.parquet`, `etapes.parquet` et
   `aretes.parquet`, et les poser dans `hdfs_v3/data/` (`vocabulaire.json` et `MANIFEST.txt` y sont déjà, via Git).
4. Vérifier les empreintes MD5 avec celles de `data/MANIFEST.txt`.
5. Travailler dans son dossier `hdfs_v3/members/<prenom>/`.

## Contenu du dossier

```
hdfs_v3/
  data/                    fichiers communs (les .parquet viennent du Drive)
  members/<prenom>/        code et notebooks de chaque membre
  results/                 prédictions sur test, summary.csv, classement.md, confusion/
  scripts/
    preparer_donnees.py    fabrique les fichiers communs à partir du zip TraceBench (une seule fois)
    classement.py          classement des modèles à partir de summary.csv
  donnees.py               fonctions pour charger les données de la même façon partout
  evaluate.py              le script de score commun (F1 macro)
  requirements.txt
```

## Les données

28 419 traces anormales, 11 types de panne. Une trace est gardée si au moins une de ses étapes a échoué
(même règle que le script de Loghub) ; son type de panne est celui du dossier TraceBench où elle se trouve.

| Label | Type de panne | Ce qui a été injecté | Traces (train / val / test) |
|---|---|---|---|
| 0 | Data_corruptBlk | bloc de données corrompu | 1840 / 395 / 394 |
| 1 | Data_corruptMeta | métadonnées d'un bloc corrompues | 2548 / 546 / 546 |
| 2 | Data_cutBlk | bloc de données tronqué | 2657 / 570 / 569 |
| 3 | Data_cutMeta | métadonnées tronquées | 2630 / 563 / 564 |
| 4 | Data_lossBlk | bloc de données supprimé | 2524 / 540 / 541 |
| 5 | Data_lossMeta | métadonnées supprimées | 78 / 16 / 17 |
| 6 | Net_disconnectDN | DataNode déconnecté du réseau | 2570 / 551 / 550 |
| 7 | Proc_killDN | processus DataNode tué | 3060 / 656 / 656 |
| 8 | Proc_suspendDN | processus DataNode suspendu | 1059 / 227 / 227 |
| 9 | Sys_deadDN | machine DataNode arrêtée | 410 / 88 / 88 |
| 10 | Sys_panicDN | machine DataNode en panique système | 517 / 111 / 111 |

Ce qui n'est pas gardé, et pourquoi (le détail est dans `data/MANIFEST.txt`) :

- les dossiers normaux (NM) et les pannes combinées (COM), dont le type exact par trace n'est pas connu ;
- les dossiers où aucune panne n'a été injectée (0 DataNode en panne, retard de 0 ms) ;
- les pannes de lenteur (Net_slowDN, Net_slowHDFS) : elles ralentissent le cluster sans faire échouer
  aucune étape, donc aucune trace n'est marquée anormale par la règle ci-dessus ;
- Sys_readOnlyDN : seulement 10 traces anormales, trop peu pour apprendre et tester.

Attention : Data_lossMeta n'a que 17 traces de test, son F1 bouge beaucoup d'un modèle à l'autre. Les paires
les plus dures à séparer sont Data_corruptMeta / Data_cutMeta et Sys_deadDN / Sys_panicDN (mêmes erreurs).

## Les fichiers communs

| Fichier | Une ligne = | Contenu | Pour qui |
|---|---|---|---|
| `traces.parquet` | une trace | trace_id, label, panne, split, métadonnées, et 90 caractéristiques résumées | tout le monde sauf le GNN |
| `etapes.parquet` | une trace | trace_id, label, split, et 6 listes de même longueur (une case par étape, dans l'ordre) | Abdelghani, Rida |
| `aretes.parquet` | une arête | trace_id, split, src, dst, type (0 = appel, 1 = suite dans le même fil) | Rida |
| `vocabulaire.json` | | noms des types de panne, des opérations, des agents, des rôles, des erreurs, liste des caractéristiques | tout le monde |
| `MANIFEST.txt` | | traces par type et par partie, choix faits, empreintes MD5 | vérifier qu'on a tous les mêmes fichiers |

Les 90 caractéristiques de `traces.parquet` (liste exacte : `colonnes_caract()`) :

- taille et forme : `n_etapes`, `n_machines`, `n_datanodes`, `n_fils`, `n_aretes_appel`, `ecriture` (1 si écriture) ;
- durées : `duree_moy_ms`, `duree_max_ms`, `duree_totale_ms`, `duree_trace_ms` ;
- échecs : `n_echecs`, `part_echecs`, `n_types_erreur`, `pos_premier_echec` (0 = au début, 1 = à la fin),
  `echecs_client`, `echecs_namenode`, `echecs_datanode` ;
- `op_...` : nombre d'étapes de chaque opération (34 opérations) ;
- `err_01` à `err_39` : nombre d'étapes avec chaque type d'erreur (texte dans `vocabulaire.json`, liste `erreurs`).

Ne jamais donner aux modèles les colonnes `scenario`, `panne`, `categorie`, `charge`, `titre`, `split` :
`scenario` et `panne` contiennent la réponse. La liste est dans `COLONNES_META` de `donnees.py`.

Les listes de `etapes.parquet` : `ops` (numéro d'opération, de 1 à 34), `agents` (1 à 6), `roles`
(0 client, 1 namenode, 2 datanode), `machines` (numéro de machine à l'intérieur de la trace), `durees_ms`,
`erreurs` (0 = étape réussie, sinon numéro du type d'erreur). Le numéro 0 est libre pour `ops` et `agents` :
il sert au remplissage quand on coupe ou complète les suites à une même longueur.

## Règles communes

1. Découpage fourni : 70 % train, 15 % val, 15 % test, stratifié par type de panne. Personne ne le refait.
2. Graine aléatoire 42 partout.
3. Entraînement sur `train`, réglages choisis selon le F1 macro sur `val`, un seul passage sur `test` à la fin.
4. Toute transformation (StandardScaler...) s'apprend sur `train` seulement.
5. Les types de panne n'ont pas tous le même nombre de traces : poids de classe (`class_weight="balanced"`,
   ou `poids_classes()` de `donnees.py` pour PyTorch et XGBoost).
6. Mesure principale : le F1 macro (moyenne des F1 de chaque type de panne, chaque type compte autant).
   On note aussi le F1 par type, la matrice de confusion, le temps d'entraînement et le temps de prédiction.
   Pas d'exactitude : un modèle qui répond toujours le type le plus fréquent aurait un bon score sans rien comprendre.
7. Le score officiel se calcule uniquement avec `evaluate.py` de l'étape 2.
8. Personne ne modifie `evaluate.py` ni les fichiers de `data/` sans prévenir le groupe.
9. Les modèles entraînés (.joblib, .pkl, .pt) ne vont pas dans Git. Ceux retenus à la fin vont sur le Drive,
   dans le dossier `LOGsentinel modeles HDFS_v3` (à créer à côté de `LOGsentinel modeles HDFS_v1`), nommés
   `hdfs_v3_<modele>_<prenom>.<extension>`.

## Exemple de code (modèle sur les caractéristiques)

```python
import sys; sys.path.append("../..")      # depuis hdfs_v3/members/<prenom>/
from donnees import charger_traces, parties, colonnes_caract, ecrire_predictions
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

df = charger_traces()
p = parties(df)                          # train, val, test
C = colonnes_caract()

model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
model.fit(p["train"][C], p["train"]["label"])
print("F1 macro val :", f1_score(p["val"]["label"], model.predict(p["val"][C]), average="macro"))

# à la fin seulement, réglages figés :
ecrire_predictions(p["test"], model.predict(p["test"][C]), "../../results/random_forest_test.csv",
                   proba=model.predict_proba(p["test"][C]))
```

Pour les suites d'étapes : `charger_etapes()`. Pour les graphes : `charger_aretes()` puis `aretes_par_trace()`,
qui donne pour chaque trace le tableau 2 x n_arêtes (src, dst) attendu par PyTorch Geometric.

## Noter son modèle

Le fichier de prédictions contient une ligne par trace de test, avec les colonnes `trace_id,pred`
(pred = le label prédit, de 0 à 10). Les colonnes `proba_0` à `proba_10` sont conseillées : la chaîne finale
s'en sert pour le niveau de confiance. Depuis le dossier `hdfs_v3` :

```
python evaluate.py --pred results/random_forest_test.csv --model "Random Forest" --member Reda --train-time 0.2 --pred-time 0.1 --machine "PC i5 16 Go"
python scripts/classement.py
```

`evaluate.py` vérifie le fichier, affiche le F1 macro, le F1 par type et la matrice de confusion, garde la
matrice dans `results/confusion/`, puis ajoute une ligne à `results/summary.csv`. Il refuse de noter deux fois
le même modèle du même membre. Après avoir noté son modèle : commiter `results/<modele>_test.csv`,
`results/summary.csv`, `results/confusion/<modele>_test.csv`, et mettre à jour le classement ci-dessus.

## Ce que chaque membre rend

- un notebook par modèle dans `members/<prenom>/`, qui tourne du début à la fin avec la graine 42 ;
- le fichier de prédictions sur test de chaque modèle, noté avec `evaluate.py` ;
- un readme court : meilleurs réglages trouvés sur val, scores, temps, machine utilisée, ce qui a posé problème ;
- pour les modèles retenus : le fichier du modèle sur le Drive et une ligne dans le readme qui dit où il est.

## Refaire les fichiers communs (Reda)

Avec `HDFS_v3_TraceBench.zip` dans `data/` à la racine du dépôt (pas besoin de l'extraire), depuis `hdfs_v3` :

```
python scripts/preparer_donnees.py
```

Environ une minute. S'il est interrompu, il reprend là où il s'était arrêté.
