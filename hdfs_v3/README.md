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

## État

Les fichiers communs de l'étape 2 sont en préparation (Reda). Personne ne prépare ses propres données en
attendant : tout le monde doit utiliser exactement les mêmes fichiers. Ce README sera complété (colonnes,
chargement, format des prédictions, commande de score) quand ils seront poussés.

## Comment les étapes 1 et 2 s'enchaînent

1. Les logs arrivent et sont regroupés par bloc.
2. Les modèles de l'étape 1 décident si le bloc est normal ou anormal. S'il est normal, on s'arrête là.
3. S'il est anormal, on lit le trace_id présent dans ses lignes de log et on récupère la trace.
4. Les modèles de l'étape 2 lisent la trace et donnent le type de panne avec un niveau de confiance.
5. L'alerte contient le bloc, ce qui était anormal et la cause probable.

HDFS_v1 et HDFS_v3 viennent de deux clusters différents : ils n'ont aucun trace_id en commun. Pour la
démonstration, le lien est simulé (un trace_id ajouté à quelques blocs anormaux, relié à une trace du type
de panne correspondant). Il sert à montrer que la chaîne fonctionne, pas à calculer un score.

## Répartition

| # | Membre | Modèle(s) | Type | État |
|---|---|---|---|---|
| 1 | Assia | PCA par type de panne | repris de l'étape 1, adapté | à faire |
| 1 | Assia | XGBoost | supervisé | à faire |
| 2 | Mouad | One-Class SVM par type de panne | repris de l'étape 1, adapté | à faire |
| 2 | Mouad | SVM multiclasse | supervisé | à faire |
| 2 | Mouad | k plus proches voisins | supervisé | à faire |
| 3 | Oumnia | Invariant Mining par type de panne | repris de l'étape 1, adapté | à faire |
| 3 | Oumnia | Régression logistique | supervisé | à faire |
| 3 | Oumnia | Arbre de décision | supervisé | à faire |
| 4 | Abdelghani | LSTM/GRU sur la suite des étapes | supervisé, séquence | à faire |
| 4 | Abdelghani | MLP | supervisé | à faire |
| 5 | Rida | Réseau de neurones sur graphe (GNN) | supervisé, graphe | à faire |
| 6 | Reda | Random Forest | supervisé | à faire |
| 6 | Reda | Chaîne finale (détection, lien trace_id, diagnostic) | intégration | à faire |
| 6 | Reda | Fichiers communs et evaluate.py de l'étape 2 | commun | en cours |

Les trois meilleurs modèles de l'étape 1 (One-Class SVM, PCA, Invariant Mining) sont repris pour leur
méthode, pas pour leurs fichiers entraînés : ceux-ci ont appris des événements de logs (E1 à E30) et
répondent à normal ou anormal. Ici, on entraîne un modèle par type de panne sur les traces de ce type, et une
nouvelle trace est attribuée au type dont le modèle lui correspond le mieux.

Le détail du travail de chaque membre est dans son dossier `members/<prenom>/readme.md`.

## Règles communes

1. Découpage fourni : 70 % train, 15 % val, 15 % test, stratifié par type de panne. Personne ne le refait.
2. Graine aléatoire 42 partout.
3. Entraînement sur `train`, réglages choisis sur `val`, un seul passage sur `test` à la fin.
4. Toute transformation (StandardScaler...) s'apprend sur `train` seulement.
5. Les types de panne n'ont pas tous le même nombre de traces : poids de classe (`class_weight="balanced"`
   ou équivalent) quand le modèle le permet.
6. Mesure principale : le F1 macro (moyenne des F1 de chaque type de panne, chaque type compte autant).
   On note aussi le F1 par type, la matrice de confusion, le temps d'entraînement et le temps de prédiction.
   Pas d'exactitude : un modèle qui répond toujours le type le plus fréquent aurait un bon score sans rien comprendre.
7. Le score officiel se calcule uniquement avec `evaluate.py` de l'étape 2.
8. Les modèles entraînés (.joblib, .pkl, .pt) ne vont pas dans Git. Ceux retenus à la fin vont sur le Drive,
   dans `modeles/hdfs_v3/`, nommés `hdfs_v3_<modele>_<prenom>.<extension>`.

## Contenu du dossier

```
hdfs_v3/
  data/                    fichiers communs (à venir ; les gros fichiers viendront du Drive)
  members/<prenom>/        code et notebooks de chaque membre
  results/                 prédictions sur test, synthèse et classement
```

## Ce que chaque membre rend

- un notebook par modèle dans `members/<prenom>/`, qui tourne du début à la fin avec la graine 42 ;
- le fichier de prédictions sur test de chaque modèle, noté avec `evaluate.py` ;
- un readme court : meilleurs réglages trouvés sur val, scores, temps, machine utilisée, ce qui a posé problème ;
- pour les modèles retenus : le fichier du modèle sur le Drive et une ligne dans le readme qui dit où il est.
