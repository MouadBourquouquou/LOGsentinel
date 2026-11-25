# DeepLog (LSTM) sur HDFS_v1 — Abdelghany

## C'est quoi le modèle ?

DeepLog est un modèle qui apprend **l'ordre normal des événements** dans un bloc HDFS.
Un bloc est une suite d'événements, par exemple : `5, 5, 22, 11, 9, 11, 9 ...`
Il utilise un réseau de neurones appelé **LSTM**, qui lit une suite dans l'ordre, comme les mots d'une phrase.

## Comment ça fonctionne ?

1. Je lui montre uniquement des blocs **normaux**.
2. Il apprend à deviner **quel événement vient ensuite**, à partir des 8 derniers événements.
3. Sur un nouveau bloc, il essaie de deviner chaque événement.
4. Si l'événement réel n'est pas dans ses 7 meilleures réponses, c'est une **surprise**.
5. Le **score du bloc = nombre de surprises**.
6. Si le score dépasse un **seuil**, le bloc est une alerte (anomalie).

Plus il y a de surprises, plus le bloc est anormal.

Réglage choisi : fenêtre = 8, taille cachée = 64, top_k = 7, 5 epochs (meilleur F1 sur val : 0,8125).

## Le rôle de chaque partie du notebook

**0. Imports et configuration**
Charger les bibliothèques, fixer la graine 42 et les constantes.

**1. Charger sequences.parquet**
Lire les données : une ligne par bloc, avec la liste de ses événements.

**2. Choisir les données de chaque étape**
Séparer les blocs : train_normal pour apprendre, val pour choisir les réglages, val_normal pour le seuil, test pour la note finale.
Je supprime aussi les séquences en double (389 427 blocs deviennent 13 751 séquences), pour aller plus vite.

**3. Fenêtres glissantes**
Transformer chaque séquence en exemples « 8 événements → événement suivant ».

**4. Le modèle LSTM**
Définir le réseau (embedding, LSTM, couche de sortie) et la fonction qui l'entraîne.

**5. Score d'anomalie**
Calculer, pour chaque bloc, son nombre de surprises.

**6. Tester plusieurs réglages**
Essayer plusieurs configurations (fenêtre, taille, top_k) et calculer le F1 de chacune sur val.

**7. Choisir le meilleur réglage**
Garder la configuration avec le meilleur F1 sur val.

**8. Modèle final et seuil**
Réentraîner avec le meilleur réglage, puis calculer le seuil avec les blocs normaux de val.

**9. Test**
Noter les blocs de test une seule fois et écrire results/deeplog_test.csv.

## Résultats du test (à remplir)

| Mesure | Valeur |
|---|---|
| Précision | |
| Rappel | |
| F1 | |
| Temps d'entraînement | |
| Temps de prédiction | |

## Analyse (à remplir)

- 3 anomalies bien trouvées :
- 3 anomalies ratées :
- Garder DeepLog dans LogSentinel ou non, et pourquoi :