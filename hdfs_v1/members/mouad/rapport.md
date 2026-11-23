# Rapport Mouad — HDFS_v1

## 1. Livrables

- `members/mouad/comptage_warn.ipynb` : baseline sans apprentissage.
- `members/mouad/pca.ipynb` : PCA non supervisée, expériences et modèle final.
- `results/comptage_warn_test.csv`, `results/pca_test.csv` : 86 260 lignes chacun, colonnes `block_id,score,pred`.
- `members/mouad/mesures_baseline.json`, `mesures_pca.json`, `resultats_val_pca.csv`, `figures/` : mesures réelles, tableau des essais et figures.
- `members/mouad/evaluation_officielle.md` : sorties des deux notations officielles.

## 2. Méthodologie

### Comptage des WARN (baseline)

Score = `n_warn`, alerte si `n_warn > 0`, seuil fixe 0. Aucun entraînement, aucune transformation, aucune utilisation des étiquettes. Le fichier de prédictions a été vérifié : identifiants uniques et conformes à `test`, scores égaux à `n_warn`, `pred = (score > 0)`.

### PCA

Caractéristiques : `COLONNES_E` (`E1` à `E30`) de `counts.parquet`. La PCA est ajustée uniquement sur les 389 427 blocs `train_normal`. Le score est l'erreur de reconstruction moyenne `s(x) = (1/d)‖x − x̂‖²` dans l'espace prétraité. Le seuil est le quantile 99,5 % des scores des 83 652 blocs `val_normal` via `seuil()`. Les réglages sont choisis au F1 sur les 86 259 blocs de `val` ; le test est scoré une seule fois après figer le pipeline.

Quatorze configurations ont été comparées : comptages bruts, centrage-réduction, `log1p` + centrage-réduction, proportions + centrage-réduction, avec 3 à 20 composantes ou un ratio de variance expliquée. Toutes les transformations sont ajustées sur `train_normal` seulement. La SVD complète est déterministe.

Configuration retenue (F1 val maximal) : **comptages bruts, 10 composantes**, variance cumulée 0,9999993, seuil `5.646577812553519e-11`. Sur validation : précision 0,9823, rappel 0,9356, F1 0,9583 (VP 2439, FP 44, FN 168, VN 83608).

Le préprocessing brut est légèrement meilleur que la mise à l'échelle : les comptages fréquents ont déjà des amplitudes comparables et la standardisation amplifie le bruit des événements quasi constants (`E30` est constant, `E17` quasi constant). Les alternatives `log_standard` et `proportion_standard` donnent des F1 val de 0,9408 à 0,9572, donc inférieurs ou égaux au meilleur choix.

## 3. Résultats officiels (evaluate.py, test)

| Modèle | Précision | Rappel | F1 | VP | FP | FN | VN | Entraînement | Prédiction test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Comptage des WARN | 1,0000 | 0,1801 | 0,3052 | 201 | 0 | 915 | 85 144 | 0 min | 0,2388 s |
| PCA | 0,8864 | 1,0000 | 0,9398 | 1116 | 143 | 0 | 85 001 | 0,009157 min (0,549 s) | 0,4141 s |

La PCA augmente le F1 de 0,6346 par rapport au baseline. Elle détecte les 1 116 anomalies, au prix de 143 faux positifs (0,17 % des blocs normaux). Le baseline n'a aucun faux positif, mais rate 915 anomalies : seules 18,0 % des anomalies de test contiennent au moins un WARN, contre 56,0 % en validation, ce qui explique son effondrement entre val (F1 0,7180) et test (F1 0,3052).

Matrice de confusion — vrais positifs : anomalies alertées ; faux positifs : normaux alertés ; faux négatifs : anomalies non alertées ; vrais négatifs : normaux non alertés.

## 4. Exemples d'anomalies détectées par la PCA

Seuil PCA = 5,65e-11.

1. `blk_-2667098508999319871` — score 0,833337, `n_warn = 0`, longueur 13. Événements : E5=3 (Receiving block), E7=3 (writeBlock a reçu une exception), E14=2 (Exception in receiveBlock), E10=1 (PacketResponder Exception), E8=1 (PacketResponder interrompu), E22=1 (allocateBlock). Vrai positif PCA, faux négatif du baseline : la multiplication d'erreurs d'écriture/réception est très éloignée du sous-espace normal, sans aucun WARN explicite.
2. `blk_3321514261845209650` — score 0,666697, `n_warn = 0`, longueur 2. Événements : E5=1 (Receiving block), E22=1 (allocateBlock). Vrai positif PCA, faux négatif du baseline : bloc très court dont la combinaison E5/E22 est atypique dans les comptages normaux.
3. `blk_1384136409987254471` — même profil E5=1, E22=1, score 0,666697. Vrai positif PCA, faux négatif du baseline.

## 5. Anomalies manquées

La PCA n'a **aucun faux négatif** sur le test (rappel 1,0000) : les trois anomalies manquées par la PCA ne sont donc pas disponibles. Les trois exemples ci-dessous sont des anomalies manquées par le **baseline WARN** mais détectées par la PCA, choisies parmi les scores PCA les plus proches du seuil :

1. `blk_-5546573378168675500` — score PCA 3,71e-07, `n_warn = 0`, longueur 26. E5=4, E21=4 (Deleting block), E26=4 (addStoredBlock), E9/E11/E23=3. Le profil mêle réception, suppression et mise à jour de réplicas ; l'absence de WARN suffit au baseline pour le classer normal.
2. `blk_-8004652959868317246` — même profil d'événements, durée 3 165 s, score 3,71e-07. Faux négatif du baseline, vrai positif PCA.
3. `blk_-2569780052298721820` — même profil, durée 2 561 s, score 3,71e-07. Faux négatif du baseline, vrai positif PCA.

Ces trois blocs ont des comptages très proches de la norme (scores juste au-dessus du seuil), ce qui illustre une limite de l'approche : les anomalies dont la distribution d'événements est quasi normale ne seraient pas détectées si le quantile de seuil était plus élevé.

Les 143 faux positifs PCA sont surtout des blocs atypiques mais étiquetés normaux, par exemple `blk_-9204682504804436825` (score 4,01e-05, E3=203 « Served block », E5/E21/E26=11, longueur 273). La PCA réagit à l'ampleur de ces comptages, pas à une erreur logique.

## 6. Synthèse et recommandation

Garder la PCA pour LOGsentinel : elle combine un entraînement très rapide (0,55 s), une inférence rapide (0,41 s pour 86 260 blocs), un rappel parfait et un F1 de 0,9398, largement au-dessus du baseline. Le baseline reste utile comme repère interprétable et sans coût, mais son faible rappel sur le test (0,1801) le rend insuffisant seul.

Limites : la PCA est linéaire, ignore l'ordre des événements (contrairement aux modèles sur `sequences.parquet`) et produit des faux positifs sur des blocs normaux aux comptages extrêmes. Le seuil quantile 99,5 % est imposé par le projet et n'a pas été optimisé sur le test.
