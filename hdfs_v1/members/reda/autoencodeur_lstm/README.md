# Autoencodeur LSTM sur HDFS_v1 (Reda)

Un encodeur LSTM compresse la séquence d'événements d'un bloc en un petit vecteur, puis un décodeur
LSTM essaie de la reconstruire. Le modèle n'apprend que sur des blocs normaux : il reconstruit bien le
normal et mal l'anormal. Le score d'un bloc est son erreur moyenne de reconstruction.

## Résultat

| Partie | Précision | Rappel | F1 |
|---|---|---|---|
| val | 0.847 | 0.874 | 0.860 |
| test (evaluate.py) | 0.534 | 0.685 | 0.600 |

Matrice de confusion sur test : VP 764, FP 667, FN 352, VN 84477.
Réglage retenu : embedding 32, LSTM 128, vecteur latent 32, longueur L = 35 (99e centile des longueurs).
Seuil : 0.0754 (quantile 99,5 % des scores des blocs normaux de val).
Entraînement : 19 époques, 102 minutes sur CPU ; prédiction du test : 24 s.

Les données utilisées sont les fichiers communs : le fichier de travail avait la même empreinte MD5
que `sequences.parquet` (069152e56a4e0da97d945b7aa6048fbf). La mention « données provisoires » dans
`results/rapport.md` date d'avant le partage du fichier commun et ne change rien aux chiffres.

À noter : l'écart entre val (0.86) et test (0.60) vient surtout des blocs très courts. Les anomalies
ratées les plus nettes sont des blocs de 2 événements (E22 E5) que le modèle reconstruit parfaitement.

## Contenu

- `src/prepare_data.py`, `src/train_tune.py`, `src/final_test.py`, `src/common.py` : préparation,
  comparaison des 6 réglages sur val, entraînement final et test
- `run_all.bat` : lance les trois étapes dans l'ordre (crée l'environnement et installe PyTorch CPU)
- `results/rapport.md` : résumé complet (réglages, mesures, blocs les plus mal reconstruits, anomalies trouvées et ratées)
- `results/reglages_val.csv`, `results/mesures.json`, `results/meilleur_reglage.json`, `results/entrainement_final.json`
- `results/figures/` : longueurs des séquences, courbe de perte, distribution des scores sur val
- les prédictions sur test sont dans `hdfs_v1/results/autoencodeur_lstm_test.csv`

Le modèle entraîné (`autoencoder_best.pt`, 0,7 Mo) n'est pas dans Git (règle du .gitignore).

## Relancer

Poser `sequences.parquet` dans `data/` de ce dossier (ou adapter `DATA` dans `src/common.py`), puis
lancer `run_all.bat`. Durée : 1 h 30 à 3 h sur un portable sans carte graphique.
