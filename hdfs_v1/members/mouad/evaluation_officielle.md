# Évaluations officielles — Mouad (HDFS_v1)

Commandes lancées depuis `hdfs_v1`, une seule fois par modèle, sans `--force`.

```bash
python evaluate.py --pred results/comptage_warn_test.csv --model "Comptage des WARN" --member Mouad --train-time 0.0 --pred-time 0.238752 --machine "Intel64 Family 6 Model 142 Stepping 10, GenuineIntel"
python evaluate.py --pred results/pca_test.csv --model "PCA" --member Mouad --train-time 0.009157 --pred-time 0.414055 --machine "Intel64 Family 6 Model 142 Stepping 10, GenuineIntel"
```

## Comptage des WARN

```text
Mouad | Comptage des WARN | 86260 blocs de test, 1116 anomalies
précision 1.0000   rappel 0.1801   F1 0.3052

matrice de confusion       prédit normal   prédit alerte
  réellement normal                85144               0
  réellement anomalie                 915             201

ligne ajoutée à results/summary.csv
```

## PCA

```text
Mouad | PCA | 86260 blocs de test, 1116 anomalies
précision 0.8864   rappel 1.0000   F1 0.9398

matrice de confusion       prédit normal   prédit alerte
  réellement normal                85001             143
  réellement anomalie                   0            1116

ligne ajoutée à results/summary.csv
```

Les lignes correspondantes sont dans `results/summary.csv` (horodatage 2026-11-23 07:05 et 07:06).
