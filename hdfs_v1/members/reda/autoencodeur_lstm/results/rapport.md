# Autoencodeur LSTM : résultats de la nuit

Données PROVISOIRES (sequences_provisoire.parquet construit depuis le zip Loghub).
Les chiffres officiels viendront de sequences.parquet de Mouad et de evaluate.py.

## Réglages comparés sur val

| essai | latent | hidden | L_nom | L  | epoques | perte_val_normal | seuil   | precision_val | rappel_val | f1_val | temps_min |
| ----- | ------ | ------ | ----- | -- | ------- | ---------------- | ------- | ------------- | ---------- | ------ | --------- |
| 1     | 32     | 64     | p99   | 35 | 20      | 0.01913          | 0.63675 | 0.7873        | 0.5949     | 0.6777 | 19.4      |
| 2     | 16     | 64     | p99   | 35 | 20      | 0.09969          | 1.25327 | 0.7833        | 0.5685     | 0.6588 | 15.5      |
| 3     | 64     | 64     | p99   | 35 | 20      | 0.01456          | 0.74853 | 0.7874        | 0.5869     | 0.6725 | 16.5      |
| 4     | 32     | 128    | p99   | 35 | 20      | 0.01246          | 0.56536 | 0.83          | 0.7733     | 0.8006 | 34.0      |
| 5     | 32     | 64     | p95   | 30 | 20      | 0.01808          | 0.65125 | 0.7961        | 0.6064     | 0.6884 | 13.6      |
| 6     | 32     | 64     | p99.9 | 38 | 20      | 0.02456          | 0.85209 | 0.7468        | 0.4741     | 0.58   | 17.0      |

Réglage retenu : latent 32, hidden 128, L 35 (p99).

## Entraînement final

- époques : 19 ; temps : 102.1 min ; machine : cpu
- seuil (quantile 99,5 % des normaux de val) : 0.07540
- val : F1 0.8602, précision 0.8466, rappel 0.8742

## Test (une seule fois)

- précision 0.5339, rappel 0.6846, F1 0.5999
- matrice de confusion : VP 764, FP 667, FN 352, VN 84477
- temps de prédiction sur le test : 24.3 s
- fichier : results/autoencoder_test.csv

## Les 3 blocs les plus mal reconstruits

### Pire n°1 : blk_-6187013877733126959

- label réel : anomalie ; prédiction : alerte ; score 16.3129 (seuil 0.0754)
- longueur : 16 événements
- séquence identique à un bloc normal de train : non
- origine     : `E5 E5 E5 E22 E13 E13 E13 E11 E9 E11 E9 E11 E9 E26 E26 E26`
- reconstruit : `E5 E5 E5 E22 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23`
- positions mal reconstruites : 12 sur 16
- événements concernés : E9 = [*]Received block[*]of size[*]from[*] ; E11 = [*]PacketResponder[*]for block[*]terminating[*] ; E13 = [*]Receiving empty packet for block[*] ; E26 = [*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]

### Pire n°2 : blk_1612037868548734444

- label réel : anomalie ; prédiction : alerte ; score 16.3129 (seuil 0.0754)
- longueur : 16 événements
- séquence identique à un bloc normal de train : non
- origine     : `E5 E5 E5 E22 E13 E13 E13 E11 E9 E11 E9 E11 E9 E26 E26 E26`
- reconstruit : `E5 E5 E5 E22 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23`
- positions mal reconstruites : 12 sur 16
- événements concernés : E9 = [*]Received block[*]of size[*]from[*] ; E11 = [*]PacketResponder[*]for block[*]terminating[*] ; E13 = [*]Receiving empty packet for block[*] ; E26 = [*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]

### Pire n°3 : blk_4435193277864387351

- label réel : anomalie ; prédiction : alerte ; score 16.3129 (seuil 0.0754)
- longueur : 16 événements
- séquence identique à un bloc normal de train : non
- origine     : `E5 E5 E5 E22 E13 E13 E13 E11 E9 E11 E9 E11 E9 E26 E26 E26`
- reconstruit : `E5 E5 E5 E22 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23`
- positions mal reconstruites : 12 sur 16
- événements concernés : E9 = [*]Received block[*]of size[*]from[*] ; E11 = [*]PacketResponder[*]for block[*]terminating[*] ; E13 = [*]Receiving empty packet for block[*] ; E26 = [*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]

## 3 anomalies trouvées

### Trouvée n°1 : blk_-6187013877733126959

- label réel : anomalie ; prédiction : alerte ; score 16.3129 (seuil 0.0754)
- longueur : 16 événements
- séquence identique à un bloc normal de train : non
- origine     : `E5 E5 E5 E22 E13 E13 E13 E11 E9 E11 E9 E11 E9 E26 E26 E26`
- reconstruit : `E5 E5 E5 E22 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23`
- positions mal reconstruites : 12 sur 16
- événements concernés : E9 = [*]Received block[*]of size[*]from[*] ; E11 = [*]PacketResponder[*]for block[*]terminating[*] ; E13 = [*]Receiving empty packet for block[*] ; E26 = [*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]

### Trouvée n°2 : blk_1612037868548734444

- label réel : anomalie ; prédiction : alerte ; score 16.3129 (seuil 0.0754)
- longueur : 16 événements
- séquence identique à un bloc normal de train : non
- origine     : `E5 E5 E5 E22 E13 E13 E13 E11 E9 E11 E9 E11 E9 E26 E26 E26`
- reconstruit : `E5 E5 E5 E22 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23`
- positions mal reconstruites : 12 sur 16
- événements concernés : E9 = [*]Received block[*]of size[*]from[*] ; E11 = [*]PacketResponder[*]for block[*]terminating[*] ; E13 = [*]Receiving empty packet for block[*] ; E26 = [*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]

### Trouvée n°3 : blk_4435193277864387351

- label réel : anomalie ; prédiction : alerte ; score 16.3129 (seuil 0.0754)
- longueur : 16 événements
- séquence identique à un bloc normal de train : non
- origine     : `E5 E5 E5 E22 E13 E13 E13 E11 E9 E11 E9 E11 E9 E26 E26 E26`
- reconstruit : `E5 E5 E5 E22 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23`
- positions mal reconstruites : 12 sur 16
- événements concernés : E9 = [*]Received block[*]of size[*]from[*] ; E11 = [*]PacketResponder[*]for block[*]terminating[*] ; E13 = [*]Receiving empty packet for block[*] ; E26 = [*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]

## 3 anomalies ratées (les scores les plus bas)

### Ratée n°1 : blk_-8197765855233616828

- label réel : anomalie ; prédiction : normal ; score 0.0000 (seuil 0.0754)
- longueur : 2 événements
- séquence identique à un bloc normal de train : non
- origine     : `E22 E5`
- reconstruit : `E22 E5`
- positions mal reconstruites : 0 sur 2

### Ratée n°2 : blk_1736556131251685830

- label réel : anomalie ; prédiction : normal ; score 0.0000 (seuil 0.0754)
- longueur : 2 événements
- séquence identique à un bloc normal de train : non
- origine     : `E22 E5`
- reconstruit : `E22 E5`
- positions mal reconstruites : 0 sur 2

### Ratée n°3 : blk_-7544736976548371919

- label réel : anomalie ; prédiction : normal ; score 0.0000 (seuil 0.0754)
- longueur : 2 événements
- séquence identique à un bloc normal de train : non
- origine     : `E22 E5`
- reconstruit : `E22 E5`
- positions mal reconstruites : 0 sur 2

## Figures

- results/figures/longueurs.png
- results/figures/courbe_perte.png
- results/figures/scores_val.png
