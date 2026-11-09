# Petit outil commun pour charger les fichiers partagés de la même façon partout.
#
# Depuis un notebook dans members/<prenom>/ :
#   import sys; sys.path.append("../..")
#   from donnees import charger_compteurs, charger_sequences, parties, COLONNES_E
#   df = charger_compteurs()
#   p = parties(df)
#   X_train = p["train_normal"][COLONNES_E].values     # modèles non supervisés
#   X_val, y_val = p["val"][COLONNES_E].values, p["val"]["label"].values
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
COLONNES_E = [f"E{i}" for i in range(1, 31)]
COLONNES_OPTION = ["n_warn", "n_error", "length", "n_unique", "duration_s"]
GRAINE = 42
QUANTILE_SEUIL = 0.995


def charger_compteurs(chemin=DATA / "counts.parquet"):
    return pd.read_parquet(chemin)


def charger_sequences(chemin=DATA / "sequences.parquet"):
    df = pd.read_parquet(chemin)
    df["events"] = df["events"].apply(lambda s: [int(e) for e in s])
    return df


def parties(df):
    """Renvoie les parties utilisées par tout le monde."""
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    return {
        "train": train,                                # tout train (Rida, modèles supervisés)
        "train_normal": train[train["label"] == 0],    # entraînement des modèles non supervisés
        "val": val,                                    # choix des réglages (F1)
        "val_normal": val[val["label"] == 0],          # seuil : quantile 99,5 % de leurs scores
        "test": test,                                  # une seule fois, à la fin
    }


def seuil(scores_val_normal):
    """Règle commune : quantile 99,5 % des scores des blocs normaux de val."""
    import numpy as np
    return float(np.quantile(scores_val_normal, QUANTILE_SEUIL))


def ecrire_predictions(test, scores, seuil_val, chemin):
    """Écrit le fichier attendu par evaluate.py : block_id, score, pred."""
    import numpy as np
    scores = np.asarray(scores, dtype=float)
    out = pd.DataFrame({"block_id": test["block_id"].values, "score": scores,
                        "pred": (scores > seuil_val).astype(int)})
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(chemin, index=False)
    return out
